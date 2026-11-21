"""Per-request activation steering for DeepSeek V4.1 on vLLM, pipeline-parallel aware.

One set of stock weights serves stock and abliterated behaviour. Each request carries
a scale s in [0, 1]; 0 leaves every bit of the stock computation alone, 1 applies the
sidecar in full. Two outputs of every steered layer are edited, the attention module's
and the shared expert's, each by its own operator:

    x' = (x + s * write * (read . x)) * (1 + s * (row_gain - 1))

Two sidecar formats are read:

  published (v1)   layers.{L}.{alpha, direction, attn_row_gain}, layers 4..36: the file
                   of NVIDIA/hyperflare-inference PR #404. Both sites of a layer use
                   read = direction and write = -alpha * direction; the row gain
                   applies to the attention output only. Same arithmetic as that PR's
                   ignition/recipes/deepseek41/steering.py.
  derived (v2)     layers.{L}.{attn,shared}.{read, write[, row_gain]}, FP32, written by
                   verify/sidecar_vs_weights.py --derive from a stock and an abliterated
                   checkpoint: the operators that rebuild the abliterated one. Read and
                   write may differ (an abliteration that preserves row norms is such an
                   operator). A file whose metadata does not say it rebuilt the
                   checkpoint is refused.

Nothing in vLLM is replaced on disk. `install()` patches four seams when the modules
that hold them are imported, in whichever process imports them:

  worker (every PP rank)  vllm.v1.worker.gpu.model_states.resolve_model_state_cls
                          -> a DeepseekV41ModelState subclass that keeps the scale per
                             request, fills a per-token buffer every step, and hooks the
                             attention and shared-expert modules of the layers this
                             rank owns
  engine core             vllm.v1.request.Request.from_engine_core_request
                          -> the scale goes into cache_salt, so prefix-cache blocks
                             computed under one scale are never reused under another
  API server              ChatCompletionRequest/CompletionRequest.to_sampling_params
                          -> validates the scale (HTTP 400) and accepts it as a
                             top-level field as well as in vllm_xargs
  model                   DeepseekV4Model._decoder_replay_supported -> False, only
                          with PP2_STEERING_REPLAY=off. Decoder bounded replay (0.33+)
                          runs the layers past the last KV source on each request's
                          last window of rows in eager prefill steps. "off" keeps those
                          layers on the whole batch, which costs prefill on that stage;
                          "keep" leaves upstream's replay alone and gives the replay
                          batch a scale buffer of its own.

A seam that cannot be patched ends the process (exit status 78) instead of raising:
an exception inside sitecustomize or inside somebody's try/except around an import
would leave an engine that boots and serves every request stock.

Configuration is by environment, set by recipe/serve_vllm_node.sh on every rank:

  PP2_STEERING_SIDECAR        path of the sidecar inside the container (required)
  PP2_STEERING_SHA256         its sha256 (required; a mismatch refuses to start)
  PP2_STEERING_DEFAULT_SCALE  scale for requests that do not ask (default 0)
  PP2_STEERING_DEBUG          1 -> accept the per-request `steering_ranks` bitmask,
                              which limits steering to the PP ranks whose bit is set.
                              For test boots only.
  PP2_STEERING_ALLOW_TP       1 -> do not refuse tensor parallel > 1. Both edits are
                              linear in the hooked tensor, so they commute with the TP
                              sum whether a hook sees the reduced tensor or one rank's
                              partial; that is an argument, not a measurement. For a
                              test boot on a box that cannot run a pipeline stage.
  PP2_STEERING_ALLOW_UNVERIFIED  1 -> load a derived sidecar whose metadata does not
                              say REBUILT. Experiments only.
  PP2_STEERING_REPLAY         off (default) | keep: see the model seam above. "keep"
                              refuses a stage that runs its replay batch in CUDA graphs
                              of its own (a single-stage lane); a PP2 pair has none.
  PP2_STEERING_KERNEL         torch (default) | fused: one Triton kernel per steered
                              output instead of 6 to 8 torch kernels, and stock rows
                              cost nothing (pp2_steering_kernel.py). Steered values can
                              differ from the torch path's in the last bit; stock rows
                              cannot.

Request API: `"vllm_xargs": {"steering_scale": 0.0..1.0}`, or `"steering_scale"` at the
top level of a chat or completion request.

Status: the defaults (replay off, torch kernel) booted and steered on the PP2 pair on
2026-11-20 with the published sidecar; what the probe showed there is in README.md.
"keep" and "fused" are unit-tested on CPU (test_pp2_steering.py, the fused kernel in
Triton's interpreter) and against the real v0.34.0 classes
(verify/steering_image_smoke.py), and have never run on a GPU.
verify/steering_probe.py is the acceptance test on a lane.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Callable

SCALE_KEY = "steering_scale"
RANKS_KEY = "steering_ranks"
SIDECAR_LAYERS = range(4, 37)  # the published (v1) sidecar's layers
SIDECAR_NAMES = ("alpha", "attn_row_gain", "direction")
V2_FORMAT = "pp2-steering-v2"
V2_KEY = r"layers\.(\d+)\.(attn|shared)\.(read|write|row_gain)"
SITES = ("attn", "shared")  # the two hooked outputs of a layer
REPLAY_MODES = ("off", "keep")
KERNELS = ("torch", "fused")
STATE_BASE = ("vllm.models.deepseek_v41.nvidia.model_state", "DeepseekV41ModelState")
MODEL_MODULE = "vllm.models.deepseek_v41.nvidia.model"


def _log(message: str) -> None:
    sys.stderr.write(f"[pp2_steering] {message}\n")
    sys.stderr.flush()


def _fatal(message: str) -> None:
    """Stop this process. Tests replace it with a function that raises."""
    _log(f"FATAL: {message}")
    os._exit(78)


def _client_error(message: str, parameter: str, value: Any) -> Exception:
    """The exception the API server turns into HTTP 400."""
    try:
        from vllm.exceptions import VLLMValidationError
    except Exception:  # an older or stand-in vLLM: ValueError is a 400 as well
        return ValueError(message)
    return VLLMValidationError(message, parameter=parameter, value=value)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Config:
    sidecar_path: str
    sha256: str
    default_scale: float
    debug: bool
    allow_tp: bool = False
    allow_unverified: bool = False
    replay: str = "off"
    kernel: str = "torch"

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Config":
        env = os.environ if environ is None else environ
        path = env.get("PP2_STEERING_SIDECAR", "")
        expected = env.get("PP2_STEERING_SHA256", "").strip().lower()
        if not path:
            raise ValueError("PP2_STEERING_SIDECAR is not set")
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("PP2_STEERING_SHA256 must be the sidecar's 64-hex sha256")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"steering sidecar {path} has sha256 {actual}, expected {expected}"
            )
        default = parse_scale(env.get("PP2_STEERING_DEFAULT_SCALE", "0") or "0", 0.0)
        replay = env.get("PP2_STEERING_REPLAY", "") or "off"
        kernel = env.get("PP2_STEERING_KERNEL", "") or "torch"
        if replay not in REPLAY_MODES:
            raise ValueError(f"PP2_STEERING_REPLAY must be one of {REPLAY_MODES}, got {replay!r}")
        if kernel not in KERNELS:
            raise ValueError(f"PP2_STEERING_KERNEL must be one of {KERNELS}, got {kernel!r}")
        # Refuse an unverified derived sidecar here, at interpreter start, not after the
        # weights have loaded.
        check_verdict(sidecar_metadata(path), env.get("PP2_STEERING_ALLOW_UNVERIFIED", "0") == "1")
        return cls(
            path,
            actual,
            default,
            env.get("PP2_STEERING_DEBUG", "0") == "1",
            env.get("PP2_STEERING_ALLOW_TP", "0") == "1",
            env.get("PP2_STEERING_ALLOW_UNVERIFIED", "0") == "1",
            replay,
            kernel,
        )


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_scale(value: Any, default: float) -> float:
    """A request's scale as a float in [0, 1]. Raises ValueError on anything else."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{SCALE_KEY} must be a number in [0, 1], got a boolean")
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            raise ValueError(f"{SCALE_KEY} must be a number in [0, 1], got {value!r}") from None
    if not isinstance(value, (int, float)):
        raise ValueError(f"{SCALE_KEY} must be a number in [0, 1], got {type(value).__name__}")
    scale = float(value)
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError(f"{SCALE_KEY} must be a number in [0, 1], got {value!r}")
    return scale


def parse_ranks(value: Any) -> int | None:
    """The debug rank bitmask, or None when the request does not carry one."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{RANKS_KEY} must be a non-negative integer bitmask")
    try:
        mask = int(value)
    except ValueError:
        raise ValueError(f"{RANKS_KEY} must be a non-negative integer bitmask") from None
    if mask < 0:
        raise ValueError(f"{RANKS_KEY} must be a non-negative integer bitmask")
    return mask


def requested(params: Any, config: Config) -> tuple[float, int | None]:
    """(scale, rank mask) a request asks for. Never raises: a malformed value that got
    past the API server (another frontend, a direct engine client) is served stock."""
    if params is None:  # pooling requests are never steered
        return 0.0, None
    extra = getattr(params, "extra_args", None) or {}
    try:
        scale = parse_scale(extra.get(SCALE_KEY), config.default_scale)
        mask = parse_ranks(extra.get(RANKS_KEY)) if config.debug else None
    except ValueError as exc:
        _log(f"ignoring malformed steering arguments, serving stock: {exc}")
        return 0.0, None
    return scale, mask


def salted_cache_salt(
    cache_salt: str | None, scale: float, mask: int | None, sha256: str
) -> str | None:
    """cache_salt with the steering state folded in. Scale 0 is the stock namespace and
    is left untouched, so stock requests keep sharing prefix blocks as before."""
    if scale == 0.0:
        return cache_salt
    ranks = "all" if mask is None else str(mask)
    return f"{cache_salt or ''}\x1fpp2steer:{scale:.6f}:{ranks}:{sha256[:16]}"


# --------------------------------------------------------------------------- sidecar


@dataclass(frozen=True)
class SiteSteering:
    """One hooked output: x' = (x + s * write * (read . x)) * (1 + s * gain_delta)."""

    read: Any  # torch.Tensor, FP32[hidden]
    write: Any  # torch.Tensor, FP32[hidden]
    gain_delta: Any = None  # torch.Tensor, FP32[hidden]: row gain minus 1; None = no gain

    @property
    def kernels(self) -> int:
        """Kernels one application adds: the float cast, the dot product, the products,
        the cast back and the select."""
        return 6 if self.gain_delta is None else 8


def sidecar_metadata(path: str) -> dict[str, str]:
    """The safetensors header's metadata. Plain Python, so the configuration check at
    interpreter start does not have to import torch."""
    import json
    import struct

    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        if not 2 <= length <= 100 << 20:
            raise ValueError(f"{path} is not a safetensors file")
        header = json.loads(handle.read(length))
    return dict(header.get("__metadata__") or {})


def check_verdict(metadata: dict[str, str], allow_unverified: bool) -> None:
    """A derived sidecar must say that its operators rebuilt the checkpoint."""
    if metadata.get("format") != V2_FORMAT:
        return
    verdict = metadata.get("verdict", "")
    if not verdict.startswith("REBUILT") and not allow_unverified:
        raise ValueError(
            f"derived steering sidecar was written with verdict {verdict or '<none>'!r}: its "
            "operators did not rebuild the checkpoint they were derived from"
        )


def load_sidecar(
    path: str, device: Any, allow_unverified: bool = False
) -> dict[int, dict[str, SiteSteering]]:
    """layer -> site ("attn", "shared") -> operator, from either sidecar format."""
    from safetensors.torch import load_file

    tensors = load_file(path, device="cpu")
    metadata = sidecar_metadata(path)
    if metadata.get("format") == V2_FORMAT or any(re.fullmatch(V2_KEY, name) for name in tensors):
        return _load_derived(tensors, metadata, device, allow_unverified)
    return _load_published(tensors, device)


def _load_published(tensors: dict[str, Any], device: Any) -> dict[int, dict[str, SiteSteering]]:
    """The published sidecar, validated exactly as the reference loader validates it."""
    import torch

    expected = {f"layers.{layer}.{name}" for layer in SIDECAR_LAYERS for name in SIDECAR_NAMES}
    if set(tensors) != expected:
        missing = sorted(expected - set(tensors))
        extra = sorted(set(tensors) - expected)
        raise ValueError(
            "steering sidecar tensor inventory differs from layers "
            f"{SIDECAR_LAYERS.start} through {SIDECAR_LAYERS.stop - 1}: "
            f"missing={missing[:6]}, extra={extra[:6]}"
        )
    layers: dict[int, dict[str, SiteSteering]] = {}
    for layer in SIDECAR_LAYERS:
        prefix = f"layers.{layer}"
        alpha_tensor = tensors[f"{prefix}.alpha"]
        direction = tensors[f"{prefix}.direction"]
        gain = tensors[f"{prefix}.attn_row_gain"]
        if alpha_tensor.dtype != torch.float32 or alpha_tensor.shape != torch.Size([]):
            raise ValueError(f"{prefix}.alpha must be a scalar FP32 tensor")
        if direction.dtype != torch.bfloat16 or direction.dim() != 1:
            raise ValueError(f"{prefix}.direction must be a BF16 vector")
        if gain.dtype != torch.bfloat16 or gain.shape != direction.shape:
            raise ValueError(f"{prefix}.attn_row_gain must be BF16 with the direction's shape")
        alpha = float(alpha_tensor.item())
        norm = float(direction.float().norm().item())
        if (
            not math.isfinite(alpha)
            or alpha <= 0.0
            or not bool(torch.isfinite(direction.float()).all())
            or not bool(torch.isfinite(gain.float()).all())
            or abs(norm - 1.0) >= 1e-3
        ):
            raise ValueError(
                f"{prefix} contains non-finite, non-positive-alpha, or non-unit-norm steering values"
            )
        read = direction.float().to(device).contiguous()
        write = (-alpha * direction.float()).to(device).contiguous()
        # The reference keeps (gain - 1) in BF16 and widens it at use; do the same
        # rounding once here so the two implementations see identical constants.
        gain_delta = (gain.float() - 1.0).to(torch.bfloat16).float()
        layers[layer] = {
            "attn": SiteSteering(read, write, _gain_or_none(gain_delta, device)),
            "shared": SiteSteering(read, write),
        }
    return layers


def _gain_or_none(gain_delta: Any, device: Any) -> Any:
    """A gain of exactly 1 everywhere multiplies by 1.0: skip its two kernels."""
    return gain_delta.to(device).contiguous() if bool((gain_delta != 0).any()) else None


def _load_derived(
    tensors: dict[str, Any], metadata: dict[str, str], device: Any, allow_unverified: bool
) -> dict[int, dict[str, SiteSteering]]:
    """A sidecar written by verify/sidecar_vs_weights.py --derive."""
    import torch

    if metadata.get("format") != V2_FORMAT:
        raise ValueError(f"derived steering sidecar without format={V2_FORMAT} in its metadata")
    check_verdict(metadata, allow_unverified)
    parts: dict[tuple[int, str], dict[str, Any]] = {}
    for name, tensor in tensors.items():
        match = re.fullmatch(V2_KEY, name)
        if match is None:
            raise ValueError(f"derived steering sidecar has an unexpected tensor {name!r}")
        if tensor.dtype != torch.float32 or tensor.dim() != 1 or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must be a finite FP32 vector")
        parts.setdefault((int(match.group(1)), match.group(2)), {})[match.group(3)] = tensor
    if not parts:
        raise ValueError("derived steering sidecar holds no operators")
    hidden = {int(t.shape[0]) for part in parts.values() for t in part.values()}
    if len(hidden) != 1:
        raise ValueError(f"derived steering sidecar mixes vector sizes {sorted(hidden)}")
    layers: dict[int, dict[str, SiteSteering]] = {}
    for (layer, site), part in sorted(parts.items()):
        if "read" not in part or "write" not in part:
            raise ValueError(f"layers.{layer}.{site} needs both read and write")
        if not bool(part["read"].any()) or not bool(part["write"].any()):
            raise ValueError(f"layers.{layer}.{site} has an all-zero read or write vector")
        gain = part.get("row_gain")
        if gain is not None and not bool((gain > 0).all()):
            raise ValueError(f"layers.{layer}.{site}.row_gain must be positive")
        layers.setdefault(layer, {})[site] = SiteSteering(
            part["read"].to(device).contiguous(),
            part["write"].to(device).contiguous(),
            None if gain is None else _gain_or_none(gain - 1.0, device),
        )
    return layers


def steer(output: Any, site: SiteSteering, token_scale: Any, stock_rows: Any = None) -> Any:
    """One hooked output, edited:

        x' = (x + s * write * (read . x)) * (1 + s * gain_delta)

    `token_scale` is FP32[T, 1]; `stock_rows` is the bool[T, 1] mask `token_scale == 0`
    when the caller already has it. Rows whose scale is 0 come back bit-identical.

    For the published sidecar this is the reference's arithmetic (FP32, then back to the
    model's dtype) with the products grouped to need 8 kernels instead of 12, so a
    result can differ from the reference's in the last BF16 bit where an FP32 value sits
    on a rounding boundary.
    """
    import torch

    if stock_rows is None:
        stock_rows = token_scale == 0.0
    edited = output.float()
    along = (edited @ site.read).unsqueeze(1) * token_scale  # s * (read . x), [T, 1]
    edited = torch.addcmul(edited, along, site.write)
    if site.gain_delta is not None:
        edited = torch.addcmul(edited, edited * token_scale, site.gain_delta)
    return torch.where(stock_rows, output, edited.to(output.dtype))


def site_module(layer: Any, site: str) -> Any:
    """The module whose output a site's operator edits, or None when it does not exist."""
    if site == "attn":
        return getattr(layer, "attn", None)
    return getattr(getattr(layer, "ffn", None), "shared_experts", None)


# --------------------------------------------------------------------------- worker side


def inner_decoder(model: Any) -> Any:
    """The decoder that owns `layers`, found the way the PP gate finds it."""
    parent = model
    if hasattr(model, "get_language_model"):
        parent = model.get_language_model()
    elif hasattr(model, "language_model"):
        parent = model.language_model
    decoder = getattr(parent, "model", None)
    if decoder is None or not hasattr(decoder, "layers"):
        raise RuntimeError(f"steering: no decoder with layers under {type(model).__name__}")
    return decoder


class _Buffers:
    """A per-row scale and its `== 0` mask. Persistent: captured CUDA graphs read both on
    replay, and every step refills them in place."""

    def __init__(self, max_num_tokens: int, device: Any) -> None:
        import torch

        self.token_scale = torch.zeros((max_num_tokens, 1), dtype=torch.float32, device=device)
        self.stock_rows = torch.ones((max_num_tokens, 1), dtype=torch.bool, device=device)
        self.scale_flat = self.token_scale.view(-1)
        self.stock_flat = self.stock_rows.view(-1)


def load_kernel() -> Any:
    """pp2_steering_kernel.py from this module's directory (it needs Triton)."""
    name = "pp2_steering_kernel"
    if name in sys.modules:
        return sys.modules[name]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"no {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _interpreted() -> bool:
    """Triton's CPU interpreter, which the unit tests run the fused kernel in."""
    return os.environ.get("TRITON_INTERPRET", "0") == "1"


class SteeringRuntime:
    """Per-rank steering state: the scale of every live request, the per-token buffers
    the hooks read, and the hooks on the layers this rank owns."""

    def __init__(
        self,
        config: Config,
        model: Any,
        device: Any,
        max_num_reqs: int,
        max_num_tokens: int,
        pp_rank: int,
        activation_dtype: Any = None,
    ) -> None:
        import torch

        self.config = config
        self.rank = pp_rank
        decoder = inner_decoder(model)
        layers = load_sidecar(config.sidecar_path, device, config.allow_unverified)
        hidden = int(next(iter(next(iter(layers.values())).values())).read.shape[0])
        model_hidden = getattr(getattr(decoder, "config", None), "hidden_size", hidden)
        if model_hidden != hidden:
            raise RuntimeError(
                f"steering: sidecar is for hidden size {hidden}, the model has {model_hidden}"
            )
        self.hidden = hidden
        # Scale per request-state index (the runner's req_index).
        self.scale = torch.zeros(max_num_reqs, dtype=torch.float32, device=device)
        self.main = _Buffers(max_num_tokens, device)
        self._rows = torch.arange(max_num_tokens, dtype=torch.int32, device=device)
        # With decoder replay kept, the layers from decoder_replay_start on run eager
        # prefill steps on a sub-batch: each request's last window of rows. Those calls
        # read a second buffer, filled for that sub-batch.
        self.replay_layers = None
        self.replay: _Buffers | None = None
        self._forward_context: Callable[[], Any] | None = None
        replay_start = int(decoder.end_layer)
        if config.replay == "keep":
            self.replay_layers = getattr(model, "decoder_replay_layers", None)
        if self.replay_layers is not None:
            from vllm.forward_context import get_forward_context

            self._forward_context = get_forward_context
            self.replay = _Buffers(max_num_tokens, device)
            replay_start = int(decoder.decoder_replay_start)
        self.kernel = load_kernel() if config.kernel == "fused" else None
        self._torch_fallbacks: set[str] = set()
        self.handles: list[Any] = []
        self.owned: list[int] = []
        start, end = int(decoder.start_layer), int(decoder.end_layer)
        kernels = on_replay = 0
        variants: dict[bool, SiteSteering] = {}
        for index, sites in layers.items():
            if not start <= index < end:
                continue
            layer = decoder.layers[index]
            for site, steering in sites.items():
                module = site_module(layer, site)
                if module is None:
                    raise RuntimeError(
                        f"steering: layer {index} has no separate "
                        f"{'attention' if site == 'attn' else 'shared-expert'} module to hook "
                        "(a fused shared expert cannot be steered)"
                    )
                in_replay = self.replay is not None and index >= replay_start
                self.handles.append(module.register_forward_hook(self._hook(steering, in_replay)))
                kernels += 1 if self.kernel is not None else steering.kernels
                on_replay += in_replay
                variants.setdefault(steering.gain_delta is not None, steering)
            self.owned.append(index)
        if self.kernel is not None and not _interpreted():
            # Compile both variants now: the first launch must not compile inside a
            # CUDA graph capture.
            dtypes = {torch.bfloat16, activation_dtype or torch.bfloat16}
            for steering in variants.values():
                for dtype in dtypes:
                    self.kernel.warm(
                        torch.zeros((2, hidden), dtype=dtype, device=device),
                        steering.read,
                        steering.write,
                        steering.gain_delta,
                        self.main.token_scale[:2],
                    )
        span = f"{self.owned[0]}-{self.owned[-1]}" if self.owned else "none"
        replay = (
            "off" if config.replay == "off"
            else f"kept ({on_replay} hooks follow the replay batch)" if self.replay is not None
            else "kept (this stage has no replay layers)"
        )
        _log(
            f"rank {pp_rank}: steering layers {span} of this stage's {start}-{end - 1}; "
            f"{len(self.handles)} hooks, {kernels} extra kernels per forward pass by op count; "
            f"kernel {config.kernel}; decoder replay {replay}; "
            f"sidecar sha256 {config.sha256[:16]}; default scale {config.default_scale}; "
            f"debug {'on' if config.debug else 'off'}"
        )

    # The whole-batch buffers under their old names.
    @property
    def token_scale(self) -> Any:
        return self.main.token_scale

    @property
    def stock_rows(self) -> Any:
        return self.main.stock_rows

    def _hook(self, steering: SiteSteering, in_replay: bool):
        main, apply, current = self.main, self._apply, self._current

        if in_replay:

            def hook(module: Any, args: Any, output: Any) -> Any:
                return apply(output, steering, current())

        else:

            def hook(module: Any, args: Any, output: Any) -> Any:
                return apply(output, steering, main)

        return hook

    def _current(self) -> _Buffers:
        """The buffers of the forward this call belongs to: the replay batch's while the
        replay layers run on it, else the whole batch's. A replay batch runs under a
        forward context of its own, which is how a call is told apart."""
        batch = getattr(self.replay_layers, "replay_batch", None)
        if batch is None or self._forward_context is None or self.replay is None:
            return self.main
        try:
            context = self._forward_context()
        except AssertionError:  # no forward context is set
            return self.main
        own = batch.forward_context
        metadata = getattr(own, "attn_metadata", None)
        if context is own or (
            metadata is not None and getattr(context, "attn_metadata", None) is metadata
        ):
            return self.replay
        return self.main

    def _apply(self, output: Any, steering: SiteSteering, buffers: _Buffers) -> Any:
        rows = output.shape[0]
        if self.kernel is not None:
            if self._fusable(output):
                self.kernel.launch(
                    output,
                    steering.read,
                    steering.write,
                    steering.gain_delta,
                    buffers.token_scale[:rows],
                )
                return output
            why = f"{tuple(output.shape)} {output.dtype} on {output.device}"
            if why not in self._torch_fallbacks:
                self._torch_fallbacks.add(why)
                _log(f"rank {self.rank}: the fused kernel does not take {why}; torch path there")
        return steer(output, steering, buffers.token_scale[:rows], buffers.stock_rows[:rows])

    def _fusable(self, output: Any) -> bool:
        import torch

        return (
            output.dim() == 2
            and output.shape[1] == self.hidden
            and output.stride(1) == 1
            and output.dtype in (torch.bfloat16, torch.float32)
            and (output.is_cuda or _interpreted())
        )

    def set_request(self, req_index: int, params: Any) -> None:
        scale, mask = requested(params, self.config)
        if mask is not None and not (mask >> self.rank) & 1:
            scale = 0.0
        self.scale[req_index] = scale
        if self.config.debug and (scale != 0.0 or mask is not None):
            _log(f"rank {self.rank}: request slot {req_index} scale {scale} ranks {mask}")

    def _gather(self, buffers: _Buffers, rows: int, ends: Any, idx_mapping: Any, num_reqs: int) -> None:
        """Each of the first `rows` rows gets the scale of the request it belongs to.
        `ends` are the requests' row ends in batch order; rows past the last one are
        padding and get 0. Five small device ops, no host sync, so the scheduler can
        keep running ahead of the GPU."""
        import torch

        target = buffers.scale_flat[:rows]
        if num_reqs == 0:
            target.zero_()
            buffers.stock_flat[:rows].fill_(True)
            return
        # Row -> position of its request in the batch; padding rows land on num_reqs.
        request = torch.bucketize(self._rows[:rows], ends, right=True)
        # Batch position -> scale, with a trailing 0 for the padding rows.
        by_request = torch.nn.functional.pad(self.scale[idx_mapping[:num_reqs]], (0, 1))
        torch.index_select(by_request, 0, request, out=target)
        torch.eq(target, 0.0, out=buffers.stock_flat[:rows])

    def fill(self, input_batch: Any) -> None:
        """The whole batch's rows, from the step's input batch."""
        num_reqs = int(input_batch.num_reqs)
        self._gather(
            self.main,
            int(input_batch.num_tokens_after_padding),
            input_batch.query_start_loc[1 : num_reqs + 1],
            input_batch.idx_mapping,
            num_reqs,
        )

    def fill_replay(self, kept_query_start_loc: Any, input_batch: Any) -> None:
        """The replay batch's rows, when this step has one: the same requests in the
        same order, each with its last rows only (`kept_query_start_loc`)."""
        batch = getattr(self.replay_layers, "replay_batch", None)
        if batch is None or self.replay is None:
            return
        rows = int(batch.forward_context.batch_descriptor.num_tokens)
        if not int(batch.rows.shape[0]) <= rows <= int(self.replay.scale_flat.shape[0]):
            raise RuntimeError(
                f"steering: replay batch of {int(batch.rows.shape[0])} rows padded to {rows}"
            )
        num_reqs = int(input_batch.num_reqs)
        self._gather(
            self.replay,
            rows,
            kept_query_start_loc[1 : num_reqs + 1],
            input_batch.idx_mapping,
            num_reqs,
        )

    def fill_dummy(self, num_tokens: int) -> None:
        self.main.token_scale[:num_tokens].zero_()
        self.main.stock_rows[:num_tokens].fill_(True)


def make_state_class(base: type, config: Config) -> type:
    """`base` with steering. Only the DeepSeek V4.1 model state is supported."""
    if base.__module__ != STATE_BASE[0] or base.__name__ != STATE_BASE[1]:
        raise RuntimeError(
            f"steering supports {STATE_BASE[1]} only; this model resolves to "
            f"{base.__module__}.{base.__name__}"
        )
    if MODEL_MODULE in sys.modules and MODEL_MODULE not in _installed["patched"]:
        raise RuntimeError(f"steering: {MODEL_MODULE} was imported without its patch")

    class SteeringModelState(base):  # type: ignore[misc, valid-type]
        def __init__(self, vllm_config, model, encoder_cache, device):
            super().__init__(vllm_config, model, encoder_cache, device)
            parallel = vllm_config.parallel_config
            # Each of these runs the layers on something other than "all rows of the
            # step, in batch order", which is what the per-token buffer assumes.
            unsupported = [
                name
                for name, bad in (
                    (
                        "tensor parallel > 1 (PP2_STEERING_ALLOW_TP=1 lifts this for a test boot)",
                        parallel.tensor_parallel_size != 1 and not config.allow_tp,
                    ),
                    ("data parallel > 1", getattr(parallel, "data_parallel_size", 1) != 1),
                    (
                        "prefill context parallel",
                        getattr(parallel, "prefill_context_parallel_size", 1) != 1,
                    ),
                    ("microbatching (DBO / ubatch)", bool(getattr(parallel, "use_ubatching", False))),
                    (
                        "sequence parallel",
                        bool(getattr(parallel, "use_sequence_parallel_moe", False)),
                    ),
                    (
                        "KV-sharing fast prefill",
                        bool(
                            getattr(
                                getattr(vllm_config, "cache_config", None),
                                "kv_sharing_fast_prefill",
                                False,
                            )
                        ),
                    ),
                )
                if bad
            ]
            if unsupported:
                raise RuntimeError(
                    "steering is validated for TP1 x PPn only; refusing: " + ", ".join(unsupported)
                )
            if parallel.tensor_parallel_size != 1:
                _log(
                    f"tensor parallel {parallel.tensor_parallel_size} allowed by "
                    "PP2_STEERING_ALLOW_TP: unvalidated, test boots only"
                )
            replay_layers = getattr(model, "decoder_replay_layers", None)
            if replay_layers is not None and config.replay != "keep":
                raise RuntimeError(
                    "steering: decoder bounded replay is still on for this stage, so the "
                    "replay layers would read the wrong rows of the scale buffer"
                )
            if replay_layers is not None:
                # The replay batch is followed through three upstream names. Refuse a
                # tree that has moved one of them, and the one arrangement not covered.
                decoder = inner_decoder(model)
                missing = [
                    name
                    for name, present in (
                        ("DeepseekV41ModelState._prepare_replay_batch", hasattr(base, "_prepare_replay_batch")),
                        ("DeepseekV41ModelState._kept_query_start_loc", hasattr(self, "_kept_query_start_loc")),
                        ("DecoderReplayLayers.replay_batch", hasattr(replay_layers, "replay_batch")),
                        ("DeepseekV4Model.decoder_replay_start", hasattr(decoder, "decoder_replay_start")),
                    )
                    if not present
                ]
                if missing:
                    raise RuntimeError(
                        "steering: PP2_STEERING_REPLAY=keep does not find " + ", ".join(missing)
                        + " in this vLLM; use PP2_STEERING_REPLAY=off"
                    )
                if getattr(self, "replay_cudagraphs", None) is not None:
                    raise RuntimeError(
                        "steering: PP2_STEERING_REPLAY=keep does not cover a stage that runs "
                        "its replay batch in CUDA graphs of its own (a single-stage lane); "
                        "use PP2_STEERING_REPLAY=off there"
                    )
                if getattr(decoder, "start_layer", None) == decoder.decoder_replay_start:
                    # Not a limit of this module: vLLM v0.34.0 cannot replay there at all.
                    # Its forward runs the layers before the replay layers, an empty range
                    # on such a stage, and _run_layers ends in an assert that nothing in
                    # an empty range satisfies (nvidia/model.py:1120, :1247). The first
                    # forward is the memory profile, and the engine reports only
                    # "Worker failed with error ''". Stop here, with the reason.
                    raise RuntimeError(
                        "steering: this stage begins at the last KV source layer (layer "
                        f"{decoder.decoder_replay_start}), and vLLM v0.34.0 fails its first forward "
                        "there when decoder bounded replay is on (an AssertionError with no "
                        "message). There is no replay to keep on this split: use "
                        "PP2_STEERING_REPLAY=off, and leave VLLM_PATCH_PP_REPLAY_BOUNDARY at its default"
                    )
            from vllm.distributed.parallel_state import get_pp_group

            self.steering = SteeringRuntime(
                config,
                model,
                device,
                self.max_num_reqs,
                self.max_num_tokens,
                get_pp_group().rank_in_group,
                getattr(getattr(vllm_config, "model_config", None), "dtype", None),
            )

        def add_request(self, req_index, new_req_data):
            super().add_request(req_index, new_req_data)
            self.steering.set_request(req_index, new_req_data.sampling_params)

        def prepare_inputs(self, input_batch, req_states):
            model_inputs = super().prepare_inputs(input_batch, req_states)
            self.steering.fill(input_batch)
            return model_inputs

        def prepare_dummy_inputs(self, num_reqs, num_tokens):
            model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
            self.steering.fill_dummy(num_tokens)
            return model_inputs

    if hasattr(base, "_prepare_replay_batch"):

        def _prepare_replay_batch(self, input_batch, *args, **kwargs):
            # Runs inside prepare_attn, before the forward. The request scales it reads
            # were set by add_request, so it does not depend on prepare_inputs.
            super(SteeringModelState, self)._prepare_replay_batch(input_batch, *args, **kwargs)
            self.steering.fill_replay(self._kept_query_start_loc, input_batch)

        SteeringModelState._prepare_replay_batch = _prepare_replay_batch

    SteeringModelState.__name__ = f"Steering{base.__name__}"
    SteeringModelState.__qualname__ = SteeringModelState.__name__
    return SteeringModelState


# --------------------------------------------------------------------------- patches


def _patch_model_states(module: Any, config: Config) -> None:
    original = module.resolve_model_state_cls

    @functools.wraps(original)
    def resolve_model_state_cls(vllm_config, model):
        return make_state_class(original(vllm_config, model), config)

    module.resolve_model_state_cls = resolve_model_state_cls


def _patch_request(module: Any, config: Config) -> None:
    request_cls = module.Request
    original = request_cls.from_engine_core_request.__func__

    @functools.wraps(original)
    def from_engine_core_request(cls, request, block_hasher, *args, **kwargs):
        scale, mask = requested(getattr(request, "sampling_params", None), config)
        salt = salted_cache_salt(request.cache_salt, scale, mask, config.sha256)
        if salt != request.cache_salt:
            request.cache_salt = salt
            # Hashes a frontend computed used the unsalted key: let the engine rehash.
            if getattr(request, "block_hashes", None) is not None:
                request.block_hashes = None
            if args:
                args = (None, *args[1:])
            if "block_hashes" in kwargs:
                kwargs["block_hashes"] = None
        return original(cls, request, block_hasher, *args, **kwargs)

    request_cls.from_engine_core_request = classmethod(from_engine_core_request)


def _patch_protocol(class_name: str) -> Callable[[Any, Config], None]:
    def patch(module: Any, config: Config) -> None:
        request_cls = getattr(module, class_name)
        original = request_cls.to_sampling_params

        @functools.wraps(original)
        def to_sampling_params(self, *args, **kwargs):
            xargs = dict(self.vllm_xargs or {})
            extra = getattr(self, "model_extra", None) or {}
            for key in (SCALE_KEY, RANKS_KEY):
                if key in extra and key not in xargs:
                    xargs[key] = extra[key]
            try:
                if SCALE_KEY in xargs:
                    xargs[SCALE_KEY] = parse_scale(xargs[SCALE_KEY], config.default_scale)
                if RANKS_KEY in xargs:
                    if not config.debug:
                        raise ValueError(f"{RANKS_KEY} is a debug knob and is off on this server")
                    xargs[RANKS_KEY] = parse_ranks(xargs[RANKS_KEY])
            except ValueError as exc:
                bad = RANKS_KEY if RANKS_KEY in str(exc) else SCALE_KEY
                raise _client_error(str(exc), bad, xargs.get(bad)) from None
            self.vllm_xargs = xargs or None
            return original(self, *args, **kwargs)

        request_cls.to_sampling_params = to_sampling_params

    return patch


def _patch_decoder_replay(module: Any, config: Config) -> None:
    decoder_cls = getattr(module, "DeepseekV4Model", None)
    if decoder_cls is None or not hasattr(decoder_cls, "_decoder_replay_supported"):
        return  # v0.31.0: no decoder-side replay to turn off
    if config.replay == "keep":
        return  # upstream decides; the model state follows the replay batch

    original = decoder_cls._decoder_replay_supported

    def _decoder_replay_supported(self, *args, **kwargs):
        # The original also logs its own reason when it answers no.
        if original(self, *args, **kwargs):
            _log(
                f"layers {self.start_layer}-{self.end_layer - 1}: decoder SWA bounded replay "
                "would be on for this stage and steering turns it off (prefill cost: the "
                "tail layers run on every row instead of each request's last window)"
            )
        return False

    decoder_cls._decoder_replay_supported = _decoder_replay_supported


# module name -> (patch, required). A required seam that fails to patch stops the
# process. The two API-server patches are optional: without them a request still
# steers through vllm_xargs, it only loses the HTTP 400 and the top-level spelling.
PATCHES: dict[str, tuple[Callable[[Any, Config], None], bool]] = {
    "vllm.v1.worker.gpu.model_states": (_patch_model_states, True),
    "vllm.v1.request": (_patch_request, True),
    MODEL_MODULE: (_patch_decoder_replay, True),
    "vllm.entrypoints.openai.chat_completion.protocol": (
        _patch_protocol("ChatCompletionRequest"),
        False,
    ),
    "vllm.entrypoints.openai.completion.protocol": (_patch_protocol("CompletionRequest"), False),
}

_installed: dict[str, Any] = {"config": None, "patched": [], "failed": []}


def _apply(name: str, module: Any, config: Config) -> None:
    patch, required = PATCHES[name]
    try:
        patch(module, config)
    except Exception as exc:
        if required:
            _fatal(f"could not patch {name}: {type(exc).__name__}: {exc}")
            raise RuntimeError(f"pp2_steering could not patch {name}") from exc  # tests only
        _installed["failed"].append(name)
        _log(f"optional patch of {name} failed ({type(exc).__name__}: {exc})")
        return
    _installed["patched"].append(name)


class _PatchOnImport:
    """Meta-path finder that runs a patch right after a target module executes."""

    def __init__(self, config: Config, pending: set[str]) -> None:
        self.config = config
        self.pending = pending

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.pending:
            return None
        self.pending.discard(fullname)  # also stops recursion into this finder
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            self.pending.add(fullname)
            return None
        execute = spec.loader.exec_module
        config = self.config

        def exec_module(module):
            execute(module)
            _apply(fullname, module, config)

        spec.loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


def install(environ: dict[str, str] | None = None) -> Config:
    """Validate the configuration and patch vLLM. Raises on a bad configuration; the
    caller (sitecustomize) turns that into a process exit."""
    if _installed["config"] is not None:
        return _installed["config"]
    config = Config.from_env(environ)
    pending: set[str] = set()
    for name in PATCHES:
        module = sys.modules.get(name)
        if module is not None:
            _apply(name, module, config)
        else:
            pending.add(name)
    if pending:
        sys.meta_path.insert(0, _PatchOnImport(config, pending))
    _installed["config"] = config
    return config


def status() -> dict[str, Any]:
    config = _installed["config"]
    return {
        "installed": config is not None,
        "sidecar": None if config is None else config.sidecar_path,
        "sha256": None if config is None else config.sha256,
        "default_scale": None if config is None else config.default_scale,
        "debug": None if config is None else config.debug,
        "replay": None if config is None else config.replay,
        "kernel": None if config is None else config.kernel,
        "patched": list(_installed["patched"]),
        "failed": list(_installed["failed"]),
        "pending": sorted(set(PATCHES) - set(_installed["patched"]) - set(_installed["failed"])),
    }


def _inspect(path: str) -> int:
    metadata = sidecar_metadata(path)
    layers = load_sidecar(path, "cpu", allow_unverified=True)
    derived = metadata.get("format") == V2_FORMAT
    first = next(iter(next(iter(layers.values())).values()))
    print(f"sidecar  {path}")
    print(f"sha256   {file_sha256(path)}")
    print(f"format   {'derived (v2), verdict ' + metadata.get('verdict', '<none>') if derived else 'published (v1)'}")
    print(f"layers   {min(layers)}-{max(layers)} ({len(layers)} layers, "
          f"{sum(len(sites) for sites in layers.values())} operators)")
    print(f"hidden   {int(first.read.shape[0])}")
    print("  strength = -(write . read): alpha when read and write are the same direction")
    for index, sites in layers.items():
        for site, op in sites.items():
            cosine = float(op.read @ op.write / (op.read.norm() * op.write.norm()))
            gain = "none" if op.gain_delta is None else (
                f"{float(op.gain_delta.min()) + 1:.4f}..{float(op.gain_delta.max()) + 1:.4f}"
            )
            print(
                f"  layer {index:>2} {site:<6} strength {-float(op.read @ op.write):.6f}  "
                f"cos(read, write) {cosine:+.4f}  |read| {float(op.read.norm()):.4f}  row gain {gain}"
            )
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "inspect":
        raise SystemExit("usage: python3 pp2_steering.py inspect <sidecar.safetensors>")
    raise SystemExit(_inspect(sys.argv[2]))
