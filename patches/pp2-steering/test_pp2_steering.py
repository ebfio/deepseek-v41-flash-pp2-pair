"""CPU tests for pp2_steering against stand-ins for the vLLM classes it patches.

Needs torch and safetensors, nothing else:

    python3 test_pp2_steering.py          # or: python3 -m pytest -q test_pp2_steering.py

The stand-ins copy the signatures read from vLLM v0.34.0 (and v0.31.0, where they are
the same). They prove the module's own logic: the math, the per-token gather, which
layers each pipeline stage hooks, the replay batch, the cache salt, the API validation
and the fail-closed paths. They say nothing about a real engine: CUDA graphs,
torch.compile over the real model and the real kernels are for verify/steering_probe.py
on a lane.

The fused kernel (PP2_STEERING_KERNEL=fused) is run in Triton's CPU interpreter when
Triton is installed, and skipped with a line saying so when it is not.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

HIDDEN = 5120
NUM_LAYERS = 40

FAKE_TREE = {
    "vllm/__init__.py": "",
    "vllm/exceptions.py": """
        class VLLMValidationError(Exception):
            def __init__(self, message, *, parameter=None, value=None):
                super().__init__(message)
                self.parameter = parameter
                self.value = value
    """,
    "vllm/distributed/__init__.py": "",
    "vllm/distributed/parallel_state.py": """
        class _Group:
            rank_in_group = 0
            world_size = 2
        GROUP = _Group()
        def get_pp_group():
            return GROUP
    """,
    "vllm/forward_context.py": """
        import contextlib
        import dataclasses

        @dataclasses.dataclass
        class ForwardContext:
            attn_metadata: object = None
            batch_descriptor: object = None
            cudagraph_runtime_mode: int = 0

        _context = None

        def get_forward_context():
            assert _context is not None, "Forward context is not set."
            return _context

        @contextlib.contextmanager
        def override_forward_context(context):
            global _context
            previous, _context = _context, context
            try:
                yield
            finally:
                _context = previous
    """,
    "vllm/models/deepseek_v41/decoder_replay_layers.py": """
        import dataclasses
        import torch
        from vllm.forward_context import override_forward_context

        @dataclasses.dataclass
        class ReplayBatch:
            rows: torch.Tensor
            forward_context: object
            run_graph: object = None
            num_padded: object = None

        class DecoderReplayLayers:
            # __call__ and _run of v0.34.0 without the PIECEWISE graph break.
            def __init__(self, window, run_layers, write_batch_kv):
                self.window = window
                self.run_layers = run_layers
                self.write_batch_kv = write_batch_kv
                self.replay_batch = None

            def __call__(self, hidden_states, *states):
                if self.replay_batch is not None:
                    self.write_batch_kv(hidden_states, *states)
                return self._run(hidden_states, *states)

            def _run(self, hidden_states, *states):
                replay_batch = self.replay_batch
                if replay_batch is None:
                    return self.run_layers(hidden_states, *states)
                rows = replay_batch.rows
                inputs = (hidden_states, *states)
                with override_forward_context(replay_batch.forward_context):
                    row_inputs = [None if t is None else t.index_select(0, rows) for t in inputs]
                    row_outputs = self.run_layers(*row_inputs)
                num_tokens = hidden_states.shape[0]
                return tuple(
                    out.new_zeros((num_tokens, *out.shape[1:])).index_copy_(0, rows, out[: rows.numel()])
                    for out in row_outputs
                )
    """,
    "vllm/v1/__init__.py": "",
    "vllm/v1/request.py": """
        class Request:
            def __init__(self, request_id, cache_salt, sampling_params, block_hasher, block_hashes=None):
                self.request_id = request_id
                self.cache_salt = cache_salt
                self.sampling_params = sampling_params
                self.block_hashes = list(block_hashes or [])
                if not self.block_hashes and block_hasher is not None:
                    self.block_hashes = block_hasher(self)

            @classmethod
            def from_engine_core_request(cls, request, block_hasher, block_hashes=None):
                return cls(request.request_id, request.cache_salt, request.sampling_params,
                           block_hasher, block_hashes)
    """,
    "vllm/v1/worker/__init__.py": "",
    "vllm/v1/worker/gpu/__init__.py": "",
    "vllm/v1/worker/gpu/model_states/__init__.py": """
        def init_model_state(vllm_config, model, encoder_cache, device):
            cls = resolve_model_state_cls(vllm_config, model)
            return cls(vllm_config, model, encoder_cache, device)

        def resolve_model_state_cls(vllm_config, model):
            return model.get_model_state_cls()
    """,
    "vllm/models/__init__.py": "",
    "vllm/models/deepseek_v41/__init__.py": "",
    "vllm/models/deepseek_v41/nvidia/__init__.py": "",
    "vllm/models/deepseek_v41/nvidia/model_state.py": """
        class DeepseekV41ModelState:
            def __init__(self, vllm_config, model, encoder_cache, device):
                self.vllm_config = vllm_config
                self.model = model
                self.device = device
                self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
                self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
                self.calls = []
                self.decoder_replay_layers = getattr(model, "decoder_replay_layers", None)
                self.replay_cudagraphs = None
                if self.decoder_replay_layers is not None:
                    import torch
                    self._kept_query_start_loc = torch.zeros(self.max_num_reqs + 1, dtype=torch.int32)

            def _prepare_replay_batch(self, input_batch, *unused):
                # v0.34.0: each prefill's last `window` rows and every decode row, in batch
                # order, as a sub-batch with a forward context of its own. Called by
                # prepare_attn, which runs before prepare_inputs.
                import types
                import torch
                from vllm.forward_context import ForwardContext
                from vllm.models.deepseek_v41.decoder_replay_layers import ReplayBatch
                layers = self.decoder_replay_layers
                layers.replay_batch = None
                starts = input_batch.query_start_loc.tolist()
                lens = [b - a for a, b in zip(starts, starts[1:])]
                kept = [min(n, layers.window) if prefill else n
                        for n, prefill in zip(lens, input_batch.is_prefilling)]
                self.calls.append(("replay", sum(kept)))
                if kept == lens:
                    return
                rows, ends = [], [0]
                for index, n in enumerate(kept):
                    rows += range(starts[index + 1] - n, starts[index + 1])
                    ends.append(ends[-1] + n)
                self._kept_query_start_loc.fill_(-7)  # stale entries past this batch's requests
                self._kept_query_start_loc[: len(ends)] = torch.tensor(ends, dtype=torch.int32)
                layers.replay_batch = ReplayBatch(
                    torch.tensor(rows),
                    ForwardContext({"replay": True}, types.SimpleNamespace(num_tokens=len(rows) + input_batch.replay_padding)),
                )

            def add_request(self, req_index, new_req_data):
                self.calls.append(("add", req_index))

            def prepare_inputs(self, input_batch, req_states):
                self.calls.append(("prepare", input_batch.num_tokens))
                return {"lookback_token_ids": None}

            def prepare_dummy_inputs(self, num_reqs, num_tokens):
                self.calls.append(("dummy", num_tokens))
                return {}

        class DeepseekV41SpecializedDecodeModelState(DeepseekV41ModelState):
            pass
    """,
    "vllm/models/deepseek_v41/nvidia/model.py": """
        class DeepseekV4Model:
            start_layer, end_layer = 20, 40
            would_replay = True
            def _decoder_replay_supported(self, vllm_config, cut):
                return self.would_replay
    """,
    "vllm/entrypoints/__init__.py": "",
    "vllm/entrypoints/openai/__init__.py": "",
    "vllm/entrypoints/openai/chat_completion/__init__.py": "",
    "vllm/entrypoints/openai/chat_completion/protocol.py": """
        import types
        class ChatCompletionRequest:
            def __init__(self, vllm_xargs=None, **extra):
                self.vllm_xargs = vllm_xargs
                self.model_extra = extra
            def to_sampling_params(self, max_tokens=16):
                return types.SimpleNamespace(extra_args=self.vllm_xargs, max_tokens=max_tokens)
    """,
    "vllm/entrypoints/openai/completion/__init__.py": "",
    "vllm/entrypoints/openai/completion/protocol.py": """
        import types
        class CompletionRequest:
            def __init__(self, vllm_xargs=None, **extra):
                self.vllm_xargs = vllm_xargs
                self.model_extra = extra
            def to_sampling_params(self, max_tokens=16):
                return types.SimpleNamespace(extra_args=self.vllm_xargs, max_tokens=max_tokens)
    """,
}


def write_sidecar(path: Path, seed: int = 1017) -> str:
    """The synthetic sidecar of the reference tests: layers 4..36, 99 tensors."""
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for layer in range(4, 37):
        raw = torch.randn(HIDDEN, generator=generator, dtype=torch.float32)
        tensors[f"layers.{layer}.alpha"] = torch.tensor(3.0 + 0.01 * (layer - 4), dtype=torch.float32)
        tensors[f"layers.{layer}.attn_row_gain"] = (
            (1.0 + 0.02 * torch.randn(HIDDEN, generator=generator)).to(torch.bfloat16).contiguous()
        )
        tensors[f"layers.{layer}.direction"] = (raw / raw.norm()).to(torch.bfloat16).contiguous()
    save_file(tensors, str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


TMP = Path(tempfile.mkdtemp(prefix="pp2steer-"))
for rel, body in FAKE_TREE.items():
    target = TMP / "fake" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(textwrap.dedent(body))
sys.path.insert(0, str(TMP / "fake"))
SIDECAR = TMP / "steering.safetensors"
SHA = write_sidecar(SIDECAR)
ENV = {
    "PP2_STEERING_SIDECAR": str(SIDECAR),
    "PP2_STEERING_SHA256": SHA,
    "PP2_STEERING_DEFAULT_SCALE": "0",
    "PP2_STEERING_DEBUG": "1",
}

import vllm.v1.request as fake_request  # noqa: E402  imported BEFORE install on purpose

import pp2_steering as ps  # noqa: E402

CONFIG = ps.install(ENV)

import vllm.distributed.parallel_state as fake_parallel  # noqa: E402
import vllm.exceptions as fake_exceptions  # noqa: E402
import vllm.entrypoints.openai.chat_completion.protocol as fake_chat  # noqa: E402
import vllm.models.deepseek_v41.nvidia.model as fake_model  # noqa: E402
import vllm.models.deepseek_v41.nvidia.model_state as fake_state  # noqa: E402
import vllm.v1.worker.gpu.model_states as fake_states  # noqa: E402


# ----------------------------------------------------------------------- stand-in model


class Missing(torch.nn.Module):
    """PPMissingLayer: a layer another pipeline stage owns."""


class Linear(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        # Rows of norm ~0.2, so the map shrinks its input and 20 layers stay finite.
        self.weight = torch.nn.Parameter(
            (0.2 * torch.randn(64, HIDDEN, generator=g) / HIDDEN**0.5).to(torch.bfloat16),
            requires_grad=False,
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        # A cheap fixed map HIDDEN -> HIDDEN through 64 dims, in the model's dtype.
        return (x @ self.weight.T) @ self.weight


class Attention(Linear):
    def forward(self, positions, hidden_states, llama_4_scaling=None):
        return self.project(hidden_states)


class SharedExperts(Linear):
    def forward(self, x):
        return self.project(x)


class FFN(torch.nn.Module):
    def __init__(self, seed: int, fused: bool = False):
        super().__init__()
        self.shared_experts = None if fused else SharedExperts(seed)
        self.routed = Linear(seed + 1)

    def forward(self, x):
        shared = 0 if self.shared_experts is None else self.shared_experts(x)
        return self.routed.project(x) + shared


class Layer(torch.nn.Module):
    def __init__(self, index: int, fused: bool = False):
        super().__init__()
        self.attn = Attention(1000 + index)
        self.ffn = FFN(2000 + 2 * index, fused)

    def forward(self, x, positions):
        x = x + self.attn(positions, x, None)
        return x + self.ffn(x)


class Decoder(torch.nn.Module):
    def __init__(self, start: int, end: int, fused: bool = False):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=HIDDEN)
        self.start_layer, self.end_layer = start, end
        self.layers = torch.nn.ModuleList(
            Layer(i, fused) if start <= i < end else Missing() for i in range(NUM_LAYERS)
        )

    def forward(self, x, positions):
        for index in range(self.start_layer, self.end_layer):
            x = self.layers[index](x, positions)
        return x


class LanguageModel(torch.nn.Module):
    def __init__(self, start: int, end: int, fused: bool = False, state_cls=None):
        super().__init__()
        self.model = Decoder(start, end, fused)
        self._state_cls = state_cls or fake_state.DeepseekV41ModelState

    def get_model_state_cls(self):
        return self._state_cls


class VisionWrapper(torch.nn.Module):
    """DeepseekV41ForCausalLM: delegates to the language model."""

    def __init__(self, language_model: LanguageModel):
        super().__init__()
        self.language_model = language_model

    def get_language_model(self):
        return self.language_model

    def get_model_state_cls(self):
        return self.language_model.get_model_state_cls()


def vllm_config(tp: int = 1, max_reqs: int = 8, max_tokens: int = 64, **parallel):
    return types.SimpleNamespace(
        scheduler_config=types.SimpleNamespace(
            max_num_seqs=max_reqs, max_num_batched_tokens=max_tokens
        ),
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=tp,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            **parallel,
        ),
    )


def params(**extra):
    return types.SimpleNamespace(extra_args=extra or None)


def new_request(**extra):
    return types.SimpleNamespace(sampling_params=params(**extra))


def batch(lens: list[int], state_indices: list[int], padded: int):
    starts = [0]
    for n in lens:
        starts.append(starts[-1] + n)
    return types.SimpleNamespace(
        num_reqs=len(lens),
        num_tokens=starts[-1],
        num_tokens_after_padding=padded,
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        idx_mapping=torch.tensor(state_indices, dtype=torch.int32),
    )


def build_state(rank: int, start: int, end: int, **kwargs):
    fake_parallel.GROUP.rank_in_group = rank
    model = VisionWrapper(LanguageModel(start, end, **kwargs))
    state = fake_states.init_model_state(vllm_config(), model, None, torch.device("cpu"))
    return model, state


def manual_reference(decoder: Decoder, x, positions, token_scale, sidecar):
    """The decoder forward with the steering written out by hand."""
    for index in range(decoder.start_layer, decoder.end_layer):
        layer = decoder.layers[index]
        attn = layer.attn.project(x)
        if index in sidecar:
            attn = ps.steer(attn, sidecar[index]["attn"], token_scale)
        x = x + attn
        shared = layer.ffn.shared_experts.project(x)
        if index in sidecar:
            shared = ps.steer(shared, sidecar[index]["shared"], token_scale)
        x = x + (layer.ffn.routed.project(x) + shared)  # same association as FFN.forward
    return x


# ----------------------------------------------------------------------- tests


def test_install_patches_modules_imported_before_and_after():
    status = ps.status()
    assert status["installed"] and status["sha256"] == SHA and status["debug"] is True
    assert set(status["patched"]) >= {
        "vllm.v1.request",  # imported before install
        "vllm.v1.worker.gpu.model_states",  # imported after install
        "vllm.models.deepseek_v41.nvidia.model",
        "vllm.entrypoints.openai.chat_completion.protocol",
    }
    assert status["failed"] == []
    # Not imported yet in this process: still pending, patched when first imported.
    assert status["pending"] == ["vllm.entrypoints.openai.completion.protocol"]
    completion = importlib.import_module("vllm.entrypoints.openai.completion.protocol")
    assert ps.status()["pending"] == []
    request = completion.CompletionRequest(steering_scale="0.25")
    assert request.to_sampling_params().extra_args == {"steering_scale": 0.25}
    assert ps.install(ENV) is CONFIG  # idempotent


def test_parse_scale_and_ranks():
    assert ps.parse_scale(None, 0.25) == 0.25
    assert ps.parse_scale(1, 0.0) == 1.0 and ps.parse_scale("0.5", 0.0) == 0.5
    for bad in (True, -0.01, 1.01, float("nan"), float("inf"), "x", [1], {"a": 1}):
        try:
            ps.parse_scale(bad, 0.0)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    assert ps.parse_ranks(None) is None and ps.parse_ranks("3") == 3
    for bad in (True, -1, 1.5, "x"):
        try:
            ps.parse_ranks(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    # Past the API server nothing raises: malformed input is served stock.
    assert ps.requested(params(steering_scale=7), CONFIG) == (0.0, None)
    assert ps.requested(None, CONFIG) == (0.0, None)
    assert ps.requested(params(steering_scale=1, steering_ranks=2), CONFIG) == (1.0, 2)


def test_cache_salt_separates_scales_and_keeps_stock_untouched():
    assert ps.salted_cache_salt(None, 0.0, None, SHA) is None
    assert ps.salted_cache_salt("user", 0.0, None, SHA) == "user"
    one = ps.salted_cache_salt(None, 1.0, None, SHA)
    half = ps.salted_cache_salt(None, 0.5, None, SHA)
    masked = ps.salted_cache_salt(None, 1.0, 1, SHA)
    other = ps.salted_cache_salt(None, 1.0, None, "f" * 64)
    assert len({one, half, masked, other}) == 4 and None not in {one, half, masked, other}
    assert ps.salted_cache_salt("user", 1.0, None, SHA).startswith("user\x1f")


def test_sidecar_loads_like_the_reference_and_rejects_bad_files():
    layers = ps.load_sidecar(str(SIDECAR), "cpu")
    assert list(layers) == list(range(4, 37))
    raw = load_file(str(SIDECAR))
    for index, sites in layers.items():
        direction = raw[f"layers.{index}.direction"].float()
        alpha = float(raw[f"layers.{index}.alpha"])
        assert set(sites) == {"attn", "shared"}
        for op in sites.values():  # both sites: read along d, write -alpha d
            assert op.read.dtype == torch.float32 and op.read.shape == (HIDDEN,)
            assert torch.equal(op.read, direction) and torch.equal(op.write, -alpha * direction)
        reference_delta = (raw[f"layers.{index}.attn_row_gain"].float() - 1.0).to(torch.bfloat16)
        assert torch.equal(sites["attn"].gain_delta, reference_delta.float())
        assert sites["shared"].gain_delta is None  # the published row gain is the attention's
        assert sites["attn"].kernels == 8 and sites["shared"].kernels == 6

    def rejects(mutate, needle):
        tensors = load_file(str(SIDECAR))
        mutate(tensors)
        bad = TMP / "bad.safetensors"
        save_file(tensors, str(bad))
        try:
            ps.load_sidecar(str(bad), "cpu")
        except ValueError as exc:
            assert needle in str(exc), exc
            return
        raise AssertionError(f"accepted a sidecar with {needle}")

    rejects(lambda t: t.pop("layers.10.alpha"), "inventory")
    rejects(lambda t: t.update({"layers.37.alpha": torch.tensor(1.0)}), "inventory")
    rejects(lambda t: t.update({"layers.10.alpha": torch.tensor(0.0)}), "non-positive-alpha")
    rejects(lambda t: t.update({"layers.9.direction": t["layers.9.direction"] * 2}), "non-unit-norm")
    rejects(lambda t: t.update({"layers.9.direction": t["layers.9.direction"].float()}), "BF16")


def test_math_keeps_unsteered_rows_bit_identical_and_matches_closed_form():
    sites = ps.load_sidecar(str(SIDECAR), "cpu")[18]
    alpha = float(load_file(str(SIDECAR))["layers.18.alpha"])
    layer = types.SimpleNamespace(direction=sites["attn"].read, alpha=alpha, gain_delta=sites["attn"].gain_delta)
    generator = torch.Generator().manual_seed(42)
    attention = torch.randn(6, HIDDEN, dtype=torch.bfloat16, generator=generator)
    shared = torch.randn(6, HIDDEN, dtype=torch.bfloat16, generator=generator)
    scale = torch.tensor([0.0, 1.0, 1.0, 0.5, 0.0, 0.0]).unsqueeze(1)
    attention_out = ps.steer(attention, sites["attn"], scale)
    shared_out = ps.steer(shared, sites["shared"], scale)
    assert attention_out.dtype == torch.bfloat16 and shared_out.dtype == torch.bfloat16
    for row in (0, 4, 5):
        assert torch.equal(attention_out[row], attention[row])
        assert torch.equal(shared_out[row], shared[row])
    for row in (1, 2, 3):
        assert not torch.equal(attention_out[row], attention[row])
        assert not torch.equal(shared_out[row], shared[row])
    d = layer.direction.double()
    for row in (1, 3):
        s = float(scale[row])
        x = shared[row].double()
        expected = x - s * layer.alpha * (x @ d) * d
        assert (shared_out[row].double() - expected).abs().max() < 0.02  # BF16 rounding
        # The component along d is scaled by (1 - s*alpha), up to rounding.
        assert abs(float(shared_out[row].double() @ d) - (1 - s * layer.alpha) * float(x @ d)) < 0.05
        a = attention[row].double()
        expected = (a - s * layer.alpha * (a @ d) * d) * (1 + s * layer.gain_delta.double())
        assert (attention_out[row].double() - expected).abs().max() < 0.03


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """Spacing of BF16 values at |x| (8 significand bits)."""
    return torch.ldexp(torch.ones_like(x), torch.frexp(x.abs().clamp_min(1e-30))[1] - 8)


def test_reference_formula_is_reproduced_to_the_last_bf16_bit():
    """Against the expressions of ignition/recipes/deepseek41/steering.py, written out
    here operation for operation: equal except where an FP32 value sits on a BF16
    rounding boundary, and never more than one BF16 step apart."""
    raw = load_file(str(SIDECAR))
    layer = ps.load_sidecar(str(SIDECAR), "cpu")[24]
    direction_bf16 = raw["layers.24.direction"]
    delta_bf16 = (raw["layers.24.attn_row_gain"].float() - 1.0).to(torch.bfloat16)
    alpha = float(raw["layers.24.alpha"])
    output = torch.randn(5, HIDDEN, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(7))
    token_scale = torch.tensor([[1.0], [0.0], [0.75], [1.0], [0.3]])

    direction = direction_bf16.float()
    projected = output.float()
    dot = (projected @ direction).unsqueeze(1)
    projected = projected - token_scale * alpha * dot * direction.unsqueeze(0)
    reference_shared = torch.where(token_scale == 0.0, output, projected.to(output.dtype))
    projected = projected * (1.0 + token_scale * delta_bf16.float().unsqueeze(0))
    reference_attention = torch.where(token_scale == 0.0, output, projected.to(output.dtype))

    for ours, reference in (
        (ps.steer(output, layer["attn"], token_scale), reference_attention),
        (ps.steer(output, layer["shared"], token_scale), reference_shared),
    ):
        assert ours.dtype == reference.dtype == torch.bfloat16
        assert torch.equal(ours[1], output[1]) and torch.equal(reference[1], output[1])
        same = (ours == reference).float().mean().item()
        gap = (ours.float() - reference.float()).abs()
        assert same > 0.999, same
        assert bool((gap <= bf16_ulp(reference.float())).all()), float(gap.max())
    # The mask the runtime precomputes gives the same result as deriving it here.
    mask = token_scale == 0.0
    assert torch.equal(
        ps.steer(output, layer["attn"], token_scale, mask), ps.steer(output, layer["attn"], token_scale)
    )


def test_fill_gathers_scale_per_row_and_zeroes_padding():
    _, state = build_state(rank=0, start=0, end=20)
    steering = state.steering
    state.add_request(5, new_request(steering_scale=1.0))
    state.add_request(2, new_request(steering_scale=0.5))
    state.add_request(7, new_request())
    assert steering.scale.tolist() == [0, 0, 0.5, 0, 0, 1.0, 0, 0]
    steering.token_scale.fill_(9.0)  # stale values from an earlier step
    out = state.prepare_inputs(batch([3, 1, 4], [5, 2, 7], padded=12), None)
    assert out == {"lookback_token_ids": None}  # the base class result is passed through
    assert steering.token_scale[:12, 0].tolist() == [1, 1, 1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0]
    assert steering.stock_rows[:12, 0].tolist() == [False] * 4 + [True] * 8
    assert steering.token_scale[12, 0] == 9.0  # rows the step does not use are not touched
    # Same requests, different batch order and no padding.
    state.prepare_inputs(batch([2, 2, 1], [7, 5, 2], padded=5), None)
    assert steering.token_scale[:5, 0].tolist() == [0, 0, 1, 1, 0.5]
    assert steering.stock_rows[:5, 0].tolist() == [True, True, False, False, False]
    state.prepare_inputs(batch([], [], padded=4), None)
    assert steering.token_scale[:4, 0].tolist() == [0, 0, 0, 0]
    assert steering.stock_rows[:4, 0].tolist() == [True] * 4
    steering.token_scale.fill_(9.0)
    steering.stock_rows.fill_(False)
    state.prepare_dummy_inputs(2, 6)
    assert steering.token_scale[:6, 0].tolist() == [0] * 6 and steering.token_scale[6, 0] == 9.0
    assert steering.stock_rows[:6, 0].tolist() == [True] * 6 and not steering.stock_rows[6, 0]
    # A slot that is reused takes the new request's scale.
    state.add_request(5, new_request())
    assert steering.scale[5] == 0.0
    assert [c[0] for c in state.calls].count("add") == 4  # super() was called every time


def test_each_stage_hooks_only_the_steered_layers_it_owns():
    sidecar = ps.load_sidecar(str(SIDECAR), "cpu")
    for rank, (start, end), owned in ((0, (0, 20), range(4, 20)), (1, (20, 40), range(20, 37))):
        model, state = build_state(rank, start, end)
        assert type(state).__name__ == "SteeringDeepseekV41ModelState"
        assert isinstance(state, fake_state.DeepseekV41ModelState)
        assert state.steering.rank == rank and state.steering.owned == list(owned)
        assert len(state.steering.handles) == 2 * len(owned)
        decoder = model.language_model.model
        state.add_request(0, new_request(steering_scale=1.0))
        state.add_request(1, new_request(steering_scale=0.0))
        state.add_request(2, new_request(steering_scale=0.4))
        state.prepare_inputs(batch([2, 3, 1], [0, 1, 2], padded=8), None)
        token_scale = torch.tensor([1, 1, 0, 0, 0, 0.4, 0, 0]).unsqueeze(1)
        generator = torch.Generator().manual_seed(100 + rank)
        x = torch.randn(8, HIDDEN, dtype=torch.bfloat16, generator=generator)
        positions = torch.arange(8)
        hooked = decoder(x, positions)
        assert bool(torch.isfinite(hooked.float()).all())  # NaN would make the checks vacuous
        assert torch.equal(hooked, manual_reference(decoder, x, positions, token_scale, sidecar))
        stock = manual_reference(decoder, x, positions, torch.zeros(8, 1), sidecar)
        for row in (2, 3, 4, 6, 7):  # scale-0 and padding rows: stock bits
            assert torch.equal(hooked[row], stock[row])
        for row in (0, 1, 5):
            assert not torch.equal(hooked[row], stock[row])


def test_debug_rank_mask_limits_steering_to_the_named_stages():
    for rank in (0, 1):
        _, state = build_state(rank, 20 * rank, 20 * rank + 20)
        for index, mask in enumerate((3, 1, 2, 0)):
            state.add_request(index, new_request(steering_scale=1.0, steering_ranks=mask))
        expected = [float((mask >> rank) & 1) for mask in (3, 1, 2, 0)]
        assert state.steering.scale[:4].tolist() == expected


def test_request_salt_is_applied_in_the_engine_and_drops_frontend_hashes():
    seen = []

    def hasher(request):
        seen.append(request.cache_salt)
        return [hashlib.sha256(repr(request.cache_salt).encode()).hexdigest()]

    def engine_request(salt, block_hashes=None, **extra):
        return types.SimpleNamespace(
            request_id="r", cache_salt=salt, sampling_params=params(**extra), block_hashes=block_hashes
        )

    stock = fake_request.Request.from_engine_core_request(engine_request(None), hasher)
    assert stock.cache_salt is None
    explicit = fake_request.Request.from_engine_core_request(
        engine_request("u", steering_scale=0.0), hasher
    )
    assert explicit.cache_salt == "u"
    steered = fake_request.Request.from_engine_core_request(
        engine_request("u", steering_scale=1.0), hasher
    )
    assert steered.cache_salt == ps.salted_cache_salt("u", 1.0, None, SHA)
    assert steered.block_hashes != explicit.block_hashes
    again = fake_request.Request.from_engine_core_request(
        engine_request("u", steering_scale=1.0), hasher
    )
    assert again.block_hashes == steered.block_hashes  # the cache still works within a scale
    # Hashes a frontend computed without the salt are discarded, however they arrive.
    for call in (
        lambda r: fake_request.Request.from_engine_core_request(r, hasher, ["stale"]),
        lambda r: fake_request.Request.from_engine_core_request(r, hasher, block_hashes=["stale"]),
    ):
        request = engine_request("u", block_hashes=["stale"], steering_scale=1.0)
        assert call(request).block_hashes == steered.block_hashes and request.block_hashes is None
    # ... and kept for stock requests.
    kept = fake_request.Request.from_engine_core_request(engine_request(None), hasher, ["front"])
    assert kept.block_hashes == ["front"]
    pooling = types.SimpleNamespace(request_id="p", cache_salt=None, sampling_params=None, block_hashes=None)
    assert fake_request.Request.from_engine_core_request(pooling, hasher).cache_salt is None


def test_api_accepts_both_spellings_and_rejects_bad_values():
    Chat = fake_chat.ChatCompletionRequest
    assert Chat().to_sampling_params().extra_args is None
    assert Chat({"steering_scale": 1}).to_sampling_params().extra_args == {"steering_scale": 1.0}
    assert Chat(steering_scale=0.5).to_sampling_params(max_tokens=3).max_tokens == 3
    assert Chat(steering_scale=0.5).to_sampling_params().extra_args == {"steering_scale": 0.5}
    # vllm_xargs wins over the top-level spelling; other keys pass through.
    both = Chat({"steering_scale": 0.2, "other": "x"}, steering_scale=0.9)
    assert both.to_sampling_params().extra_args == {"steering_scale": 0.2, "other": "x"}
    assert Chat({"steering_scale": 1, "steering_ranks": "2"}).to_sampling_params().extra_args == {
        "steering_scale": 1.0,
        "steering_ranks": 2,
    }
    for bad in ({"steering_scale": 1.5}, {"steering_scale": "abc"}, {"steering_scale": -0.1},
                {"steering_scale": True}, {"steering_ranks": -1}):
        try:
            Chat(bad).to_sampling_params()
        except fake_exceptions.VLLMValidationError as exc:  # what the server maps to 400
            assert exc.parameter == next(iter(bad)) and exc.value == bad[exc.parameter], exc
            continue
        raise AssertionError(f"accepted {bad}")
    try:
        Chat(steering_scale=2).to_sampling_params()  # the top-level spelling is validated too
    except fake_exceptions.VLLMValidationError as exc:
        assert exc.parameter == "steering_scale"
    else:
        raise AssertionError("accepted a top-level scale of 2")
    # With debug off the rank mask is refused instead of silently ignored.
    production = ps.Config(CONFIG.sidecar_path, CONFIG.sha256, 0.0, False)
    module = types.SimpleNamespace(ChatCompletionRequest=type("C", (), {
        "__init__": lambda self, x: setattr(self, "vllm_xargs", x),
        "to_sampling_params": lambda self: self.vllm_xargs,
    }))
    ps._patch_protocol("ChatCompletionRequest")(module, production)
    try:
        module.ChatCompletionRequest({"steering_ranks": 1}).to_sampling_params()
    except fake_exceptions.VLLMValidationError as exc:
        assert "debug" in str(exc) and exc.parameter == "steering_ranks"
    else:
        raise AssertionError("accepted the debug knob with debug off")
    assert ps.requested(params(steering_scale=1, steering_ranks=1), production) == (1.0, None)


def test_default_scale_applies_when_a_request_does_not_ask():
    config = ps.Config(CONFIG.sidecar_path, CONFIG.sha256, 1.0, False)
    assert ps.requested(params(), config) == (1.0, None)
    assert ps.requested(params(steering_scale=0), config) == (0.0, None)
    assert ps.requested(None, config) == (0.0, None)  # pooling stays stock
    assert ps.Config.from_env({**ENV, "PP2_STEERING_DEFAULT_SCALE": "1"}).default_scale == 1.0


def test_configuration_fails_closed():
    def raises(env, needle):
        try:
            ps.Config.from_env(env)
        except ValueError as exc:
            assert needle in str(exc), exc
            return
        raise AssertionError(f"accepted {env}")

    raises({}, "PP2_STEERING_SIDECAR")
    raises({**ENV, "PP2_STEERING_SHA256": ""}, "64-hex")
    raises({**ENV, "PP2_STEERING_SHA256": "0" * 64}, "expected")
    raises({**ENV, "PP2_STEERING_DEFAULT_SCALE": "2"}, "[0, 1]")
    try:
        ps.Config.from_env({**ENV, "PP2_STEERING_SIDECAR": str(TMP / "absent")})
    except OSError:
        pass
    else:
        raise AssertionError("accepted a missing sidecar")


def test_unsupported_setups_refuse_to_start():
    def raises(build, needle):
        try:
            build()
        except RuntimeError as exc:
            assert needle in str(exc), exc
            return
        raise AssertionError(f"started with {needle}")

    fake_parallel.GROUP.rank_in_group = 0
    # Another model state (specialized decode) is not the class this was written for.
    special = VisionWrapper(
        LanguageModel(0, 20, state_cls=fake_state.DeepseekV41SpecializedDecodeModelState)
    )
    raises(lambda: fake_states.init_model_state(vllm_config(), special, None, "cpu"), "only")
    raises(lambda: ps.make_state_class(dict, CONFIG), "only")
    # Tensor parallel was not validated.
    model = VisionWrapper(LanguageModel(0, 20))
    raises(lambda: fake_states.init_model_state(vllm_config(tp=2), model, None, "cpu"), "tensor parallel")
    for knob, needle in (("use_ubatching", "microbatching"), ("use_sequence_parallel_moe", "sequence")):
        config = vllm_config(**{knob: True})
        raises(lambda: fake_states.init_model_state(config, model, None, "cpu"), needle)
    # A stage that still has decoder replay layers would read the wrong rows.
    replaying = VisionWrapper(LanguageModel(0, 20))
    replaying.decoder_replay_layers = object()
    raises(lambda: fake_states.init_model_state(vllm_config(), replaying, None, "cpu"), "replay")
    # A fused shared expert leaves nothing to hook.
    fused = VisionWrapper(LanguageModel(0, 20, fused=True))
    raises(lambda: fake_states.init_model_state(vllm_config(), fused, None, "cpu"), "fused")
    # The sidecar has to fit the model.
    narrow = VisionWrapper(LanguageModel(0, 20))
    narrow.language_model.model.config.hidden_size = 4096
    raises(lambda: fake_states.init_model_state(vllm_config(), narrow, None, "cpu"), "hidden size")
    raises(lambda: ps.inner_decoder(torch.nn.Identity()), "no decoder")


def test_decoder_replay_is_switched_off():
    logged = []
    keep, ps._log = ps._log, logged.append
    try:
        decoder = fake_model.DeepseekV4Model()
        assert decoder._decoder_replay_supported(None, 20) is False
        assert len(logged) == 1 and "layers 20-39" in logged[0] and "turns it off" in logged[0]
        # Upstream already said no (wrong stage, drafter window ...): nothing to report.
        decoder.would_replay = False
        assert decoder._decoder_replay_supported(None, cut=20) is False and len(logged) == 1
    finally:
        ps._log = keep
    # v0.31.0 has no such method: the patch is a no-op there.
    ps._patch_decoder_replay(types.SimpleNamespace(DeepseekV4Model=type("D", (), {})), CONFIG)
    ps._patch_decoder_replay(types.SimpleNamespace(), CONFIG)


def test_a_required_seam_that_cannot_be_patched_stops_the_import():
    broken = TMP / "broken" / "brokenpkg"
    broken.mkdir(parents=True)
    (broken / "__init__.py").write_text("")
    (broken / "target.py").write_text("VALUE = 1\n")
    sys.path.insert(0, str(TMP / "broken"))
    ps.PATCHES["brokenpkg.target"] = (lambda module, config: module.missing_attribute, True)

    class WouldExit(BaseException):
        pass

    def fatal(message):
        raise WouldExit(message)

    keep, ps._fatal = ps._fatal, fatal
    try:
        sys.meta_path.insert(0, ps._PatchOnImport(CONFIG, {"brokenpkg.target"}))
        try:
            importlib.import_module("brokenpkg.target")
        except WouldExit as exc:  # in production: os._exit(78)
            assert "could not patch brokenpkg.target" in str(exc)
        else:
            raise AssertionError("a failed required patch did not stop the process")
    finally:
        ps._fatal = keep
        ps.PATCHES.pop("brokenpkg.target")
    # An optional seam only logs and is reported by status().
    ps.PATCHES["brokenpkg.optional"] = (lambda module, config: module.missing_attribute, False)
    try:
        ps._apply("brokenpkg.optional", types.SimpleNamespace(), CONFIG)
        assert "brokenpkg.optional" in ps.status()["failed"]
    finally:
        ps.PATCHES.pop("brokenpkg.optional")
        ps._installed["failed"].remove("brokenpkg.optional")


def test_steering_survives_fullgraph_tracing():
    """Not needed by vLLM v0.34.0 (the DeepSeek decoder is not torch.compiled there);
    kept because the reference implementation requires it and a compiled path may come."""
    layer = ps.load_sidecar(str(SIDECAR), "cpu")[24]["shared"]
    buffer = torch.zeros(16, 1)
    buffer[2:5] = 1.0

    def step(hidden: torch.Tensor) -> torch.Tensor:
        return ps.steer(hidden, layer, buffer[: hidden.shape[0]])

    try:
        compiled = torch.compile(step, backend="eager", fullgraph=True)
        hidden = torch.randn(6, HIDDEN, dtype=torch.bfloat16)
        out = compiled(hidden)
    except Exception as exc:  # pragma: no cover - depends on the local torch build
        if "not supported" in str(exc).lower() or "dynamo" in type(exc).__module__:
            print(f"  (skipped: torch.compile unavailable here: {type(exc).__name__})")
            return
        raise
    assert torch.equal(out, step(hidden))
    assert torch.equal(out[0], hidden[0]) and not torch.equal(out[3], hidden[3])
    buffer.zero_()  # the compiled function reads the buffer, it did not bake it in
    assert torch.equal(compiled(hidden), hidden)


def test_sitecustomize_installs_at_interpreter_start_and_fails_closed():
    """The v0.34.0 overlay's sitecustomize, run the way the container runs it:
    PYTHONPATH names the overlay directory and every Python process starts through it."""
    overlay = HERE.parent / "vllm-pp2-flag-v034"
    if not (overlay / "sitecustomize.py").is_file():
        print("  (skipped: ../vllm-pp2-flag-v034/sitecustomize.py is not next to this directory)")
        return

    def python(code, tree=TMP / "fake", **env):
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.pathsep.join([str(overlay), str(tree)]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PP2_STEERING_HOME": str(HERE),
            **{k: v for k, v in env.items() if v is not None},
        }
        return subprocess.run(
            [sys.executable, "-c", code], env=environment, capture_output=True, text=True, timeout=120
        )

    flag = (
        "import sys, json; early = 'vllm' in sys.modules; "
        "import vllm.models.deepseek_v41.nvidia.model as m; "
        "flag = getattr(m.DeepseekV4Model, 'supports_aux_hidden_states_over_pp', None); "
    )
    # Steering on: installed before any user code without importing vLLM, and both the
    # flag flip and the steering patch land when the model module is imported.
    on = python(
        flag + "import pp2_steering; "
        "print(json.dumps({'flag': flag, 'early': early, **pp2_steering.status()}))",
        **ENV,
    )
    assert on.returncode == 0, on.stderr
    status = json.loads(on.stdout)
    assert status["flag"] is True and status["early"] is False
    assert status["installed"] and status["sha256"] == SHA
    assert "vllm.models.deepseek_v41.nvidia.model" in status["patched"]
    assert "vllm.v1.request" in status["pending"]  # patched when that process imports it
    replay = python(
        "import vllm.models.deepseek_v41.nvidia.model as m; "
        "print(m.DeepseekV4Model()._decoder_replay_supported(None, 20))",
        **ENV,
    )
    assert replay.returncode == 0 and replay.stdout.strip() == "False", replay.stderr
    # ... and a process that imports the engine modules gets them patched on the way in.
    engine = python(
        "import vllm.v1.request, vllm.v1.worker.gpu.model_states, pp2_steering; "
        "print(sorted(pp2_steering.status()['patched']))",
        **ENV,
    )
    assert engine.returncode == 0, engine.stderr
    assert "vllm.v1.request" in engine.stdout and "model_states" in engine.stdout
    # Steering off: the module is never imported and the flag flip is unchanged.
    off = python(flag + "print(flag, early, 'pp2_steering' in sys.modules)")
    assert off.returncode == 0 and off.stdout.split() == ["True", "False", "False"], off.stderr
    open_path = TMP / "derived-open-for-start.safetensors"
    write_derived(open_path, verdict="OPEN")
    unverified = {**ENV, "PP2_STEERING_SIDECAR": str(open_path), "PP2_STEERING_SHA256": ps.file_sha256(str(open_path))}
    # Bad configurations end the process with status 78 before any user code runs.
    for name, env in (
        ("wrong sha256", {**ENV, "PP2_STEERING_SHA256": "0" * 64}),
        ("no sha256", {**ENV, "PP2_STEERING_SHA256": None}),
        ("sidecar missing", {**ENV, "PP2_STEERING_SIDECAR": str(TMP / "absent.safetensors")}),
        ("module missing", {**ENV, "PP2_STEERING_HOME": str(TMP / "nowhere")}),
        ("bad default scale", {**ENV, "PP2_STEERING_DEFAULT_SCALE": "7"}),
        ("derived sidecar that did not rebuild its checkpoint", unverified),
    ):
        bad = python("print('user code ran')", **env)
        assert bad.returncode == 78, (name, bad.returncode, bad.stderr)
        assert "FATAL" in bad.stderr and "user code ran" not in bad.stdout, (name, bad.stderr)
    # A vLLM whose seam is gone: the import that should have been patched kills the process.
    drifted = TMP / "drifted"
    shutil.copytree(TMP / "fake", drifted)
    (drifted / "vllm" / "v1" / "request.py").write_text("class SomethingElse:\n    pass\n")
    gone = python("import vllm.v1.request; print('booted unpatched')", tree=drifted, **ENV)
    assert gone.returncode == 78, (gone.returncode, gone.stderr)
    assert "could not patch vllm.v1.request" in gone.stderr and "booted" not in gone.stdout


def write_derived(path: Path, verdict: str = "REBUILT", **changes) -> dict[str, torch.Tensor]:
    """A derived (v2) sidecar: layer 4 with both sites (attention has a row gain, read and
    write differ), layer 20 with the shared expert only."""
    generator = torch.Generator().manual_seed(99)
    vector = lambda scale=1.0: scale * torch.randn(HIDDEN, generator=generator) / HIDDEN**0.5  # noqa: E731
    tensors = {
        "layers.4.attn.read": vector(), "layers.4.attn.write": vector(-3.0),
        "layers.4.attn.row_gain": 1.0 + 0.01 * torch.randn(HIDDEN, generator=generator),
        "layers.4.shared.read": vector(), "layers.4.shared.write": vector(-3.0),
        "layers.20.shared.read": vector(), "layers.20.shared.write": vector(-3.0),
    }
    for name, value in changes.items():
        key = name.replace("__", ".")
        if value is None:
            tensors.pop(key)
        else:
            tensors[key] = value
    metadata = {"format": ps.V2_FORMAT, "verdict": verdict} if verdict is not None else None
    save_file(tensors, str(path), metadata=metadata)
    return tensors


def test_derived_sidecar_loads_and_applies_its_own_operators():
    path = TMP / "derived.safetensors"
    raw = write_derived(path)
    layers = ps.load_sidecar(str(path), "cpu")
    assert sorted(layers) == [4, 20] and set(layers[4]) == {"attn", "shared"} and set(layers[20]) == {"shared"}
    attn = layers[4]["attn"]
    assert torch.equal(attn.read, raw["layers.4.attn.read"]) and torch.equal(attn.write, raw["layers.4.attn.write"])
    assert torch.equal(attn.gain_delta, raw["layers.4.attn.row_gain"] - 1.0)
    assert layers[4]["shared"].gain_delta is None and attn.kernels == 8
    x = torch.randn(5, HIDDEN, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(5))
    scale = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0]).unsqueeze(1)
    out = ps.steer(x, attn, scale)
    assert out.dtype == torch.bfloat16
    for row in (1, 4):
        assert torch.equal(out[row], x[row])
    xd, read, write, gain = x.double(), attn.read.double(), attn.write.double(), attn.gain_delta.double()
    for row in (0, 2, 3):
        s = float(scale[row])
        expected = (xd[row] + s * write * (read @ xd[row])) * (1 + s * gain)
        assert (out[row].double() - expected).abs().max() < 0.03 and not torch.equal(out[row], x[row])
    # A stage hooks the sites the file has, and nothing else.
    config = ps.Config(str(path), "0" * 64, 0.0, False)
    decoder = Decoder(0, 20)
    runtime = ps.SteeringRuntime(config, types.SimpleNamespace(model=decoder), torch.device("cpu"), 8, 64, 0)
    assert runtime.owned == [4] and len(runtime.handles) == 2
    late = ps.SteeringRuntime(config, types.SimpleNamespace(model=Decoder(20, 40)), torch.device("cpu"), 8, 64, 1)
    assert late.owned == [20] and len(late.handles) == 1
    assert ps._inspect(str(path)) == 0


def test_derived_sidecar_refuses_what_it_cannot_vouch_for():
    def rejects(needle: str, allow_unverified: bool = False, **kwargs) -> None:
        path = TMP / "derived-bad.safetensors"
        write_derived(path, **kwargs)
        try:
            ps.load_sidecar(str(path), "cpu", allow_unverified)
        except ValueError as exc:
            assert needle in str(exc), exc
            return
        raise AssertionError(f"accepted a derived sidecar with {kwargs}")

    rejects("verdict 'OPEN'", verdict="OPEN")  # the tool writes the file even when nothing rebuilt
    rejects("without format", verdict=None)
    rejects("needs both read and write", layers__4__attn__write=None)
    rejects("finite FP32 vector", layers__4__attn__read=torch.randn(HIDDEN).to(torch.bfloat16))
    rejects("finite FP32 vector", layers__4__attn__read=torch.full((HIDDEN,), float("nan")))
    rejects("unexpected tensor", layers__4__attn__direction=torch.randn(HIDDEN))
    rejects("mixes vector sizes", layers__20__shared__read=torch.randn(64), layers__20__shared__write=torch.randn(64))
    rejects("must be positive", layers__4__attn__row_gain=torch.zeros(HIDDEN))
    rejects("all-zero", layers__4__shared__write=torch.zeros(HIDDEN))
    path = TMP / "derived-open.safetensors"
    write_derived(path, verdict="OPEN")
    assert sorted(ps.load_sidecar(str(path), "cpu", allow_unverified=True)) == [4, 20]  # experiments only
    env = {**ENV, "PP2_STEERING_SIDECAR": str(path), "PP2_STEERING_SHA256": ps.file_sha256(str(path))}
    try:  # refused when the configuration is read: at interpreter start, before any weights load
        ps.Config.from_env(env)
    except ValueError as exc:
        assert "did not rebuild" in str(exc)
    else:
        raise AssertionError("the configuration accepted a sidecar that did not rebuild its checkpoint")
    assert ps.Config.from_env({**env, "PP2_STEERING_ALLOW_UNVERIFIED": "1"}).allow_unverified is True
    unchecked = ps.Config(str(path), "0" * 64, 0.0, False)
    try:  # and a stage that somehow got that far does not come up either
        ps.SteeringRuntime(unchecked, types.SimpleNamespace(model=Decoder(0, 20)), "cpu", 8, 64, 0)
    except ValueError as exc:
        assert "did not rebuild" in str(exc)
    else:
        raise AssertionError("a stage started from a sidecar that did not rebuild its checkpoint")
    assert ps.sidecar_metadata(str(SIDECAR)) == {}  # the published file carries none


def test_stage_check_agrees_with_the_loader_on_what_the_two_files_show():
    """stage_check.py reads the sidecar's header and the module's source before a boot.
    On everything that shows there it must decide as the loader does, and it must never
    refuse a pair the loader takes."""
    import stage_check as sc

    module = str(HERE / "pp2_steering.py")
    sys.path.insert(0, str(HERE.parent.parent / "verify"))
    import make_tracer_sidecar as tracer_tool

    def decide(path: Path, allow: bool) -> tuple[bool, bool, str]:
        try:
            said, staged = sc.check(str(path), module, allow), True
        except sc.Refused as exc:
            said, staged = str(exc), False
        try:
            ps.check_verdict(ps.sidecar_metadata(str(path)), allow)  # as the interpreter start does
            ps.load_sidecar(str(path), "cpu", allow)
            loaded = True
        except Exception:  # noqa: BLE001  safetensors has its own error type for a broken file
            loaded = False
        return staged, loaded, said

    work = TMP / "stage"
    work.mkdir(exist_ok=True)

    def published(name: str, mutate=None) -> Path:
        tensors = load_file(str(SIDECAR))
        if mutate:
            mutate(tensors)
        path = work / name
        save_file(tensors, str(path))
        return path

    def derived(name: str, **kwargs) -> Path:
        path = work / name
        write_derived(path, **kwargs)
        return path

    tracer = work / "TRACER-stage.safetensors"
    tensors, metadata = tracer_tool.build(list(range(4, 37)), HIDDEN, 1 / 64)
    save_file(tensors, str(tracer), metadata=metadata)
    garbage = work / "not-a-sidecar.safetensors"
    garbage.write_bytes(b"this is not a safetensors file at all")

    # What shows in the header: the two must agree, with and without the permission.
    visible = [
        published("published.safetensors"),
        published("missing.safetensors", lambda t: t.pop("layers.10.alpha")),
        published("extra.safetensors", lambda t: t.update({"layers.37.alpha": torch.tensor(1.0)})),
        published("dtype.safetensors", lambda t: t.update({"layers.9.direction": t["layers.9.direction"].float()})),
        published("alpha-shape.safetensors", lambda t: t.update({"layers.9.alpha": torch.ones(2)})),
        derived("rebuilt.safetensors"),
        derived("open.safetensors", verdict="OPEN"),
        derived("no-format.safetensors", verdict=None),
        derived("no-write.safetensors", layers__4__attn__write=None),
        derived("bf16.safetensors", layers__4__attn__read=torch.randn(HIDDEN).to(torch.bfloat16)),
        derived("unexpected.safetensors", layers__4__attn__direction=torch.randn(HIDDEN)),
        derived("sizes.safetensors", layers__20__shared__read=torch.randn(64), layers__20__shared__write=torch.randn(64)),
        tracer,
        garbage,
    ]
    taken = 0
    for path in visible:
        for allow in (False, True):
            staged, loaded, said = decide(path, allow)
            assert staged == loaded, (path.name, allow, staged, loaded, said)
            taken += staged
    assert taken == 6, taken  # published, rebuilt (twice each), open and the tracer with the permission
    assert decide(tracer, False)[2].count("tracer") == 1 and "TRACER sidecar" in decide(tracer, True)[2]

    # What does not show there is the loader's alone: the stage check passes these on.
    for path in (
        published("norm.safetensors", lambda t: t.update({"layers.9.direction": t["layers.9.direction"] * 2})),
        published("alpha-zero.safetensors", lambda t: t.update({"layers.10.alpha": torch.tensor(0.0)})),
        derived("nan.safetensors", layers__4__attn__read=torch.full((HIDDEN,), float("nan"))),
        derived("zero-write.safetensors", layers__4__shared__write=torch.zeros(HIDDEN)),
        derived("gain.safetensors", layers__4__attn__row_gain=torch.zeros(HIDDEN)),
    ):
        staged, loaded, said = decide(path, True)
        assert staged and not loaded, (path.name, staged, loaded, said)

    # The staging mistake of 2026-11-20: a derived sidecar next to a module that reads only
    # the published format. Named as that, whatever the permission.
    old_module = work / "pp2_steering_published_only.py"
    old_module.write_text("SIDECAR_LAYERS = range(4, 37)\n")
    for allow in (False, True):
        try:
            sc.check(str(tracer), str(old_module), allow)
        except sc.Refused as exc:
            assert "reads only the published format" in str(exc) and "revision 3 or later" in str(exc), exc
        else:
            raise AssertionError("a derived sidecar passed next to a module that cannot read it")
    assert "published sidecar" in sc.check(str(visible[0]), str(old_module), False)  # which that module does read
    # The table of shipped modules knows this one: change the module and this line says so.
    digest, revision, reads_derived = sc.module_facts(module)
    assert revision.startswith("revision 4") and reads_derived, (digest, revision)
    assert sc.KNOWN_MODULES[digest] == revision
    run = subprocess.run([sys.executable, "-B", str(HERE / "stage_check.py"), "--sidecar", str(tracer)],
                         capture_output=True, text=True, env={**os.environ, "PP2_STEERING_ALLOW_UNVERIFIED": "1"})
    assert run.returncode == 0 and run.stdout.startswith("STAGE-CHECK OK: TRACER sidecar"), run.stdout + run.stderr
    run = subprocess.run([sys.executable, "-B", str(HERE / "stage_check.py"), "--sidecar", str(tracer)],
                         capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k != "PP2_STEERING_ALLOW_UNVERIFIED"})
    assert run.returncode == 1 and run.stdout.startswith("STAGE-CHECK REFUSED:"), run.stdout + run.stderr


def test_tensor_parallel_needs_the_explicit_opt_in():
    fake_parallel.GROUP.rank_in_group = 0
    model = VisionWrapper(LanguageModel(0, 20))
    assert ps.Config.from_env(ENV).allow_tp is False
    allowed = ps.Config.from_env({**ENV, "PP2_STEERING_ALLOW_TP": "1"})
    assert allowed.allow_tp is True
    state_cls = ps.make_state_class(fake_state.DeepseekV41ModelState, allowed)
    state = state_cls(vllm_config(tp=4), model, None, torch.device("cpu"))
    assert state.steering.owned == list(range(4, 20))
    try:  # the other refusals do not depend on it
        state_cls(vllm_config(tp=4, use_ubatching=True), VisionWrapper(LanguageModel(0, 20)), None, "cpu")
    except RuntimeError as exc:
        assert "microbatching" in str(exc)
    else:
        raise AssertionError("microbatching was accepted")


def test_the_tools_derived_sidecar_applies_the_float_build():
    """verify/sidecar_vs_weights.py --derive and this module, end to end on a synthetic
    checkpoint built with a row-normalised edit: the operators the tool writes, applied by
    steer() to the stock writer's output, give the abliterated float writer's output."""
    tool = HERE.parent.parent / "verify" / "sidecar_vs_weights.py"
    if not tool.is_file():
        print("  (skipped: verify/sidecar_vs_weights.py is not two levels up)")
        return
    spec = importlib.util.spec_from_file_location("sidecar_vs_weights", tool)
    svw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svw)
    torch.manual_seed(31)
    made = svw.synthetic(
        str(TMP), "tool", lambda S, d, site, layer: svw.row_normalised_build(S, d, 3.0),
        svw.symmetric_sidecar, hidden=256, layers=(4,), in_attn=192, in_shared=128,
    )
    result = svw.analyse(made["stock"], made["ablit"], made["sidecar"], quiet=True, derive=made["derived"])
    assert result["verdict"] == "REBUILT", result["detail"]
    layers = ps.load_sidecar(made["derived"], "cpu")
    assert sorted(layers) == [4] and set(layers[4]) == {"attn", "shared"}
    for (layer, name), (S, X, d) in made["truth"].items():
        op = layers[layer]["attn" if name.startswith("attn") else "shared"]
        inputs = torch.randn(S.shape[1], 7)
        stock_out = (S @ inputs).T.to(torch.bfloat16)        # what the stock writer emits
        target = (X @ inputs).T                              # what the abliterated writer would
        steered = ps.steer(stock_out, op, torch.ones(7, 1)).float()
        edit = (target - stock_out.float()).norm()
        assert float((steered - target).norm() / edit) < 0.03, float((steered - target).norm() / edit)
        # ... and the published-style symmetric operator is a different edit.
        symmetric = stock_out.float() - 3.0 * torch.outer(stock_out.float() @ d, d)
        assert float((symmetric - target).norm() / edit) > 0.2


def test_inspect_prints_the_inventory(capsys=None):
    assert ps._inspect(str(SIDECAR)) == 0
    op = ps.load_sidecar(str(SIDECAR), "cpu")[4]["shared"]
    assert math.isclose(-float(op.write @ op.read), 3.0, rel_tol=5e-3)  # alpha, for a unit direction


# ----------------------------------------------------------------------- decoder replay kept


import vllm.forward_context as fake_context  # noqa: E402
from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers  # noqa: E402


class ReplayDecoder(Decoder):
    """v0.34.0's forward: the layers before the cut on every row, the rest through
    DecoderReplayLayers, which runs them on the step's replay batch when there is one."""

    def __init__(self, start: int, end: int, cut: int, window: int):
        super().__init__(start, end)
        self.decoder_replay_start = cut
        self.kv_writes: list[int] = []
        self.main_context = fake_context.ForwardContext({"whole batch": True})
        self.decoder_replay_layers = DecoderReplayLayers(
            window, self._run_replay_layers, lambda x, positions: self.kv_writes.append(x.shape[0])
        )

    def _run_replay_layers(self, x, positions):
        for index in range(self.decoder_replay_start, self.end_layer):
            x = self.layers[index](x, positions)
        return (x,)

    def forward(self, x, positions):
        with fake_context.override_forward_context(self.main_context):  # the runner's context
            for index in range(self.start_layer, self.decoder_replay_start):
                x = self.layers[index](x, positions)
            (x,) = self.decoder_replay_layers(x, positions)
        return x


class ReplayModel(torch.nn.Module):
    """The top-level model as the model state sees it: it exposes the replay layers."""

    def __init__(self, start: int, end: int, cut: int, window: int):
        super().__init__()
        self.language_model = types.SimpleNamespace(model=ReplayDecoder(start, end, cut, window))

    def get_language_model(self):
        return self.language_model

    @property
    def decoder_replay_layers(self):
        return self.language_model.model.decoder_replay_layers


def replay_batch_of(lens, state_indices, prefilling, padding=0):
    out = batch(lens, state_indices, padded=sum(lens))
    out.is_prefilling, out.replay_padding = prefilling, padding
    return out


def keep_state(start=16, end=40, cut=20, window=6, rank=1, **config):
    fake_parallel.GROUP.rank_in_group = rank
    model = ReplayModel(start, end, cut, window)
    state_cls = ps.make_state_class(
        fake_state.DeepseekV41ModelState, dataclasses.replace(CONFIG, replay="keep", **config)
    )
    return model, state_cls(vllm_config(), model, None, torch.device("cpu"))


def replay_reference(decoder, x, positions, token_scale, rows, replay_scale, sidecar):
    """ReplayDecoder.forward with the steering written out: the whole batch's scales
    before the cut, `replay_scale` on the kept rows after it."""

    def run(indices, x, scale):
        for index in indices:
            layer = decoder.layers[index]
            attn = layer.attn.project(x)
            if index in sidecar:
                attn = ps.steer(attn, sidecar[index]["attn"], scale)
            x = x + attn
            shared = layer.ffn.shared_experts.project(x)
            if index in sidecar:
                shared = ps.steer(shared, sidecar[index]["shared"], scale)
            x = x + (layer.ffn.routed.project(x) + shared)
        return x

    x = run(range(decoder.start_layer, decoder.decoder_replay_start), x, token_scale)
    if rows is None:
        return run(range(decoder.decoder_replay_start, decoder.end_layer), x, token_scale)
    kept = run(range(decoder.decoder_replay_start, decoder.end_layer), x.index_select(0, rows), replay_scale)
    return x.new_zeros(x.shape).index_copy_(0, rows, kept)


def test_kept_decoder_replay_steers_the_replay_batch_with_its_own_rows():
    sidecar = ps.load_sidecar(str(SIDECAR), "cpu")
    model, state = keep_state()
    decoder, steering = model.language_model.model, state.steering
    # Hooks of layers 16-19 always see the whole batch; those of 20-36 may see the replay batch.
    assert steering.replay is not None and steering.owned == list(range(16, 37))
    state.add_request(3, new_request(steering_scale=1.0))  # prefill of 14 rows: trimmed to 6
    state.add_request(0, new_request())                    # prefill of 9 rows at scale 0: trimmed to 6
    state.add_request(5, new_request(steering_scale=0.5))  # decode, 2 rows: never trimmed
    step = replay_batch_of([14, 9, 2], [3, 0, 5], [True, True, False], padding=3)
    steering.replay.token_scale.fill_(9.0)  # stale values from an earlier step
    state._prepare_replay_batch(step)       # prepare_attn runs first ...
    state.prepare_inputs(step, None)        # ... then prepare_inputs
    rows = torch.tensor([*range(8, 14), *range(17, 23), 23, 24])
    assert torch.equal(decoder.decoder_replay_layers.replay_batch.rows, rows)
    token_scale = torch.tensor([1.0] * 14 + [0.0] * 9 + [0.5] * 2).unsqueeze(1)
    replay_scale = token_scale.index_select(0, rows)
    assert torch.equal(steering.token_scale[:25], token_scale)
    assert torch.equal(steering.replay.token_scale[:14], replay_scale)
    assert steering.replay.token_scale[14:17, 0].tolist() == [0, 0, 0]  # padding rows of the sub-batch
    assert steering.replay.token_scale[17, 0] == 9.0                    # rows past it are not touched
    assert steering.replay.stock_rows[:17, 0].tolist() == [False] * 6 + [True] * 6 + [False] * 2 + [True] * 3
    x = torch.randn(25, HIDDEN, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(7))
    positions = torch.arange(25)
    hooked = decoder(x, positions)
    assert decoder.kv_writes == [25]  # the first replay layer still wrote KV for every row
    assert bool(torch.isfinite(hooked.float()).all())
    right = replay_reference(decoder, x, positions, token_scale, rows, replay_scale, sidecar)
    assert torch.equal(hooked, right)
    # What revision 2 would have done had replay been left on: the first 14 entries of the
    # whole batch's buffer, all 1.0, for the sub-batch's rows. The test has to tell the two apart.
    wrong = replay_reference(decoder, x, positions, token_scale, rows, token_scale[:14], sidecar)
    assert not torch.equal(wrong, right)
    stock = replay_reference(decoder, x, positions, torch.zeros(25, 1), rows, torch.zeros(14, 1), sidecar)
    for row in range(17, 23):  # the scale-0 request's kept rows: stock bits
        assert torch.equal(hooked[row], stock[row])
    assert not torch.equal(hooked[8], stock[8]) and not torch.equal(hooked[24], stock[24])

    # A step that trims nothing has no replay batch: every layer runs on the whole batch.
    step = replay_batch_of([1, 2], [3, 5], [False, False])
    state._prepare_replay_batch(step)
    state.prepare_inputs(step, None)
    assert decoder.decoder_replay_layers.replay_batch is None
    x = torch.randn(3, HIDDEN, dtype=torch.bfloat16, generator=torch.Generator().manual_seed(8))
    scale = torch.tensor([1.0, 0.5, 0.5]).unsqueeze(1)
    assert torch.equal(decoder(x, torch.arange(3)), replay_reference(decoder, x, None, scale, None, None, sidecar))
    # A replay batch left over from another forward does not capture a whole-batch call:
    # the call is told apart by the forward context it runs under.
    stale = replay_batch_of([14, 2], [0, 5], [True, False])
    state._prepare_replay_batch(stale)
    decoder.decoder_replay_layers.replay_batch, leftover = None, decoder.decoder_replay_layers.replay_batch
    state.prepare_inputs(step, None)
    whole = decoder(x, torch.arange(3))
    decoder.decoder_replay_layers.replay_batch = leftover
    steering.replay.token_scale.fill_(0.25)
    with fake_context.override_forward_context(decoder.main_context):
        again = decoder._run_replay_layers(x, torch.arange(3))[0]
    reference = x
    for index in range(20, 40):
        layer = decoder.layers[index]
        attn = layer.attn.project(reference)
        if index in sidecar:
            attn = ps.steer(attn, sidecar[index]["attn"], scale)
        reference = reference + attn
        shared = layer.ffn.shared_experts.project(reference)
        if index in sidecar:
            shared = ps.steer(shared, sidecar[index]["shared"], scale)
        reference = reference + (layer.ffn.routed.project(reference) + shared)
    assert torch.equal(again, reference) and bool(torch.isfinite(whole.float()).all())
    # ... and a context that only shares the replay batch's attention metadata (a graph
    # capture works on a copy of the context) is the replay batch.
    copy = dataclasses.replace(leftover.forward_context, cudagraph_runtime_mode=2)
    with fake_context.override_forward_context(copy):
        assert steering._current() is steering.replay
    with fake_context.override_forward_context(decoder.main_context):
        assert steering._current() is steering.main
    assert steering._current() is steering.main  # no forward context at all


def test_kept_decoder_replay_refuses_what_it_does_not_cover():
    def raises(build, needle):
        try:
            build()
        except RuntimeError as exc:
            assert needle in str(exc), exc
            return
        raise AssertionError(f"started with {needle}")

    config = dataclasses.replace(CONFIG, replay="keep")
    fake_parallel.GROUP.rank_in_group = 1
    # Replay graphs (a single-stage lane): the replay batch's rows are fixed at capture.
    with_graphs = type("WithGraphs", (fake_state.DeepseekV41ModelState,), {})
    with_graphs.__module__, with_graphs.__name__ = ps.STATE_BASE
    original = fake_state.DeepseekV41ModelState.__init__

    def init(self, *args):
        original(self, *args)
        self.replay_cudagraphs = object()

    with_graphs.__init__ = init
    raises(lambda: ps.make_state_class(with_graphs, config)(vllm_config(), ReplayModel(16, 40, 20, 6), None, "cpu"),
           "CUDA graphs of its own")
    # A tree that moved one of the names the replay batch is followed through.
    moved = type("Moved", (fake_state.DeepseekV41ModelState,), {})
    moved.__module__, moved.__name__ = ps.STATE_BASE

    def init_without(self, *args):
        original(self, *args)
        del self._kept_query_start_loc

    moved.__init__ = init_without
    raises(lambda: ps.make_state_class(moved, config)(vllm_config(), ReplayModel(16, 40, 20, 6), None, "cpu"),
           "_kept_query_start_loc")
    # A stage that begins at the last KV source layer (rank 1 of a 20/20 split): vLLM
    # v0.34.0 dies in its first forward there with replay on, so "keep" stops with the reason.
    raises(lambda: ps.make_state_class(fake_state.DeepseekV41ModelState, config)(
        vllm_config(), ReplayModel(20, 40, 20, 6), None, "cpu"), "begins at the last KV source layer")
    # With replay off, a stage that still replays is refused as before.
    raises(lambda: ps.make_state_class(fake_state.DeepseekV41ModelState, CONFIG)(
        vllm_config(), ReplayModel(20, 40, 20, 6), None, "cpu"), "decoder bounded replay is still on")
    # KV-sharing fast prefill skips rows the same way and is not followed.
    fast = vllm_config()
    fast.cache_config = types.SimpleNamespace(kv_sharing_fast_prefill=True)
    raises(lambda: fake_states.init_model_state(fast, VisionWrapper(LanguageModel(0, 20)), None, "cpu"), "fast prefill")
    # A stage without replay layers (rank 0 of the pair) starts with "keep" and has one buffer.
    fake_parallel.GROUP.rank_in_group = 0
    plain = VisionWrapper(LanguageModel(0, 20))
    state = ps.make_state_class(fake_state.DeepseekV41ModelState, config)(vllm_config(), plain, None, "cpu")
    assert state.steering.replay is None and len(state.steering.handles) == 32
    # "keep" leaves upstream's own decision alone; "off" overrides it.
    probe = type("D", (), {"_decoder_replay_supported": lambda self, *a, **k: True})
    ps._patch_decoder_replay(types.SimpleNamespace(DeepseekV4Model=probe), config)
    assert probe()._decoder_replay_supported(None, 20) is True


def test_replay_and_kernel_knobs_are_validated_at_interpreter_start():
    for name, value, needle in (("PP2_STEERING_REPLAY", "sometimes", "PP2_STEERING_REPLAY"),
                                ("PP2_STEERING_KERNEL", "cuda", "PP2_STEERING_KERNEL")):
        try:
            ps.Config.from_env({**ENV, name: value})
        except ValueError as exc:
            assert needle in str(exc), exc
        else:
            raise AssertionError(f"accepted {name}={value}")
    config = ps.Config.from_env({**ENV, "PP2_STEERING_REPLAY": "keep", "PP2_STEERING_KERNEL": "fused"})
    assert (config.replay, config.kernel) == ("keep", "fused")
    assert (CONFIG.replay, CONFIG.kernel) == ("off", "torch")  # unset means what revision 2 did
    assert ps.Config.from_env({**ENV, "PP2_STEERING_REPLAY": "", "PP2_STEERING_KERNEL": ""}).kernel == "torch"


# ----------------------------------------------------------------------- fused kernel


def fused_kernel_body() -> None:
    """Run by test_fused_kernel_matches_the_torch_path in a process with TRITON_INTERPRET=1."""
    sidecar = ps.load_sidecar(str(SIDECAR), "cpu")
    generator = torch.Generator().manual_seed(11)
    scale = torch.tensor([0.0, 1.0, 0.5, 0.0, 0.25, 1.0, 0.0]).unsqueeze(1)
    stock = scale.view(-1) == 0
    fake_parallel.GROUP.rank_in_group = 0
    model = VisionWrapper(LanguageModel(0, 20))
    state = ps.make_state_class(fake_state.DeepseekV41ModelState, dataclasses.replace(CONFIG, kernel="fused"))(
        vllm_config(), model, None, torch.device("cpu"))
    runtime = state.steering
    assert runtime.kernel is not None and len(runtime.handles) == 32
    buffers = ps._Buffers(16, "cpu")
    buffers.token_scale[:7] = scale
    buffers.stock_rows[:7] = scale == 0
    for dtype, tolerance in ((torch.float32, 2e-6), (torch.bfloat16, None)):
        for site in (sidecar[4]["attn"], sidecar[4]["shared"]):  # with and without a row gain
            x = torch.randn(7, HIDDEN, generator=generator).to(dtype)
            torch_path = ps.steer(x, site, scale)
            fused = runtime._apply(x.clone(), site, buffers)
            assert torch.equal(fused[stock], x[stock]), "a stock row was written"
            assert not torch.equal(fused[~stock], x[~stock])
            difference = (fused.float() - torch_path.float()).abs()
            if tolerance is None:
                # The interpreter truncates where the GPU rounds: one unit in the last place.
                ulp = torch.exp2(torch.floor(torch.log2(torch_path.float().abs().clamp_min(1e-30))) - 7)
                assert bool((difference <= ulp).all()), float((difference / ulp).max())
            else:
                assert float(difference.max()) <= tolerance * float(torch_path.abs().max())
    # In place: the hook hands back the tensor it was given.
    x = torch.randn(7, HIDDEN, generator=generator).to(torch.bfloat16)
    assert runtime._apply(x, sidecar[4]["shared"], buffers) is x
    # Through the hooks of a whole stage: stock rows come out with stock bits.
    state.add_request(0, new_request(steering_scale=1.0))
    state.add_request(1, new_request())
    state.prepare_inputs(batch([2, 3], [0, 1], padded=6), None)
    decoder = model.language_model.model
    x = torch.randn(6, HIDDEN, dtype=torch.bfloat16, generator=generator)
    positions = torch.arange(6)
    hooked = decoder(x.clone(), positions)
    unsteered = manual_reference(decoder, x, positions, torch.zeros(6, 1), sidecar)
    steered = manual_reference(decoder, x, positions, torch.tensor([1, 1, 0, 0, 0, 0.0]).unsqueeze(1), sidecar)
    assert torch.equal(hooked[2:], unsteered[2:])
    assert not torch.equal(hooked[:2], unsteered[:2])
    relative = float((hooked[:2].float() - steered[:2].float()).norm() / steered[:2].float().norm())
    assert relative < 0.02, relative
    # An output the kernel does not take falls back to the torch path, and says so once.
    logged = []
    keep, ps._log = ps._log, logged.append
    try:
        wide = torch.randn(7, HIDDEN, generator=generator).double()
        assert torch.equal(runtime._apply(wide.clone(), sidecar[4]["shared"], buffers),
                           ps.steer(wide, sidecar[4]["shared"], scale))
        runtime._apply(wide.clone(), sidecar[4]["shared"], buffers)
    finally:
        ps._log = keep
    assert len(logged) == 1 and "torch path" in logged[0]
    print("fused kernel body ok")


def test_fused_kernel_matches_the_torch_path():
    if importlib.util.find_spec("triton") is None:
        print("      (skipped: Triton is not installed here, so the fused kernel was not run)")
        return
    run = subprocess.run(
        [sys.executable, "-B", "-c", "import test_pp2_steering as t; t.fused_kernel_body()"],
        cwd=str(HERE), capture_output=True, text=True, timeout=900,
        env={**os.environ, "TRITON_INTERPRET": "1", "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert run.returncode == 0 and "fused kernel body ok" in run.stdout, (run.stdout + run.stderr)[-1500:]


if __name__ == "__main__":
    failures = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    # Run the install-order test first: it depends on what has been imported so far.
    tests.sort(key=lambda item: item[0] != "test_install_patches_modules_imported_before_and_after")
    for name, function in tests:
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failures} passed, {failures} failed")
    raise SystemExit(1 if failures else 0)
