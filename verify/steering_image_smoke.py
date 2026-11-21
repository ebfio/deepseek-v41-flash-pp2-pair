#!/usr/bin/env python3
"""Steering, between the static seam check and a boot: do the patches work on the REAL
vLLM classes of this image? CPU only: no GPU, no weights, no engine, so it can run on a
box whose GPU is busy serving.

Run it in the image with the same mounts and environment serve_vllm_node.sh sets up:

    SIDECAR=/path/to/warmind-v1.1-orth4c-steering.safetensors
    docker run --rm --entrypoint python3 \
        -e CUDA_VISIBLE_DEVICES= -e PYTHONPATH=/opt/pp2-flag \
        -e PP2_STEERING_SIDECAR=/opt/pp2-steering-sidecar/steering.safetensors \
        -e PP2_STEERING_SHA256="$(sha256sum "$SIDECAR" | cut -d' ' -f1)" \
        -e PP2_STEERING_DEBUG=1 \
        -v "$PWD/patches/vllm-pp2-flag-v034:/opt/pp2-flag:ro" \
        -v "$PWD/patches/pp2-steering:/opt/pp2-steering:ro" \
        -v "$SIDECAR:/opt/pp2-steering-sidecar/steering.safetensors:ro" \
        -v "$PWD/verify:/verify:ro" \
        vllm/vllm-openai:v0.34.0 /verify/steering_image_smoke.py

What it covers, on real classes: interpreter start stays cheap and steering is installed
before any vLLM import; all five seams get patched; the chat and completion request
classes validate and forward the scale; a steered request gets its own prefix-cache block
hashes from the real block hasher and a stock request keeps its old ones; the model-state
resolver returns a subclass of the real DeepseekV41ModelState and refuses the specialized
decode state; the per-token gather reads a real InputBatch; the real DecoderReplayLayers
runs its replay batch under a forward context the hooks can tell from the whole batch's
(what VLLM_STEERING_REPLAY=keep relies on); the real SharedExperts wrapper reaches its
inner module through __call__ on both of its paths (read from its source: the wrapper is
not constructed, because its constructor asks for a CUDA stream); the fused kernel
compiles with this image's Triton for the GB300 without a GPU; the sidecar loads.

Add -e PP2_STEERING_REPLAY=keep and/or -e PP2_STEERING_KERNEL=fused to check the image the
way the launcher would start it with those knobs.

It must never touch CUDA: the last check fails the run if anything initialised it.
Revision 2 of this script constructed SharedExperts and died in the image with
"no CUDA-capable device is detected"; that was this script's defect, not the image's.

What it cannot cover: the real decoder, CUDA graphs, the auxiliary stream, two stages.
That is verify/steering_probe.py on a booted lane.
"""

from __future__ import annotations

import dataclasses
import inspect
import os
import sys
import time
import types

HEAVY_AT_START = sorted(m for m in ("torch", "vllm", "transformers") if m in sys.modules)
INSTALLED_AT_START = "pp2_steering" in sys.modules

RESULTS: list[tuple[str, str, str]] = []


def record(ok: bool, name: str, detail: str, required: bool = True) -> None:
    status = "ok" if ok else ("FAIL" if required else "warn")
    RESULTS.append((status, name, detail))
    print(f"{status:<4}  {name:<46}  {detail}", flush=True)


def attempt(name: str, function) -> None:
    try:
        detail = function()
    except BaseException as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {str(exc)[:260]}"
        if "CUDA" in text or "cuda" in text:
            text += "  [this check tried to use a GPU: a defect of this script, report it]"
        record(False, name, text)
    else:
        record(True, name, detail or "")


def main() -> int:
    record(not HEAVY_AT_START, "interpreter start imports nothing heavy",
           "torch, vllm and transformers not imported" if not HEAVY_AT_START
           else f"sitecustomize imported {', '.join(HEAVY_AT_START)}")
    record(INSTALLED_AT_START, "steering installed by sitecustomize",
           "pp2_steering was imported before this script ran" if INSTALLED_AT_START
           else "pp2_steering is not loaded: is PP2_STEERING_SIDECAR set and PYTHONPATH the overlay dir?")
    if not INSTALLED_AT_START:
        print("\nSTEERING-IMAGE FAIL: steering is not installed in this interpreter")
        return 1
    import pp2_steering as ps

    config = ps.install()

    started = time.time()
    import torch

    import vllm.entrypoints.openai.chat_completion.protocol as chat_protocol
    import vllm.entrypoints.openai.completion.protocol as completion_protocol
    import vllm.models.deepseek_v41.nvidia.model as model_module
    import vllm.v1.request as request_module
    import vllm.v1.worker.gpu.model_states as model_states
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingParams

    status = ps.status()
    record(sorted(status["patched"]) == sorted(ps.PATCHES) and not status["failed"],
           "all five seams patched",
           f"patched {len(status['patched'])}/{len(ps.PATCHES)}, failed {status['failed']}, "
           f"pending {status['pending']}; imports took {time.time() - started:.1f} s")
    record(not torch.cuda.is_initialized(), "imports left CUDA alone",
           f"torch {torch.__version__}, cuda initialised: {torch.cuda.is_initialized()}", required=False)

    # ---- API server ------------------------------------------------------------------
    def chat(**fields):
        return chat_protocol.ChatCompletionRequest(
            model="m", messages=[{"role": "user", "content": "hi"}], **fields
        )

    def api():
        plain = chat().to_sampling_params(16, {})
        assert not (plain.extra_args or {}).get("steering_scale"), plain.extra_args
        xargs = chat(vllm_xargs={"steering_scale": 1}).to_sampling_params(16, {})
        assert xargs.extra_args == {"steering_scale": 1.0}, xargs.extra_args
        top = chat(steering_scale=0.5).to_sampling_params(16, {})
        assert top.extra_args == {"steering_scale": 0.5}, top.extra_args
        completion = completion_protocol.CompletionRequest(
            model="m", prompt="hi", vllm_xargs={"steering_scale": "0.25"}
        ).to_sampling_params(16)
        assert completion.extra_args == {"steering_scale": 0.25}, completion.extra_args
        assert isinstance(xargs, SamplingParams)
        return "omitted, vllm_xargs, top-level and completion spellings reach SamplingParams.extra_args"

    attempt("API: the scale reaches extra_args", api)

    def api_rejects():
        rejected = []
        for fields in ({"vllm_xargs": {"steering_scale": 1.5}}, {"vllm_xargs": {"steering_scale": "abc"}},
                       {"vllm_xargs": {"steering_scale": -0.1}}, {"steering_scale": 2}):
            try:
                chat(**fields).to_sampling_params(16, {})
            except VLLMValidationError as exc:
                assert exc.parameter == "steering_scale", exc.parameter
                rejected.append(fields)
        assert len(rejected) == 4, f"only {len(rejected)} of 4 bad scales were rejected"
        if not config.debug:
            try:
                chat(vllm_xargs={"steering_scale": 1, "steering_ranks": 1}).to_sampling_params(16, {})
            except VLLMValidationError:
                return "4 bad scales and the debug knob raise VLLMValidationError (HTTP 400)"
            raise AssertionError("steering_ranks was accepted with debug off")
        masked = chat(vllm_xargs={"steering_scale": 1, "steering_ranks": 2}).to_sampling_params(16, {})
        assert masked.extra_args == {"steering_scale": 1.0, "steering_ranks": 2}, masked.extra_args
        return "4 bad scales raise VLLMValidationError (HTTP 400); steering_ranks accepted (debug on)"

    attempt("API: bad values are client errors", api_rejects)

    # ---- engine core -----------------------------------------------------------------
    def engine():
        from vllm.utils.hashing import sha256
        from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
        from vllm.v1.engine import EngineCoreRequest

        init_none_hash(sha256)  # the engine core does this once at start-up
        hasher = get_request_block_hasher(16, sha256)
        tokens = list(range(1000, 1000 + 80))

        def build(salt, block_hashes=None, **extra):
            core = EngineCoreRequest(
                request_id=f"r-{len(extra)}-{salt}",
                prompt_token_ids=list(tokens),
                mm_features=None,
                sampling_params=SamplingParams(max_tokens=8, extra_args=extra or None),
                pooling_params=None,
                arrival_time=0.0,
                lora_request=None,
                cache_salt=salt,
                data_parallel_rank=None,
                block_hashes=block_hashes,
            )
            return request_module.Request.from_engine_core_request(core, hasher), core

        stock, _ = build(None)
        zero, _ = build(None, steering_scale=0.0)
        one, _ = build(None, steering_scale=1.0)
        again, _ = build(None, steering_scale=1.0)
        half, _ = build(None, steering_scale=0.5)
        masked, _ = build(None, steering_scale=1.0, steering_ranks=1)
        user, _ = build("tenant", steering_scale=1.0)
        hashes = lambda r: [bytes(h) for h in r.block_hashes]  # noqa: E731
        assert len(hashes(stock)) == 5, len(hashes(stock))
        assert stock.cache_salt is None and zero.cache_salt is None
        assert hashes(zero) == hashes(stock), "scale 0 must stay in the stock cache namespace"
        assert hashes(one) == hashes(again), "the cache must still work within one scale"
        distinct = {tuple(hashes(r)) for r in (stock, one, half, user) + ((masked,) if config.debug else ())}
        assert len(distinct) == (5 if config.debug else 4), "scales (and stage masks) must not share blocks"
        assert not set(hashes(one)) & set(hashes(stock)), "no block of a steered request may match stock"
        assert user.cache_salt.startswith("tenant")
        # Hashes a frontend computed without the salt are dropped for a steered request.
        stale = hashes(stock)
        steered, core = build(None, block_hashes=stale, steering_scale=1.0)
        assert hashes(steered) == hashes(one) and core.block_hashes is None
        kept, _ = build(None, block_hashes=stale)
        assert hashes(kept) == stale
        return (f"{len(hashes(stock))} blocks; scale 0 = stock hashes; 0.5, 1 and a user salt each "
                f"get their own; salt suffix {one.cache_salt!r}")

    attempt("engine: prefix-cache hashes are per scale", engine)

    # ---- worker ----------------------------------------------------------------------
    def resolver():
        from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState

        stand_in = types.SimpleNamespace(get_model_state_cls=lambda: DeepseekV41ModelState)
        cls = model_states.resolve_model_state_cls(None, stand_in)
        assert issubclass(cls, DeepseekV41ModelState) and cls is not DeepseekV41ModelState
        want = {"__init__": ["self", "vllm_config", "model", "encoder_cache", "device"],
                "add_request": ["self", "req_index", "new_req_data"],
                "prepare_inputs": ["self", "input_batch", "req_states"],
                "prepare_dummy_inputs": ["self", "num_reqs", "num_tokens"]}
        for name, params in want.items():
            got = list(inspect.signature(getattr(DeepseekV41ModelState, name)).parameters)
            assert got == params, f"{name}{got}"
            assert name in cls.__dict__, f"{name} is not overridden"
        replay_hook = getattr(DeepseekV41ModelState, "_prepare_replay_batch", None)
        if replay_hook is not None:
            got = list(inspect.signature(replay_hook).parameters)[:2]
            assert got == ["self", "input_batch"], f"_prepare_replay_batch{got}"
            assert "_prepare_replay_batch" in cls.__dict__, "_prepare_replay_batch is not overridden"
        refused = "no specialized decode state in this tree"
        try:
            from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime.model_state import (
                DeepseekV41SpecializedDecodeModelState as Special,
            )
        except ImportError:
            pass
        else:
            try:
                model_states.resolve_model_state_cls(None, types.SimpleNamespace(get_model_state_cls=lambda: Special))
            except RuntimeError:
                refused = "the specialized decode state is refused"
            else:
                raise AssertionError("the specialized decode model state was accepted")
        return f"{cls.__name__} subclasses the real class; {refused}"

    attempt("worker: model-state subclass", resolver)

    def replay_gate():
        gate = getattr(model_module.DeepseekV4Model, "_decoder_replay_supported", None)
        if gate is None:
            return "no decoder replay in this tree"
        # sitecustomize.py's boundary guard, when present, is the outer wrapper: it answers
        # "no" on a stage that begins at the last KV source layer and asks the rest inward.
        guarded = bool(getattr(gate, "_pp2_boundary_guard", False))
        inner = getattr(gate, "_pp2_upstream", gate)
        wrapped = "_patch_decoder_replay" in inner.__qualname__
        guard = ("; the boundary guard is around it" if guarded
                 else "; no boundary guard (VLLM_PATCH_PP_REPLAY_BOUNDARY=0, or an overlay older than revision 4)")
        if config.replay == "keep":
            assert not wrapped, "replay is to be kept, and the gate is wrapped"
            return "replay kept: DeepseekV4Model._decoder_replay_supported is upstream's own" + guard
        assert wrapped, inner.__qualname__
        return "replay off: DeepseekV4Model._decoder_replay_supported is the steering wrapper" + guard

    attempt("worker: decoder replay gate", replay_gate)

    def replay_batch():
        from vllm.config.compilation import CUDAGraphMode
        from vllm.forward_context import (
            BatchDescriptor,
            ForwardContext,
            get_forward_context,
            override_forward_context,
        )

        try:
            from vllm.models.deepseek_v41.decoder_replay_layers import DecoderReplayLayers, ReplayBatch
        except ImportError:
            return "no decoder replay in this tree"
        runtime = object.__new__(ps.SteeringRuntime)
        runtime.main, runtime.replay = ps._Buffers(8, "cpu"), ps._Buffers(8, "cpu")
        runtime._forward_context = get_forward_context
        seen = []

        def run_layers(hidden, positions):
            seen.append(("replay buffers" if runtime._current() is runtime.replay else "whole batch", hidden.shape[0]))
            return (hidden + 1.0,)

        layers = DecoderReplayLayers(
            4, run_layers, lambda hidden, positions: seen.append(("kv", hidden.shape[0])), set(), "swa"
        )
        runtime.replay_layers = layers
        whole = ForwardContext(no_compile_layers={}, attn_metadata={"whole": None}, slot_mapping={})
        own = ForwardContext(no_compile_layers={}, attn_metadata={"replay": None}, slot_mapping={},
                             batch_descriptor=BatchDescriptor(num_tokens=4))
        hidden, positions = torch.zeros(9, 3), torch.arange(9)
        with override_forward_context(whole):
            layers(hidden, positions)
            layers.replay_batch = ReplayBatch(torch.tensor([5, 6, 7, 8]), own)
            (out,) = layers(hidden, positions)
            assert runtime._current() is runtime.main, "outside the replay layers, with a replay batch set"
        assert seen == [("whole batch", 9), ("kv", 9), ("replay buffers", 4)], seen
        assert out[:5].abs().sum() == 0 and bool((out[5:] == 1.0).all()), "outputs are scattered to the kept rows"
        assert int(layers.replay_batch.forward_context.batch_descriptor.num_tokens) == 4
        # A replay-graph capture runs under a copy of the context.
        with override_forward_context(dataclasses.replace(own, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)):
            assert runtime._current() is runtime.replay
        assert runtime._current() is runtime.main, "no forward context at all"
        return ("real DecoderReplayLayers: whole batch without a replay batch, the first replay layer's KV "
                "for every row, then the replay layers on 4 kept rows under their own forward context")

    attempt("worker: the replay batch can be told apart", replay_batch)

    hidden = 5120

    class Inner(torch.nn.Module):
        def forward(self, x):
            return x * 0.5

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Inner()
            self.ffn = torch.nn.Module()
            self.ffn.shared_experts = Inner()

    class Decoder(torch.nn.Module):
        def __init__(self, start, end):
            super().__init__()
            self.config = types.SimpleNamespace(hidden_size=hidden)
            self.start_layer, self.end_layer = start, end
            self.layers = torch.nn.ModuleList(
                Layer() if start <= i < end else torch.nn.Identity() for i in range(40)
            )

    def gather():
        from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

        device = torch.device("cpu")
        model = types.SimpleNamespace(model=Decoder(20, 40))
        # The torch path: the fused kernel needs a GPU (it is compiled, not run, below).
        runtime = ps.SteeringRuntime(dataclasses.replace(config, kernel="torch"), model, device, 16, 256, 1)
        assert runtime.owned and all(20 <= layer < 40 for layer in runtime.owned), runtime.owned
        steered_layer = runtime.owned[0]
        buffers = InputBuffers(16, 256, device)
        batch = InputBatch.make_dummy(3, 11, buffers)
        starts = batch.query_start_loc[:4].tolist()
        for slot, scale in zip(batch.idx_mapping[:3].tolist(), (1.0, 0.0, 0.5)):
            runtime.set_request(slot, types.SimpleNamespace(extra_args={"steering_scale": scale}))
        runtime.token_scale.fill_(7.0)
        runtime.fill(batch)
        rows = batch.num_tokens_after_padding
        expected = [0.0] * rows
        for index, scale in enumerate((1.0, 0.0, 0.5)):
            for row in range(starts[index], starts[index + 1]):
                expected[row] = scale
        got = runtime.token_scale[:rows, 0].tolist()
        assert got == expected, (got, expected)
        assert runtime.stock_rows[:rows, 0].tolist() == [v == 0.0 for v in expected]
        x = torch.randn(rows, hidden).to(torch.bfloat16)
        layer = model.model.layers[steered_layer]
        steered = layer.ffn.shared_experts(x) if not layer.attn._forward_hooks else layer.attn(x)
        stock = x * 0.5
        changed = [not torch.equal(steered[i], stock[i]) for i in range(rows)]
        assert changed == [v != 0.0 for v in expected], changed
        return (f"real InputBatch ({batch.num_reqs} requests, {rows} rows, query_start_loc "
                f"{batch.query_start_loc.dtype}, idx_mapping {batch.idx_mapping.dtype}); only the "
                "steered rows change")

    attempt("worker: per-token gather on a real InputBatch", gather)

    def shared_wrapper():
        from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

        # Not constructed: SharedExperts.__init__ creates the auxiliary CUDA stream.
        found = []
        for method in ("maybe_forward_async", "forward"):
            source = inspect.getsource(getattr(SharedExperts, method))
            assert "self._layer(shared_experts_input)" in source, f"{method} does not call self._layer(...)"
            assert "self._layer.forward(" not in source, f"{method} bypasses Module.__call__"
            found.append(method)
        assert "self._layer = layer" in inspect.getsource(SharedExperts.__init__)
        inner = Inner()
        calls = []
        inner.register_forward_hook(lambda module, args, output: (calls.append(1), output + 1.0)[1])
        assert torch.equal(inner(torch.ones(2, 4)), torch.full((2, 4), 1.5)) and calls == [1]
        return f"{' and '.join(found)} call the module they were given through __call__, where hooks fire"

    attempt("worker: SharedExperts reaches hooks", shared_wrapper)

    def sidecar():
        layers = ps.load_sidecar(config.sidecar_path, "cpu", config.allow_unverified)
        operators = [op for sites in layers.values() for op in sites.values()]
        strengths = [-float(op.read @ op.write) for op in operators]
        kind = "derived (v2)" if ps.sidecar_metadata(config.sidecar_path).get("format") == ps.V2_FORMAT else "published (v1)"
        return (f"{kind}; layers {min(layers)}-{max(layers)}, {len(operators)} operators, hidden "
                f"{int(operators[0].read.shape[0])}, strength {min(strengths):.4g}..{max(strengths):.4g}, "
                f"sha256 {config.sha256[:16]}")

    def fused_kernel():
        import importlib.util

        if importlib.util.find_spec("triton") is None:
            raise RuntimeError("Triton is not installed in this image")
        import triton
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource, compile as triton_compile

        kernel = ps.load_kernel()
        built = []
        for dtype in ("*bf16", "*fp32"):
            for has_gain in (False, True):
                signature = {"x_ptr": dtype, "read_ptr": "*fp32", "write_ptr": "*fp32", "gain_ptr": "*fp32",
                             "scale_ptr": "*fp32", "row_stride": "i64", "H": "i32"}
                source = ASTSource(fn=kernel.steer_rows, signature=signature,
                                   constexprs={"HAS_GAIN": has_gain, "BLOCK": kernel.block_size(hidden)})
                compiled = triton_compile(source, target=GPUTarget("cuda", 103, 32))
                assert compiled.asm.get("cubin"), "no cubin"
                assert "cvt.rn.bf16.f32" in compiled.asm["ptx"] or dtype != "*bf16", "the BF16 store does not round to nearest"
                built.append(f"{dtype[1:]}{'+gain' if has_gain else ''}")
        return (f"Triton {triton.__version__}: steer_rows compiled for sm_103 ({', '.join(built)}); "
                "compiled only, never launched here")

    if config.kernel == "fused":
        attempt("fused kernel compiles for the GB300", fused_kernel)
    else:
        try:
            record(True, "fused kernel compiles for the GB300", fused_kernel() + " (not in use: kernel is torch)")
        except BaseException as exc:  # noqa: BLE001
            record(False, "fused kernel compiles for the GB300",
                   f"{type(exc).__name__}: {str(exc)[:200]} (not in use: kernel is torch)", required=False)

    attempt("sidecar loads and validates", sidecar)
    record(not torch.cuda.is_initialized(), "nothing here touched CUDA",
           "torch.cuda.is_initialized() is False" if not torch.cuda.is_initialized()
           else "a check initialised CUDA: a defect of this script, report it")

    failed = [r for r in RESULTS if r[0] == "FAIL"]
    print(f"\nSTEERING-IMAGE {'PASS' if not failed else 'FAIL'}: {len(failed)} required check(s) failed; "
          f"vllm {getattr(sys.modules['vllm'], '__version__', 'unknown')}, python {sys.version.split()[0]}, "
          f"debug {'on' if config.debug else 'off'}, replay {config.replay}, kernel {config.kernel}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
