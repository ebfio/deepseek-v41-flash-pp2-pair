#!/usr/bin/env python3
"""Step 0, the runtime half of fix 1: does sitecustomize.py set the flag in THIS image?

sitecustomize.py runs in every Python process of the container. It must do two things:
stay cheap (import nothing heavy at interpreter start) and set
supports_aux_hidden_states_over_pp on the inner decoder classes at the moment vLLM
imports them. If the flag is missing the engine dies seconds into load with
"does not support dspark with pipeline parallelism".

Run it inside the image, with the overlay directory mounted and on PYTHONPATH exactly
as serve_vllm_node.sh does it. It needs no GPU and loads no weights, so it can run on a
box whose GPU is busy serving:

    docker run --rm --entrypoint python3 \
        -e PYTHONPATH=/opt/pp2-flag \
        -v "$PWD/patches/vllm-pp2-flag-v034:/opt/pp2-flag:ro" \
        -v "$PWD/verify:/verify:ro" \
        vllm/vllm-openai:v0.34.0 /verify/step0_import_smoke.py

Exit code 0 means: nothing heavy was imported at interpreter start, and both inner
decoder classes carry the flag once their modules are imported.
"""

from __future__ import annotations

import os
import sys
import time

# What sitecustomize had imported before this script ran a single line of its own.
HEAVY_AT_START = sorted(m for m in ("torch", "vllm", "transformers") if m in sys.modules)

import importlib  # noqa: E402

# (module, class, required). The two inner decoders are what the PP gate reads
# (eagle3_utils._inner_decoder). The draft wrapper is belt and braces in
# sitecustomize.py. Its other wrapper target, DeepseekV41ForCausalLM, is not checked:
# that class lives in vl_model.py in both v0.31.0 and v0.34.0, so the entry never matches.
TARGETS = [
    ("vllm.models.deepseek_v41.nvidia.model", "DeepseekV4Model", True),
    ("vllm.models.deepseek_v41.nvidia.dspark", "DSparkDeepseekV4Model", True),
    ("vllm.models.deepseek_v41.nvidia.dspark", "DSparkDeepseekV4ForCausalLM", False),
]
FLAG = "supports_aux_hidden_states_over_pp"


def replay_boundary(line) -> int:
    """The guard on decoder replay, asked on the real class with made-up stages.

    v0.34.0 answers "supported" for a pipeline stage that begins at the last KV source
    layer and then fails that stage's first forward with an AssertionError that has no
    message. sitecustomize.py makes the answer "not supported" there and nowhere else.
    """
    from types import SimpleNamespace

    module = sys.modules.get("vllm.models.deepseek_v41.nvidia.model")
    cls = getattr(module, "DeepseekV4Model", None)
    method = getattr(cls, "_decoder_replay_supported", None)
    name = "decoder replay on a stage that begins at the cut"
    if method is None:
        line("ok", name, "this vLLM has no decoder-side replay: nothing to guard")
        return 0
    off = os.environ.get("VLLM_PATCH_PP_REPLAY_BOUNDARY", "1") == "0"
    upstream = getattr(method, "_pp2_upstream", None)
    if upstream is None and not off:
        line("FAIL", name, "sitecustomize.py did not wrap DeepseekV4Model._decoder_replay_supported")
        return 1

    def stage(start: int, end: int):
        return SimpleNamespace(
            start_layer=start, end_layer=end, use_sequence_parallel=False,
            config=SimpleNamespace(sliding_window=128, num_hidden_layers=40, engram_layer_ids=()),
        )

    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(use_ubatching=False), speculative_config=None)
    cut = 20
    try:
        answers = {span: bool(method(stage(*span), vllm_config, cut)) for span in ((20, 40), (16, 40), (0, 40), (24, 40))}
        alone = bool((upstream or method)(stage(20, 40), vllm_config, cut))
    except BaseException as exc:  # noqa: BLE001
        line("FAIL", name, f"the method could not be asked: {type(exc).__name__}: {exc}")
        return 1
    want = {(20, 40): off, (16, 40): True, (0, 40): True, (24, 40): False}
    good = answers == want and alone is True
    text = ", ".join(f"layers {a}-{b - 1}: {'on' if v else 'off'}" for (a, b), v in answers.items())
    line("ok" if good else "FAIL", name,
         f"last KV source {cut}; {text}; upstream alone answers {'on' if alone else 'off'} for layers 20-39"
         + (" (guard removed by VLLM_PATCH_PP_REPLAY_BOUNDARY=0)" if off else ""))

    # What that stage's first forward does with replay on: forward() calls _run_layers on
    # range(start_layer, decoder_replay_start), which is empty there. This runs upstream's
    # own _run_layers on that empty range, with a stand-in for the model object.
    stand_in = SimpleNamespace(_mega_gate_metadata=lambda ids: None, layers=[], aux_hidden_state_layers=(),
                               use_sequence_parallel=False)
    positions = SimpleNamespace(shape=(4,))
    try:
        cls._run_layers(stand_in, range(20, 20), object(), positions, None, object(), None, None, None, {})
        line("warn", "upstream's first forward on that stage",
             "_run_layers accepted an empty range: upstream changed, the guard may no longer be needed")
    except AssertionError as exc:
        line("ok", "upstream's first forward on that stage",
             f"_run_layers on the empty range raises AssertionError with the message {str(exc)!r}: the "
             "engine then reports \"Worker failed with error ''\"")
    except BaseException as exc:  # noqa: BLE001
        line("warn", "upstream's first forward on that stage",
             f"_run_layers on the empty range raises {type(exc).__name__}: {exc}")
    return 0 if good else 1


def main() -> int:
    failed = 0

    def line(status: str, name: str, detail: str) -> None:
        print(f"{status:<4}  {name:<44}  {detail}")

    site = sys.modules.get("sitecustomize")
    origin = getattr(site, "__file__", None)
    on_path = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    ours = origin is not None and any(
        os.path.dirname(os.path.abspath(origin)) == os.path.abspath(p) for p in on_path
    )
    if not ours:
        failed += 1
    line("ok" if ours else "FAIL", "sitecustomize loaded from PYTHONPATH",
         f"{origin} (PYTHONPATH={os.pathsep.join(on_path) or '<unset>'})")

    cheap = not HEAVY_AT_START
    if not cheap:
        failed += 1
    line("ok" if cheap else "FAIL", "interpreter start imports nothing heavy",
         "torch, vllm and transformers were not imported by sitecustomize" if cheap
         else f"sitecustomize imported {', '.join(HEAVY_AT_START)}: every process in the "
              "container pays for that")

    for module_name, class_name, required in TARGETS:
        name = f"{module_name.rsplit('.', 1)[-1]}.{class_name}"
        first = module_name not in sys.modules
        started = time.time()
        try:
            module = importlib.import_module(module_name)
        except BaseException as exc:  # noqa: BLE001 - report it, whatever it is
            if required:
                failed += 1
            line("FAIL" if required else "warn", name,
                 f"module does not import: {type(exc).__name__}: {exc}")
            continue
        took = f"; importing the module took {time.time() - started:.1f} s" if first else ""
        cls = getattr(module, class_name, None)
        if cls is None:
            if required:
                failed += 1
            line("FAIL" if required else "warn", name, "class not found in module")
            continue
        value = getattr(cls, FLAG, None)
        good = value is True
        if required and not good:
            failed += 1
        line("ok" if good else ("FAIL" if required else "warn"), name, f"{FLAG}={value!r}{took}")

    failed += replay_boundary(line)

    try:
        import torch

        cuda = torch.cuda.is_initialized()
        line("ok" if not cuda else "warn", "CUDA untouched by the imports",
             f"torch.cuda.is_initialized()={cuda}")
    except BaseException as exc:  # noqa: BLE001
        line("warn", "CUDA untouched by the imports", f"torch not importable: {exc}")

    try:
        import vllm

        line("ok", "vllm version", getattr(vllm, "__version__", "unknown"))
    except BaseException as exc:  # noqa: BLE001
        line("warn", "vllm version", f"{type(exc).__name__}: {exc}")

    print(f"\nSTEP0-IMPORT {'PASS' if not failed else 'FAIL'}: {failed} required check(s) failed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
