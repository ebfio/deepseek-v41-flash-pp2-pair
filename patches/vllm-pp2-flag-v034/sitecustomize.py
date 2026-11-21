# patches/vllm-pp2-flag-v034/ — the one-line PP2+DSpark overlay, for vLLM v0.34.0.
# The gate in vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py reads
# supports_aux_hidden_states_over_pp on the INNER decoder
# (model.get_language_model() -> .model), not the ForCausalLM wrapper, so
# the attr is set on DeepseekV4Model / DSparkDeepseekV4Model here.
# Loaded via PYTHONPATH sitecustomize at engine start, in EVERY Python process of the
# container. So this file imports nothing heavy: it registers an import hook, and the
# flag is set at the moment vLLM itself imports the model module. (The v0.31.0 file
# imports the model modules right here, which costs every process a full vllm + torch
# import: measured at 8.9 s per process in the 0.34.0 image, 2026-11-20.)

import importlib.util
import os
import sys

# Per-request steering (patches/pp2-steering). Only when the launcher set the sidecar.
# It runs first and it fails closed: when sitecustomize raises, Python prints
# "Error in sitecustomize" and carries on, which would boot an engine that serves
# every request stock while the operator believes steering is on.
if os.environ.get("PP2_STEERING_SIDECAR"):
    try:
        sys.path.append(os.environ.get("PP2_STEERING_HOME", "/opt/pp2-steering"))
        import pp2_steering

        pp2_steering.install()
    except BaseException as _exc:  # noqa: BLE001
        sys.stderr.write(
            f"[pp2_steering] FATAL: steering was requested and could not be installed: {_exc!r}\n"
        )
        sys.stderr.flush()
        os._exit(78)

_FLAG = "supports_aux_hidden_states_over_pp"
_TARGETS = {
    "vllm.models.deepseek_v41.nvidia.model": (
        "DeepseekV4Model",
        # Belt and braces: the wrapper, for any gate that reads it directly.
        "DeepseekV41ForCausalLM",
    ),
    "vllm.models.deepseek_v41.nvidia.dspark": (
        "DSparkDeepseekV4Model",
        "DSparkDeepseekV4ForCausalLM",
    ),
}


def _flip(module):
    for _class_name in _TARGETS.get(module.__name__, ()):
        _cls = getattr(module, _class_name, None)
        if _cls is not None:
            setattr(_cls, _FLAG, True)
    if module.__name__ == _MODEL_MODULE:
        _guard_replay_boundary(module)


# Decoder SWA bounded replay on a pipeline stage that BEGINS at the last KV source layer.
# v0.34.0 turns replay on there: _decoder_replay_supported refuses only
# `self.start_layer > cut` (nvidia/model.py:1332). forward() then runs the layers before
# the replay layers, which on such a stage is the empty range [start_layer, cut)
# (nvidia/model.py:1120), and _run_layers ends in `assert residual is not None and
# post_mix is not None` (nvidia/model.py:1247), which nothing in an empty range has set.
# The first forward of that rank is the memory profile, so the boot dies there, and all
# the engine prints is "Worker failed with error ''": an AssertionError has no message.
# This pair splits 20/20, so rank 1 is such a stage if the checkpoint's last KV source
# layer is 20 (max of kv_source_layer_ids in its config.json). The guard answers "not
# supported" for exactly that case and nothing else: every other stage gets upstream's
# own answer. VLLM_PATCH_PP_REPLAY_BOUNDARY=0 removes it.
_MODEL_MODULE = "vllm.models.deepseek_v41.nvidia.model"
_BOUNDARY_ENV = "VLLM_PATCH_PP_REPLAY_BOUNDARY"
_boundary_logged = False


def _guard_replay_boundary(module):
    if os.environ.get(_BOUNDARY_ENV, "1") == "0":
        return
    cls = getattr(module, "DeepseekV4Model", None)
    original = getattr(cls, "_decoder_replay_supported", None)
    if original is None or getattr(original, "_pp2_boundary_guard", False):
        return  # v0.31.0 has no decoder-side replay; or already guarded

    def _decoder_replay_supported(self, vllm_config, cut, *args, **kwargs):
        global _boundary_logged
        supported = original(self, vllm_config, cut, *args, **kwargs)
        if supported and getattr(self, "start_layer", None) == cut:
            if not _boundary_logged:
                _boundary_logged = True
                sys.stderr.write(
                    f"[pp2_flag] layers {self.start_layer}-{self.end_layer - 1}: decoder SWA bounded "
                    f"replay is off for this stage. It begins at the last KV source layer ({cut}), "
                    "where vLLM v0.34.0 would fail its first forward with an AssertionError that has "
                    f"no message. {_BOUNDARY_ENV}=0 removes this guard.\n"
                )
                sys.stderr.flush()
            return False
        return supported

    _decoder_replay_supported._pp2_boundary_guard = True
    _decoder_replay_supported._pp2_upstream = original  # verify/step0_import_smoke.py asks it
    cls._decoder_replay_supported = _decoder_replay_supported


class _FlagOnImport:
    """Meta-path finder: let the normal machinery find the module, then set the flag
    right after the module body has run."""

    def __init__(self):
        self.pending = set(_TARGETS)

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.pending:
            return None
        self.pending.discard(fullname)  # also stops the recursion two lines down
        spec = importlib.util.find_spec(fullname)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            # Wrong image or missing module: nothing to gate, let the boot speak.
            return None
        run = spec.loader.exec_module

        def exec_module(module):
            run(module)
            _flip(module)

        spec.loader.exec_module = exec_module
        return spec


for _module_name in _TARGETS:
    if _module_name in sys.modules:  # not at interpreter start; harmless if it ever is
        _flip(sys.modules[_module_name])
sys.meta_path.insert(0, _FlagOnImport())

# v0.34.0: no autotune hook. Upstream flashinfer_autotune() tunes per PP stage
# over the TP group with a per-rank cache file (kernel_warmup.py:569-572, 600),
# and the v0.31.0 pp_autotune.py calls kernel_warmup names that are gone here.
# What upstream kept is the world barrier after the tune (kernel_warmup.py:630-631);
# the recipe passes --cpu-distributed-timeout-seconds for that. Do not re-add the hook.
