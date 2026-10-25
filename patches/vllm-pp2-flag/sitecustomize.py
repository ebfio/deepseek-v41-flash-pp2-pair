# patches/vllm-pp2-flag/ — the one-line PP2+DSpark overlay.
# The gate in vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py reads
# supports_aux_hidden_states_over_pp on the INNER decoder
# (model.get_language_model() -> .model), not the ForCausalLM wrapper, so
# the attr is set on DeepseekV4Model / DSparkDeepseekV4Model here.
# Loaded via PYTHONPATH sitecustomize at engine start.

import importlib

_TARGETS = [
    ("vllm.models.deepseek_v41.nvidia.model", "DeepseekV4Model"),
    ("vllm.models.deepseek_v41.nvidia.dspark", "DSparkDeepseekV4Model"),
    # Belt and braces: the wrappers, for any gate that reads them directly.
    ("vllm.models.deepseek_v41.nvidia.model", "DeepseekV41ForCausalLM"),
    ("vllm.models.deepseek_v41.nvidia.dspark", "DSparkDeepseekV4ForCausalLM"),
]

for _module_name, _class_name in _TARGETS:
    try:
        _module = importlib.import_module(_module_name)
        _cls = getattr(_module, _class_name, None)
        if _cls is not None:
            setattr(_cls, "supports_aux_hidden_states_over_pp", True)
    except ImportError:
        # Wrong image or missing module: nothing to gate, let the boot speak.
        pass
