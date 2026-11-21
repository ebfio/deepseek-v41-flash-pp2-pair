"""Rebuild patches/vllm-pp2-flag-v034/dspark.py from a pristine vLLM dspark.py.

Usage: python3 -I rebase_dspark.py <pristine_dspark.py> <out_overlay.py>

Use it when the image's vllm/models/deepseek_v41/nvidia/dspark.py is not the
v0.34.0 file in PRISTINE.sha256 (a new tag, a rebuilt image). Each of the four
edits is anchored on text that must occur exactly once; any drift aborts
instead of guessing. After a successful run, regenerate dspark.py.diff with
`diff -u` against the pristine file, update PRISTINE.sha256, and re-run
verify/step0_overlays.py.
"""
import sys

if len(sys.argv) != 3 or sys.argv[1].startswith("-"):
    raise SystemExit(__doc__)
src_path, out_path = sys.argv[1], sys.argv[2]
src = open(src_path, encoding="utf-8").read()


def replace_once(text: str, anchor: str, replacement: str) -> str:
    n = text.count(anchor)
    if n != 1:
        raise SystemExit(f"anchor occurs {n} times, expected 1:\n{anchor!r}")
    return text.replace(anchor, replacement)


# 1. import: 0.34.0 no longer imports from vllm.distributed in this module.
src = replace_once(
    src,
    "from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS\n",
    "from vllm.config.kernel import NATIVE_MEGA_MOE_BACKENDS\n"
    "from vllm.distributed.parallel_state import get_pp_group\n",
)

# 2. DSparkDeepseekV4ForCausalLM.__init__: own the embedding under PP>1.
src = replace_once(
    src,
    "        self.draft_model_config = vllm_config.speculative_config.draft_model_config\n"
    "        self.config = self.draft_model_config.hf_config\n"
    "        self.quant_config = vllm_config.quant_config\n",
    "        self.draft_model_config = vllm_config.speculative_config.draft_model_config\n"
    "        self.config = self.draft_model_config.hf_config\n"
    "        # PP>1: the target's embed_tokens is a PPMissingLayer on the\n"
    "        # last stage, so the draft must load its own copy (see\n"
    "        # load_weights). Gates maybe_share_target_embed's share path.\n"
    "        self.has_own_embed_tokens = get_pp_group().world_size > 1\n"
    "        self.quant_config = vllm_config.quant_config\n",
)

# 3. load_weights: catch the checkpoint's embed.weight before the mtp.* remap.
src = replace_once(
    src,
    "        for name, loaded_weight in weights:\n"
    "            mapped = self._remap_dspark_name(name)\n",
    "        for name, loaded_weight in weights:\n"
    "            if name == \"embed.weight\" and self.has_own_embed_tokens:\n"
    "                # PP>1: load the draft's own embedding from the\n"
    "                # checkpoint; the target's lives on the first stage.\n"
    "                param = params_dict[\"model.embed_tokens.weight\"]\n"
    "                param.weight_loader(param, loaded_weight)\n"
    "                loaded_params.add(\"model.embed_tokens.weight\")\n"
    "                continue\n"
    "            mapped = self._remap_dspark_name(name)\n",
)

# 4. post-load assert: a renamed checkpoint key must fail loudly.
src = replace_once(
    src,
    "        logger.info_once(\"DSpark draft model loaded: %d params\", len(loaded_params))\n"
    "        return loaded_params\n",
    "        logger.info_once(\"DSpark draft model loaded: %d params\", len(loaded_params))\n"
    "        if self.has_own_embed_tokens:\n"
    "            # A renamed checkpoint key must fail loudly here, not produce a\n"
    "            # draft with an unloaded (zero/garbage) embedding table.\n"
    "            assert \"model.embed_tokens.weight\" in loaded_params, (\n"
    "                \"PP>1: the draft must load its own embed.weight, but the \"\n"
    "                \"checkpoint never yielded it\"\n"
    "            )\n"
    "        return loaded_params\n",
)

open(out_path, "w", encoding="utf-8").write(src)
print("ok")
