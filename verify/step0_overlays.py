#!/usr/bin/env python3
"""Step 0 receipt: is this vLLM tree the one the v0.34.0 PP2 overlay set was cut for?

Static only. It reads source files; it never imports vllm or touches a GPU.
Run it inside the image you are about to serve from, on every node:

    docker run --rm --entrypoint python3 \
        -v "$PWD/patches/vllm-pp2-flag-v034:/overlay:ro" \
        -v "$PWD/verify:/verify:ro" \
        vllm/vllm-openai:v0.34.0 \
        /verify/step0_overlays.py --overlay-dir /overlay --json -

Every line starts with how the check went: ok, WARN or FAIL. Exit code 0 means
no FAIL. A WARN is something a boot must still confirm; on v0.34.0 exactly one
is expected, the cold-tune world barrier. A FAIL means do not mount this overlay
set on this image.

What each check stands for is in patches/vllm-pp2-flag-v034/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sys
from pathlib import Path

REQUIRED, WARN, INFO = "REQUIRED", "WARN", "INFO"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vllm is not installed here; pass --vllm-root")
    return Path(list(spec.submodule_search_locations)[0])


def apply_unified_diff_strict(original: str, diff_text: str) -> str:
    """Apply a single-file unified diff with zero offset and zero fuzz."""
    src = original.splitlines(keepends=True)
    out: list[str] = []
    cursor = 0  # index into src of the next unconsumed line
    hunk_re = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
    lines = diff_text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and not lines[i].startswith("@@"):
        i += 1
    if i == len(lines):
        raise ValueError("diff has no hunks")
    while i < len(lines):
        m = hunk_re.match(lines[i])
        if not m:
            raise ValueError(f"expected hunk header, got: {lines[i]!r}")
        old_start = int(m.group(1))
        i += 1
        target = old_start - 1
        if target < cursor:
            raise ValueError("hunks overlap or are out of order")
        out.extend(src[cursor:target])
        cursor = target
        while i < len(lines) and not lines[i].startswith("@@"):
            tag, body = lines[i][:1], lines[i][1:]
            if tag == " " or tag == "-":
                if cursor >= len(src) or src[cursor] != body:
                    got = src[cursor] if cursor < len(src) else "<EOF>"
                    raise ValueError(
                        f"context mismatch at source line {cursor + 1}: "
                        f"diff has {body!r}, file has {got!r}"
                    )
                if tag == " ":
                    out.append(body)
                cursor += 1
            elif tag == "+":
                out.append(body)
            elif tag == "\\":
                pass  # "\ No newline at end of file"
            else:
                raise ValueError(f"unexpected diff line: {lines[i]!r}")
            i += 1
    out.extend(src[cursor:])
    return "".join(out)


def class_body(text: str, class_name: str) -> str | None:
    """Source of one top-level class, from its header to the next top-level def/class."""
    m = re.search(rf"^class {re.escape(class_name)}\b.*$", text, re.M)
    if not m:
        return None
    rest = text[m.end():]
    nxt = re.search(r"^(class |def |@)", rest, re.M)
    return text[m.start(): m.end() + (nxt.start() if nxt else len(rest))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vllm-root", type=Path, default=None,
                    help="directory of the installed vllm package (default: locate it)")
    ap.add_argument("--overlay-dir", type=Path, required=True,
                    help="patches/vllm-pp2-flag-v034")
    ap.add_argument("--json", default=None, help="write the receipt as JSON here ('-' = stdout)")
    args = ap.parse_args()

    root = args.vllm_root or find_vllm_root()
    overlay = args.overlay_dir
    results: list[dict] = []

    def record(name: str, level: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "level": level, "ok": bool(ok), "detail": detail})

    def read(rel: str) -> str | None:
        p = root / rel
        return p.read_text(encoding="utf-8") if p.is_file() else None

    try:
        version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown (no dist metadata; a source checkout)"
    record("vllm version", INFO, True, version)

    # 1. Pristine hashes. Only dspark.py is REQUIRED to match: it is the one file
    #    this set replaces whole. The rest are the files the evidence was read from.
    expected: dict[str, str] = {}
    for line in (overlay / "PRISTINE.sha256").read_text().splitlines():
        if line.strip():
            digest, rel = line.split(None, 1)
            expected[rel.strip()] = digest
    for rel, digest in expected.items():
        p = root.parent / rel
        whole_file = rel.endswith("deepseek_v41/nvidia/dspark.py")
        if not p.is_file():
            record(f"pristine {rel}", REQUIRED if whole_file else WARN, False, "file missing")
            continue
        got = sha256(p)
        record(
            f"pristine {rel}",
            REQUIRED if whole_file else WARN,
            got == digest,
            "matches v0.34.0" if got == digest else f"differs from v0.34.0 (got {got[:16]}…)",
        )

    # 2. The .diff applied strictly to this image's dspark.py must reproduce the
    #    whole-file overlay byte for byte.
    dspark_rel = "models/deepseek_v41/nvidia/dspark.py"
    pristine = read(dspark_rel)
    try:
        if pristine is None:
            raise ValueError("dspark.py missing in this tree")
        rebuilt = apply_unified_diff_strict(
            pristine, (overlay / "dspark.py.diff").read_text(encoding="utf-8")
        )
        same = rebuilt == (overlay / "dspark.py").read_text(encoding="utf-8")
        record("dspark.py.diff -> dspark.py round-trip", REQUIRED, same,
               "byte-identical" if same else "diff applies but result != shipped overlay")
    except (ValueError, OSError) as exc:
        record("dspark.py.diff -> dspark.py round-trip", REQUIRED, False, str(exc))

    # 3. Fix 3 (raw input tokens on non-first PP ranks) is upstream: runner + capture.
    runner = read("v1/worker/gpu/model_runner.py") or ""
    ok = bool(re.search(
        r"if not self\.is_first_pp_rank:\n(?:\s*#.*\n)*\s*if not requires_raw_input_tokens\(self\.model\):\n"
        r"\s*model_inputs\[\"input_ids\"\] = None", runner))
    record("upstream: runner keeps input_ids on non-first PP ranks", REQUIRED, ok,
           "present; do not mount model_runner.py" if ok else "NOT found; the v0.31.0 fix is not in this tree")
    capture = read("v1/worker/gpu/cudagraph_utils.py") or ""
    ok = bool(re.search(
        r"if not requires_raw_input_tokens\(model\):\n\s*model_inputs\[\"input_ids\"\] = None", capture))
    record("upstream: graph capture keeps input_ids on non-first PP ranks", REQUIRED, ok,
           "present; do not mount cudagraph_utils.py" if ok else "NOT found")

    # 4. Fix 4 (KV tensors only for a group's own layers) is upstream, in a new file.
    planning = read("v1/core/kv_cache_planning.py")
    if planning is None:
        record("upstream: KV tensors emitted per owned layer", REQUIRED, False,
               "kv_cache_planning.py missing (pre-0.33 layout)")
    else:
        new_loop = bool(re.search(
            r"for layer_name in group\.layer_names:\n\s*layers_by_spec\[_get_per_layer_spec\(group, layer_name\)\]"
            r"\.append\(layer_name\)", planning))
        old_loop = "for layer_name, spec in group_spec.kv_cache_specs.items():" in planning
        record("upstream: KV tensors emitted per owned layer", REQUIRED, new_loop and not old_loop,
               "present; do not mount kv_cache_utils.py" if new_loop and not old_loop
               else "old unfiltered loop still here (this is 0.33.x behaviour)")

    # 5. Fix 1 (the ClassVar flip) is still needed, and its gate and targets exist.
    model = read("models/deepseek_v41/nvidia/model.py") or ""
    body = class_body(model, "DeepseekV4Model")
    if body is None:
        record("sitecustomize target DeepseekV4Model", REQUIRED, False, "class not found")
    else:
        has_flag = "supports_aux_hidden_states_over_pp" in body
        record("sitecustomize target DeepseekV4Model", REQUIRED, True, "class found")
        record("flag flip still needed", INFO, True,
               "upstream now sets it; sitecustomize.py can be dropped" if has_flag
               else "DSV4.1 decoder lacks supports_aux_hidden_states_over_pp; keep sitecustomize.py")
    eagle3 = read("v1/worker/gpu/spec_decode/eagle/eagle3_utils.py") or ""
    record("PP gate present", REQUIRED,
           "def verify_supports_aux_hidden_states_over_pp" in eagle3
           and "supports_aux_hidden_states_over_pp" in eagle3,
           "eagle3_utils.verify_supports_aux_hidden_states_over_pp")
    ds = pristine or ""
    record("draft classes present", REQUIRED,
           "class DSparkDeepseekV4Model(" in ds and "class DSparkDeepseekV4ForCausalLM(" in ds,
           "DSparkDeepseekV4Model, DSparkDeepseekV4ForCausalLM")

    # 6. Fix 2 preconditions: the share helper still honours has_own_embed_tokens.
    eutils = read("v1/worker/gpu/spec_decode/eagle/utils.py") or ""
    ok = ("def maybe_share_target_embed" in eutils
          and "has_own_embed_tokens" in eutils
          and "needs the target input embedding" in eutils)
    record("embed-share helper honours has_own_embed_tokens", REQUIRED, ok,
           "maybe_share_target_embed unchanged in the path the overlay relies on" if ok
           else "helper changed; re-read before trusting the dspark overlay")
    vl = read("models/deepseek_v41/nvidia/vl_model.py") or ""
    record("model still declares requires_raw_input_tokens", REQUIRED,
           "requires_raw_input_tokens = True" in vl, "vl_model.py")

    # 7. Autotune: upstream is PP-aware, the old overlay is dead, the barrier remains.
    warm = read("model_executor/warmup/kernel_warmup.py") or ""
    record("upstream: autotune tunes per PP stage with per-rank files", REQUIRED,
           "tune_group = get_tp_group() if pp_size > 1 else world" in warm
           and "resolve_rank_flashinfer_autotune_file" in warm,
           "kernel_warmup.flashinfer_autotune")
    dead = ("_run_flashinfer_bf16_autotune_dummy_run" not in warm
            and not re.search(r"(?<!rank_)\bresolve_flashinfer_autotune_file\b", warm))
    record("v0.31.0 pp_autotune.py overlay must stay off", INFO, True,
           "it calls kernel_warmup names that are gone here" if dead
           else "its kernel_warmup names still exist here")
    barrier = bool(re.search(r"if world\.world_size > 1:\n\s*world\.barrier\(\)", warm))
    record("cold-tune world barrier", WARN, not barrier,
           "still present: the stage that tunes faster waits here for the other "
           "(gloo default 1800 s); the recipe passes --cpu-distributed-timeout-seconds" if barrier
           else "no world barrier after the tune")

    # 8. Decoder replay on a stage that begins at the last KV source layer. Upstream turns
    #    replay on there and then fails its first forward; sitecustomize.py guards it.
    if "def _decoder_replay_supported" in model:
        pieces = {
            "the support check lets start_layer == cut through":
                "if self.start_layer > cut or self.end_layer < self.config.num_hidden_layers:" in model,
            "forward runs the layers before the replay layers, an empty range on such a stage":
                "range(self.start_layer, self.decoder_replay_start)," in model,
            "_run_layers ends in an assert with no message":
                bool(re.search(r"\n        assert residual is not None and post_mix is not None\n", model)),
        }
        present = [name for name, found in pieces.items() if found]
        if len(present) == len(pieces):
            record("upstream: replay on a stage that begins at the last KV source", INFO, True,
                   "still fails its first forward there (support check, empty range, bare assert): "
                   "keep the guard in sitecustomize.py")
        else:
            record("upstream: replay on a stage that begins at the last KV source", WARN, False,
                   f"{len(present)} of {len(pieces)} pieces of the defect found: upstream changed this "
                   "path; re-read it before keeping or dropping the guard")
        site = (args.overlay_dir / "sitecustomize.py").read_text(encoding="utf-8")
        guarded = ("_guard_replay_boundary" in site and "VLLM_PATCH_PP_REPLAY_BOUNDARY" in site
                   and bool(re.search(r"def _decoder_replay_supported\(\s*self, vllm_config: VllmConfig, cut: int\s*\)",
                                      model)))
        record("sitecustomize guards that stage", REQUIRED, guarded,
               "guard present and DeepseekV4Model._decoder_replay_supported(self, vllm_config, cut) is "
               "the method it wraps" if guarded else "guard missing from this overlay set, or the method's "
               "signature moved")

    width = max(len(r["check"]) for r in results)
    # One word per line says how the check went: ok, WARN (a non-required check did not
    # pass) or FAIL. On v0.34.0 one WARN is expected: the cold-tune world barrier.
    for r in results:
        status = "ok" if r["ok"] else ("FAIL" if r["level"] == REQUIRED else "WARN")
        print(f"{status:<4}  {r['check']:<{width}}  {r['detail']}")
    failed = [r for r in results if r["level"] == REQUIRED and not r["ok"]]
    warned = [r for r in results if r["level"] != REQUIRED and not r["ok"]]
    verdict = "PASS" if not failed else "FAIL"
    print(f"\nSTEP0 {verdict}: {len(failed)} required check(s) failed, {len(warned)} warning(s); vllm root = {root}")

    if args.json:
        payload = json.dumps({"verdict": verdict, "vllm_root": str(root), "results": results}, indent=2)
        if args.json == "-":
            print(payload)
        else:
            Path(args.json).write_text(payload + "\n", encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
