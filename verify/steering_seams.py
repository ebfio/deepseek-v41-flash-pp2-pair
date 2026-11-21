#!/usr/bin/env python3
"""Steering receipt 1 of 2: does this vLLM tree still have the seams pp2_steering patches?

Static only. It parses source files; it never imports vllm or touches a GPU.
Run it inside the image you are about to serve from, on every node:

    docker run --rm --entrypoint python3 \
        -v "$PWD/patches/pp2-steering:/steering:ro" \
        -v "$PWD/verify:/verify:ro" \
        vllm/vllm-openai:v0.34.0 \
        /verify/steering_seams.py --steering-dir /steering --json -

Every line starts with how the check went: ok, WARN or FAIL. Exit code 0 means no FAIL:
every required seam is where patches/pp2-steering/pp2_steering.py expects it. A WARN on a
"pristine" line means that file is not byte-identical to the v0.34.0 tag the seams were
read on: the seam may still be fine, but somebody has to re-read that file. On the
v0.34.0 image expect "0 required check(s) failed, 0 warning(s)". A FAIL means do not boot
with VLLM_STEERING_SIDECAR set on this image.

This says nothing about behaviour. verify/steering_probe.py is the functional receipt.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REQUIRED, WARN, INFO = "REQUIRED", "WARN", "INFO"

MODEL_STATES = "v1/worker/gpu/model_states/__init__.py"
MODEL_STATE = "models/deepseek_v41/nvidia/model_state.py"
STATE_INTERFACE = "v1/worker/gpu/model_states/interface.py"
STATE_DEFAULT = "v1/worker/gpu/model_states/default.py"
MODEL = "models/deepseek_v41/nvidia/model.py"
MOE = "models/deepseek_v4/nvidia/model.py"
SHARED = "model_executor/layers/fused_moe/runner/shared_experts.py"
RUNNER = "v1/worker/gpu/model_runner.py"
GRAPHS = "v1/worker/gpu/cudagraph_utils.py"
BATCH = "v1/worker/gpu/input_batch.py"
REQUEST = "v1/request.py"
KV_UTILS = "v1/core/kv_cache_utils.py"
CHAT = "entrypoints/openai/chat_completion/protocol.py"
COMPLETION = "entrypoints/openai/completion/protocol.py"
PARALLEL = "config/parallel.py"
EXCEPTIONS = "exceptions.py"
REPLAY_LAYERS = "models/deepseek_v41/decoder_replay_layers.py"
FORWARD_CONTEXT = "forward_context.py"


class Tree:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._text: dict[str, str | None] = {}
        self._ast: dict[str, ast.Module | None] = {}

    def text(self, rel: str) -> str | None:
        if rel not in self._text:
            path = self.root / rel
            self._text[rel] = path.read_text(encoding="utf-8") if path.is_file() else None
        return self._text[rel]

    def tree(self, rel: str) -> ast.Module | None:
        if rel not in self._ast:
            text = self.text(rel)
            self._ast[rel] = ast.parse(text) if text is not None else None
        return self._ast[rel]

    def function(self, rel: str, name: str, cls: str | None = None) -> ast.FunctionDef | None:
        module = self.tree(rel)
        if module is None:
            return None
        body: list[ast.stmt] = module.body
        if cls is not None:
            owner = next(
                (n for n in module.body if isinstance(n, ast.ClassDef) and n.name == cls), None
            )
            if owner is None:
                return None
            body = owner.body
        return next(
            (
                n
                for n in body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
            ),
            None,
        )

    def class_fields(self, rel: str, cls: str) -> set[str]:
        module = self.tree(rel)
        owner = next(
            (n for n in (module.body if module else []) if isinstance(n, ast.ClassDef) and n.name == cls),
            None,
        )
        if owner is None:
            return set()
        return {
            n.target.id
            for n in owner.body
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
        }


def positional(function: ast.FunctionDef | None) -> list[str]:
    if function is None:
        return []
    return [a.arg for a in function.args.posonlyargs + function.args.args]


def calls(function: ast.FunctionDef | None, name: str) -> bool:
    if function is None:
        return False
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            target = node.func
            called = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
            if called == name:
                return True
    return False


def is_classmethod(function: ast.FunctionDef | None) -> bool:
    return function is not None and any(
        isinstance(d, ast.Name) and d.id == "classmethod" for d in function.decorator_list
    )


def run(root: Path, steering_dir: Path) -> list[dict]:
    tree = Tree(root)
    results: list[dict] = []

    def record(name: str, level: str, ok: bool, detail: str) -> None:
        results.append({"check": name, "level": level, "ok": bool(ok), "detail": detail})

    def has(rel: str, needle: str) -> bool:
        text = tree.text(rel)
        return text is not None and needle in text

    # 1. worker: where the model-state class is resolved
    resolve = tree.function(MODEL_STATES, "resolve_model_state_cls")
    init = tree.function(MODEL_STATES, "init_model_state")
    record(
        "model-state resolver",
        REQUIRED,
        positional(resolve) == ["vllm_config", "model"] and calls(init, "resolve_model_state_cls"),
        f"resolve_model_state_cls({', '.join(positional(resolve))}); "
        f"init_model_state calls it: {calls(init, 'resolve_model_state_cls')}",
    )

    # 2. worker: the class that is subclassed, and the three methods that are overridden
    expected = {
        "__init__": ["self", "vllm_config", "model", "encoder_cache", "device"],
        "add_request": ["self", "req_index", "new_req_data"],
        "prepare_inputs": ["self", "input_batch", "req_states"],
        "prepare_dummy_inputs": ["self", "num_reqs", "num_tokens"],
    }
    for method, want in expected.items():
        found = None
        for rel, cls in (
            (MODEL_STATE, "DeepseekV41ModelState"),
            (STATE_DEFAULT, "DefaultModelState"),
            (STATE_INTERFACE, "ModelState"),
        ):
            function = tree.function(rel, method, cls)
            if function is not None:
                found = (rel, positional(function))
                break
        record(
            f"DeepseekV41ModelState.{method}",
            REQUIRED,
            found is not None and found[1] == want,
            "not found" if found is None else f"({', '.join(found[1])}) in {found[0]}",
        )
    record(
        "runner feeds the model state",
        REQUIRED,
        has(RUNNER, "self.model_state.add_request(req_index, new_req_data)")
        and has(RUNNER, "self.model_state.prepare_inputs(input_batch, self.req_states)"),
        "add_request(req_index, new_req_data) and prepare_inputs(input_batch, self.req_states) "
        "are called from model_runner.py",
    )
    record(
        "graph capture zeroes through the model state",
        REQUIRED,
        has(GRAPHS, "model_state.prepare_dummy_inputs(num_reqs, num_tokens)"),
        "cudagraph_utils.py calls model_state.prepare_dummy_inputs(num_reqs, num_tokens)",
    )
    fields = tree.class_fields(BATCH, "InputBatch")
    needed = {"num_reqs", "idx_mapping", "query_start_loc", "num_tokens_after_padding"}
    record(
        "InputBatch fields",
        REQUIRED,
        needed <= fields,
        "all present" if needed <= fields else f"missing {sorted(needed - fields)}",
    )

    # 3. model: the two module calls the hooks sit on
    record(
        "attention output goes through Module.__call__",
        REQUIRED,
        has(MODEL, "x = self.attn(positions, x, None)"),
        "DeepseekV4DecoderLayer.forward: x = self.attn(positions, x, None)",
    )
    shared_text = tree.text(SHARED) or ""
    through_call = shared_text.count("self._layer(shared_experts_input)")
    record(
        "shared expert goes through Module.__call__",
        REQUIRED,
        through_call >= 1 and "self._layer.forward(" not in shared_text,
        f"{through_call} call(s) of self._layer(shared_experts_input), "
        f"{shared_text.count('self._layer.forward(')} of self._layer.forward(",
    )
    record(
        "shared expert is a module unless MegaMoE fuses it",
        REQUIRED,
        has(MOE, "self.shared_experts = DeepseekV4MLP(") and has(MOE, "enable_expert_parallel"),
        "ffn.shared_experts is a DeepseekV4MLP; fusion needs a MegaMoE backend, which needs "
        "--enable-expert-parallel (pp2_steering refuses a fused one at boot)",
    )
    record(
        "stage layer range",
        REQUIRED,
        has(MODEL, "self.start_layer") and has(MODEL, "self.end_layer"),
        "DeepseekV4Model has start_layer / end_layer",
    )
    replay = tree.function(MODEL, "_decoder_replay_supported", "DeepseekV4Model")
    if replay is None:
        record(
            "decoder bounded replay",
            INFO,
            not has(MODEL, "decoder_replay"),
            "no _decoder_replay_supported on this tree (pre-0.33): nothing to switch off"
            if not has(MODEL, "decoder_replay")
            else "decoder replay exists but its gate was renamed: re-read model.py",
        )
        if has(MODEL, "decoder_replay"):
            record("decoder replay gate", REQUIRED, False, "_decoder_replay_supported not found")
    else:
        record(
            "decoder replay gate",
            REQUIRED,
            has(MODEL, "self._decoder_replay_supported(") and has(MODEL_STATE, '"decoder_replay_layers"'),
            "DeepseekV4Model._decoder_replay_supported gates decoder_replay_layers, which the "
            "model state reads with getattr(model, \"decoder_replay_layers\", None)",
        )

    # 3b. only for VLLM_STEERING_REPLAY=keep: the names the replay batch is followed
    # through. A WARN here means: leave that knob at off on this image (the module
    # itself refuses "keep" at boot when one of them is gone).
    if replay is not None:
        prepare = tree.function(MODEL_STATE, "_prepare_replay_batch", "DeepseekV41ModelState")
        runner_text = tree.text(RUNNER) or ""
        attn_at = runner_text.find("self.model_state.prepare_attn(")
        inputs_at = runner_text.find("self.model_state.prepare_inputs(input_batch, self.req_states)")
        keep = {
            "_prepare_replay_batch(self, input_batch, ...)": positional(prepare)[:2] == ["self", "input_batch"],
            "prepare_attn calls it": has(MODEL_STATE, "self._prepare_replay_batch("),
            "the runner calls prepare_attn before prepare_inputs": 0 <= attn_at < inputs_at,
            "the sub-batch's row ends are self._kept_query_start_loc": has(
                MODEL_STATE, "query_start_loc=self._kept_query_start_loc[: num_reqs + 1]"
            ),
            "its forward context names its padded size": has(
                MODEL_STATE, "batch_descriptor=BatchDescriptor(num_tokens=num_padded)"
            ),
            "the replay layers run under that context": has(
                REPLAY_LAYERS, "with override_forward_context(replay_batch.forward_context):"
            ),
            "layers.replay_batch is reset every step": has(MODEL_STATE, "layers.replay_batch = None"),
            "decoder_replay_start bounds the replay layers": has(
                MODEL, "range(self.decoder_replay_start, self.end_layer)"
            ),
            "get_forward_context": has(FORWARD_CONTEXT, "def get_forward_context("),
            "replay graphs exist on the first rank only": has(
                MODEL_STATE, "and get_pp_group().is_first_rank"
            ),
        }
        gone = [name for name, ok in keep.items() if not ok]
        record(
            "replay batch can be followed (VLLM_STEERING_REPLAY=keep)",
            WARN,
            not gone,
            f"{len(keep)} names and call orders as read on v0.34.0"
            if not gone
            else "changed: " + "; ".join(gone) + ". Keep VLLM_STEERING_REPLAY=off on this image",
        )

    # 4. engine core: where the cache salt goes in
    request = tree.function(REQUEST, "from_engine_core_request", "Request")
    args = positional(request)
    record(
        "Request.from_engine_core_request",
        REQUIRED,
        is_classmethod(request) and args[:3] == ["cls", "request", "block_hasher"],
        f"classmethod: {is_classmethod(request)}; ({', '.join(args)})",
    )
    record(
        "cache_salt reaches the block hash",
        REQUIRED,
        has(REQUEST, "cache_salt") and has(KV_UTILS, "request.cache_salt"),
        "Request carries cache_salt and kv_cache_utils.py hashes request.cache_salt",
    )
    core_text = tree.text("v1/engine/core.py") or ""
    hashers = [
        rel
        for rel in ("v1/engine/core.py", "v1/worker/gpu_worker.py", RUNNER, "v1/executor/multiproc_executor.py")
        if "get_request_block_hasher(" in (tree.text(rel) or "")
    ]
    record(
        "one block hasher, in the engine core",
        REQUIRED,
        hashers == ["v1/engine/core.py"] and "Request.from_engine_core_request(" in core_text,
        "get_request_block_hasher is built in v1/engine/core.py only, and that is where "
        "Request.from_engine_core_request is called: no worker or pipeline stage hashes"
        if hashers == ["v1/engine/core.py"]
        else f"block hashers are built in {hashers}: re-read where the salt has to go",
    )

    # 5. API server: optional seams (validation and the top-level spelling)
    for rel, cls in ((CHAT, "ChatCompletionRequest"), (COMPLETION, "CompletionRequest")):
        function = tree.function(rel, "to_sampling_params", cls)
        record(
            f"{cls}.to_sampling_params",
            WARN,
            function is not None and "vllm_xargs" in tree.class_fields(rel, cls),
            "present with a vllm_xargs field"
            if function is not None
            else "not found: steering still works through vllm_xargs, without the HTTP 400",
        )
    record(
        "VLLMValidationError",
        INFO,
        has(EXCEPTIONS, "class VLLMValidationError"),
        "present" if has(EXCEPTIONS, "class VLLMValidationError") else "absent: ValueError is used",
    )
    record(
        "microbatching flag",
        INFO,
        has(PARALLEL, "def use_ubatching"),
        "ParallelConfig.use_ubatching exists (pp2_steering refuses to start when it is true)",
    )

    # 6. are these the files the seams were read on?
    pins = steering_dir / "SEAMS-v0.34.0.sha256"
    if pins.is_file():
        for line in pins.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            want, rel = line.split()
            rel = rel.removeprefix("vllm/")
            path = root / rel
            got = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            record(
                f"pristine vllm/{rel}",
                WARN,
                got == want,
                "matches v0.34.0" if got == want else f"differs from v0.34.0 (got {str(got)[:16]}…)",
            )
    else:
        record("pristine file list", WARN, False, f"{pins} not found")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--vllm-root", type=Path, help="installed vllm package dir (default: locate it)")
    parser.add_argument("--steering-dir", type=Path, required=True, help="patches/pp2-steering")
    parser.add_argument("--json", help="also write the results as JSON to this file ('-' = stdout)")
    args = parser.parse_args()
    root = args.vllm_root
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            print("vllm is not installed here; pass --vllm-root", file=sys.stderr)
            return 2
        root = Path(list(spec.submodule_search_locations)[0])
    results = run(root, args.steering_dir)
    # One word per line says how the check went: ok, WARN (a non-required check did not
    # pass) or FAIL (a required one did not). Revision 2 printed each check's severity
    # in a second column, so every passing "pristine" line carried the word WARN.
    for item in results:
        mark = "ok" if item["ok"] else ("FAIL" if item["level"] == REQUIRED else "WARN")
        print(f"{mark:<4}  {item['check']:<50}  {item['detail']}")
    failed = [r for r in results if not r["ok"] and r["level"] == REQUIRED]
    warned = [r for r in results if not r["ok"] and r["level"] != REQUIRED]
    verdict = "PASS" if not failed else "FAIL"
    print(f"STEERING-SEAMS {verdict}: {len(failed)} required check(s) failed, {len(warned)} warning(s); "
          f"vllm root = {root}")
    if args.json:
        payload = json.dumps({"verdict": verdict, "warnings": len(warned), "vllm_root": str(root),
                              "results": results}, indent=2)
        if args.json == "-":
            print(payload)
        else:
            Path(args.json).write_text(payload + "\n")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
