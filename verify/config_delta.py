#!/usr/bin/env python3
"""What differs between the config.env a box really runs and the one this repo ships,
and would the patched launcher start from it?

Static and read-only: it parses the files, it does not source them.

    python3 verify/config_delta.py --shipped /path/on/box/config.env \
        [--repo recipe/config.env] [--launcher recipe/serve_vllm_node.sh] \
        [--lane v0.31.0|v0.34.0] [--steering] [--redact]

Section 1 lists every key whose default differs, and keys only one side has.
Section 2 lists what the launcher reads that the shipped file never defines: the launcher
runs under `set -u`, so each of those is an "unbound variable" abort at launch.
Section 3 checks the shipped values against what the launcher refuses for one lane:
--lane names it (default: the file's own VLLM_OVERLAY_TAG, or v0.31.0 when it has none, which
is what the launcher assumes). The steering checks count only when the file configures a
sidecar or --steering is given. So for "would v0.34.0 with steering start from the file as
shipped today": --lane v0.34.0 --steering.
--redact prints IPs, hosts and paths as <set>, so the output can be pasted outside.
Exit status 1 when section 2 or 3 found something that stops that launch.
"""

from __future__ import annotations

import argparse
import re
import sys

DEFAULT = re.compile(r'^\s*:\s*"\$\{([A-Za-z_][A-Za-z0-9_]*):=(.*)$')
ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
USED = re.compile(r"\$\{?([A-Z][A-Z0-9_]{2,})(:?[-=?+])?")
SENSITIVE = re.compile(r"IP|SSH|DIR|ROOT|MOUNT|EXPORT|HOST|PORT|IFACE|HCAS")


def parse(path: str) -> tuple[dict[str, str], set[str]]:
    """({key: default or assigned value}, keys that are hard-assigned)."""
    values: dict[str, str] = {}
    hard: set[str] = set()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = DEFAULT.match(line)
            if match:
                key, rest = match.groups()
                depth, value = 1, []
                for char in rest:  # the default ends at the brace that closes ${KEY:=
                    depth += char == "{"
                    depth -= char == "}"
                    if depth == 0:
                        break
                    value.append(char)
                # Later lines win, as they would when sourced with the key unset
                # (a conditional that defines a key twice shows its last branch).
                values[key] = "".join(value).strip()
                continue
            match = ASSIGN.match(line)
            if match and not line.lstrip().startswith(("_t=", "export ")):
                key, value = match.groups()
                values[key] = value.strip().strip("'\"")
                hard.add(key)
    return values, hard


def launcher_reads(path: str) -> set[str]:
    """Upper-case variables the launcher expands without a fallback of its own."""
    text = open(path, encoding="utf-8").read()
    strict, guarded = set(), set()
    for name, operator in USED.findall(text):
        (guarded if operator else strict).add(name)
    assigned = set(re.findall(r"^\s*([A-Z][A-Z0-9_]{2,})=", text, re.MULTILINE))
    shell = {"BASH_SOURCE", "PWD", "HOME", "PATH", "IFS", "NODE_RANK", "SECONDS"}
    # A name the launcher also expands with a fallback (${X:-}, ${X:=...}) is one it
    # defaults or tests itself before the bare use.
    return strict - guarded - assigned - shell


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--shipped", required=True, help="the config.env the box runs")
    parser.add_argument("--repo", default="recipe/config.env")
    parser.add_argument("--launcher", default="recipe/serve_vllm_node.sh")
    parser.add_argument("--lane", choices=("v0.31.0", "v0.34.0"), help="lane to check (default: the file's tag)")
    parser.add_argument("--steering", action="store_true", help="check for a steering launch even if no sidecar is configured")
    parser.add_argument("--redact", action="store_true")
    args = parser.parse_args()
    shipped, shipped_hard = parse(args.shipped)
    repo, _ = parse(args.repo)

    def show(key: str, value: str | None) -> str:
        if value is None:
            return "<absent>"
        if args.redact and SENSITIVE.search(key) and value:
            return "<set>"
        return repr(value)

    print(f"1. defaults: {args.shipped} (shipped) vs {args.repo} (repo)")
    differ = sorted(k for k in set(shipped) & set(repo) if shipped[k] != repo[k])
    for key in differ:
        print(f"   differs        {key:<42} shipped {show(key, shipped[key])}  repo {show(key, repo[key])}")
    for key in sorted(set(shipped) - set(repo)):
        print(f"   only shipped   {key:<42} {show(key, shipped[key])}")
    for key in sorted(set(repo) - set(shipped)):
        print(f"   only repo      {key:<42} {show(key, repo[key])}")
    if not differ and set(shipped) == set(repo):
        print("   identical key for key")

    stops = 0
    print(f"\n2. variables {args.launcher} reads that the shipped file does not define")
    missing = sorted(launcher_reads(args.launcher) - set(shipped))
    for key in missing:
        stops += 1
        where = f"repo default {show(key, repo.get(key))}" if key in repo else "not in the repo file either"
        print(f"   UNBOUND        {key:<42} ({where})")
    if not missing:
        print("   none")

    get = lambda key, default="": shipped.get(key, default)  # noqa: E731
    lane = args.lane or get("VLLM_OVERLAY_TAG") or "v0.31.0"
    steering = args.steering or bool(get("VLLM_STEERING_SIDECAR"))
    print(f"\n3. shipped values against what the launcher refuses: lane {lane}"
          + (", with steering" if steering else ", steering not configured (its lines are for information)"))
    autotune = get("VLLM_PATCH_PP_AUTOTUNE", "0") or "0"
    checks = [
        (lane, True, "VLLM_PATCH_PP_AUTOTUNE", lane == "v0.31.0" or autotune == "0",
         f"is {autotune}; must be 0 on v0.34.0 (the launcher refuses the v0.31.0 autotune overlay there)"),
        (lane, True, "VLLM_PATCH_PP2_FLAG", get("VLLM_PATCH_PP2_FLAG", "0") == "1", "must be 1 for PP + DSpark"),
        (lane, True, "VLLM_PARALLEL", get("VLLM_PARALLEL", "pp") == "pp", "must be pp"),
        ("steering", steering, "VLLM_OVERLAY_TAG", lane == "v0.34.0", "steering is wired into the v0.34.0 lane only"),
        ("steering", steering, "MODEL_DIR_RANK0 = MODEL_DIR_RANK1", get("MODEL_DIR_RANK0") == get("MODEL_DIR_RANK1"),
         "both stages must load the same stock checkpoint"),
        ("steering", steering, "VLLM_RUST_FRONTEND", get("VLLM_RUST_FRONTEND", "0") == "0", "must be 0"),
        ("steering", steering, "VLLM_PATCH_GRAPH_ATTR", get("VLLM_PATCH_GRAPH_ATTR", "0") == "0",
         "must be 0: that overlay replaces PYTHONPATH"),
        ("steering", steering, "VLLM_STEERING_REPLAY", get("VLLM_STEERING_REPLAY", "off") in ("off", "keep"),
         "must be off or keep"),
        ("steering", steering, "VLLM_STEERING_KERNEL", get("VLLM_STEERING_KERNEL", "torch") in ("torch", "fused"),
         "must be torch or fused"),
        ("steering", False, "VLLM_STEERING_SIDECAR", get("VLLM_STEERING_SIDECAR").startswith("/") or not get("VLLM_STEERING_SIDECAR"),
         "a relative path is taken from the recipe directory (revision 2 handed it to docker as a volume name)"),
        ("steering", False, "VLLM_PREFIX_CACHING", get("VLLM_PREFIX_CACHING", "1") in ("0", "1"),
         "either value works; the probe's T6 needs 1 to see the cache"),
    ]
    for scope, counts, key, good, why in checks:
        if not good and counts:
            stops += 1
        mark = "ok" if good else ("REFUSED" if counts else "would refuse")
        print(f"   {mark:<13} {scope:<9} {key:<36} {why}")
    for key in sorted(shipped_hard & {k for k in repo}):
        if key.startswith("VLLM_") and key not in ("VLLM_ATTENTION_CONFIG", "VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS"):
            print(f"   note     hard-assigned  {key:<36} an environment override cannot change it")
    what = f"a {lane} launch" + (" with steering" if steering else "")
    print(f"\nCONFIG-DELTA: {len(differ)} defaults differ, {len(missing)} unbound, "
          f"{stops - len(missing)} refused; {what + ' would stop' if stops else 'nothing here stops ' + what}")
    return 1 if stops else 0


if __name__ == "__main__":
    sys.exit(main())
