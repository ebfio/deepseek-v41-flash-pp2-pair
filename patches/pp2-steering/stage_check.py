#!/usr/bin/env python3
"""Will the steering module staged here load the sidecar staged here? One line, before a boot.

Standard library only: it reads the sidecar's header and the module's source and imports
neither torch nor the module. recipe/serve_vllm_node.sh runs it on each node before it
starts the container; run it by hand, or from a role, after staging either file:

    python3 patches/pp2-steering/stage_check.py --sidecar <file.safetensors> \
        [--module patches/pp2-steering/pp2_steering.py] [--allow-unverified]

    STAGE-CHECK OK: ...        exit 0
    STAGE-CHECK REFUSED: ...   exit 1   the module would refuse this sidecar when the
                                        model is built, minutes into a boot

Why it exists. On 2026-11-20 a tracer sidecar (derived format) was staged next to
revision 2's module, which reads only the published format. The module said so nine
minutes into the boot, in words that name the sidecar ("tensor inventory differs from
layers 4 through 36") and not the mistake, which was the module left in place.

What it checks is what the module checks at load that can be read off the two files:
which formats the staged module reads, the sidecar's format, its tensor names, types and
shapes, and for a derived sidecar the verdict rule (a verdict that does not start with
REBUILT loads only with PP2_STEERING_ALLOW_UNVERIFIED=1). What it leaves to the module:
the values (finite, unit norm, non-zero, positive gains), the sha256 (the node script
checks that) and whether the vector size is the model's. test_pp2_steering.py holds the
two against each other on a set of sidecars, so they cannot drift apart unnoticed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
from pathlib import Path

V2_FORMAT = "pp2-steering-v2"
V2_KEY = r"layers\.(\d+)\.(attn|shared)\.(read|write|row_gain)"
PUBLISHED_LAYERS = range(4, 37)
PUBLISHED_NAMES = ("alpha", "attn_row_gain", "direction")
# sha256 of patches/pp2-steering/pp2_steering.py as each bundle revision shipped it.
KNOWN_MODULES = {
    "06d95b972c5ad29cc251ec413e0dce123b986ffce90e92368e0035da351a03b5": "revision 2",
    "720e54b2a8271a7ed6d9d3a1cbd8b3f8548bcdd30fc87ef72e0985f637de97ac": "revision 3",
    "9d34c0a32494b6bc19af9d29771a4a22a150d4e921d6539712f5f4d70f48ad8e": "revision 4 (unchanged in revision 5)",
}


class Refused(Exception):
    pass


def read_header(path: str) -> tuple[dict[str, dict], dict[str, str]]:
    """(tensor name -> {dtype, shape}, metadata) of a safetensors file."""
    try:
        with open(path, "rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            if not 2 <= length <= 100 << 20:
                raise ValueError("header length out of range")
            header = json.loads(handle.read(length))
    except (OSError, ValueError, struct.error) as exc:
        raise Refused(f"{path} is not a readable safetensors file ({exc})") from None
    metadata = dict(header.pop("__metadata__", None) or {})
    return header, metadata


def module_facts(path: str) -> tuple[str, str, bool]:
    """(sha256, what revision that is, whether the module reads the derived format)."""
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise Refused(f"no steering module at {path} ({exc})") from None
    digest = hashlib.sha256(data).hexdigest()
    reads_derived = f'"{V2_FORMAT}"'.encode() in data
    name = KNOWN_MODULES.get(digest, "a file no bundle revision shipped")
    return digest, name, reads_derived


def check_published(tensors: dict[str, dict]) -> str:
    expected = {f"layers.{layer}.{name}" for layer in PUBLISHED_LAYERS for name in PUBLISHED_NAMES}
    if set(tensors) != expected:
        missing, extra = sorted(expected - set(tensors)), sorted(set(tensors) - expected)
        raise Refused("the sidecar is in neither format the module knows: not the published one (layers "
                      f"{PUBLISHED_LAYERS.start} through {PUBLISHED_LAYERS.stop - 1}, each with alpha, attn_row_gain "
                      f"and direction; missing {missing[:3]}, extra {extra[:3]}) and not the derived one "
                      f"(no format={V2_FORMAT} in its metadata)")
    sizes = set()
    for layer in PUBLISHED_LAYERS:
        prefix = f"layers.{layer}"
        alpha, direction, gain = (tensors[f"{prefix}.{name}"] for name in ("alpha", "direction", "attn_row_gain"))
        if alpha["dtype"] != "F32" or alpha["shape"] != []:
            raise Refused(f"{prefix}.alpha must be a scalar FP32 tensor")
        if direction["dtype"] != "BF16" or len(direction["shape"]) != 1:
            raise Refused(f"{prefix}.direction must be a BF16 vector")
        if gain["dtype"] != "BF16" or gain["shape"] != direction["shape"]:
            raise Refused(f"{prefix}.attn_row_gain must be BF16 with the direction's shape")
        sizes.add(direction["shape"][0])
    return (f"published sidecar, layers {PUBLISHED_LAYERS.start} through {PUBLISHED_LAYERS.stop - 1}, vectors of "
            f"{'/'.join(str(size) for size in sorted(sizes))}")


def check_derived(tensors: dict[str, dict], metadata: dict[str, str], allow_unverified: bool) -> str:
    if metadata.get("format") != V2_FORMAT:
        raise Refused(f"the sidecar has derived-format tensor names and no format={V2_FORMAT} in its metadata")
    verdict = metadata.get("verdict", "")
    tracer = verdict.startswith("TRACER")
    if not verdict.startswith("REBUILT") and not allow_unverified:
        raise Refused(
            f"the sidecar's verdict is {verdict[:60] or '<none>'!r}, not REBUILT: the module loads it only with "
            "PP2_STEERING_ALLOW_UNVERIFIED=1 (EXTRA_ENV='PP2_STEERING_ALLOW_UNVERIFIED=1')"
            + (". It is a tracer: a test boot only, never a serving one" if tracer else ""))
    parts: dict[tuple[int, str], set[str]] = {}
    sizes = set()
    for name, entry in tensors.items():
        match = re.fullmatch(V2_KEY, name)
        if match is None:
            raise Refused(f"the derived sidecar has an unexpected tensor {name!r}")
        if entry["dtype"] != "F32" or len(entry["shape"]) != 1:
            raise Refused(f"{name} must be an FP32 vector")
        parts.setdefault((int(match.group(1)), match.group(2)), set()).add(match.group(3))
        sizes.add(entry["shape"][0])
    if not parts:
        raise Refused("the derived sidecar holds no operators")
    if len(sizes) != 1:
        raise Refused(f"the derived sidecar mixes vector sizes {sorted(sizes)}")
    for (layer, site), names in sorted(parts.items()):
        if not {"read", "write"} <= names:
            raise Refused(f"layers.{layer}.{site} needs both read and write")
    layers = sorted({layer for layer, _ in parts})
    kind = "TRACER sidecar (test boots only)" if tracer else "derived sidecar"
    state = "" if verdict.startswith("REBUILT") else ", loaded only because PP2_STEERING_ALLOW_UNVERIFIED=1"
    return (f"{kind}, {len(parts)} operators on layers {layers[0]} through {layers[-1]}, vectors of {sizes.pop()}, "
            f"verdict {verdict[:40]!r}{state}")


def check(sidecar: str, module: str, allow_unverified: bool) -> str:
    digest, revision, reads_derived = module_facts(module)
    tensors, metadata = read_header(sidecar)
    derived = metadata.get("format") == V2_FORMAT or any(re.fullmatch(V2_KEY, name) for name in tensors)
    module_text = f"the module is {revision} (sha256 {digest[:12]})"
    if derived and not reads_derived:
        raise Refused(
            f"the sidecar is in the derived format ({V2_FORMAT}: tracers and sidecar_vs_weights.py --derive "
            f"write it) and {module_text}, which reads only the published format. It would say \"tensor "
            "inventory differs from layers 4 through 36\" when the model is built. Stage "
            "patches/pp2-steering/ from revision 3 or later")
    what = check_derived(tensors, metadata, allow_unverified) if derived else check_published(tensors)
    return f"{what}; {module_text}"


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--module", default=str(Path(__file__).resolve().parent / "pp2_steering.py"))
    parser.add_argument("--allow-unverified", action="store_true",
                        help="as PP2_STEERING_ALLOW_UNVERIFIED=1 (which is also read from the environment)")
    args = parser.parse_args()
    allow = args.allow_unverified or os.environ.get("PP2_STEERING_ALLOW_UNVERIFIED", "0") == "1"
    try:
        print(f"STAGE-CHECK OK: {check(args.sidecar, args.module, allow)}")
    except Refused as exc:
        print(f"STAGE-CHECK REFUSED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
