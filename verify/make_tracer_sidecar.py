#!/usr/bin/env python3
"""A tracer sidecar: steering whose effect cannot be mistaken for anything else.

What it is for. T4 of verify/steering_probe.py asks whether a request's rows ever read
another request's scale. Which rows read which scale is decided before the operator is
applied and does not depend on what the operator is, so the question can also be put
with an operator nobody can miss. It is the second instrument, and it costs a test boot:
the first is T4 on the serving sidecar, which costs none.

What it does. At scale 1 it multiplies the two hooked outputs of every steered layer
(attention and the shared experts) by --gain, 1/64 by default. At scale 0 a row is
untouched, as with any sidecar. The read and write vectors are present because the
format wants them and are too small to matter.

What the pair showed on 2026-11-20 (as reported from the test boot). A steered answer
was 15 generated tokens with no content and no reasoning text, and requests without a
scale decoded as on any other boot (369.9 tok/s, 3.69 tokens per round). Revision 4 of
the probe could not see that: its distance compared only the tokens that both top-20
lists name, a tracer row's list shares next to none with a stock row's, and it read
0.008 while the text differed on 4 of 4 prompts. Its T2 failed and T4 was skipped.
Revision 5 measures what the two lists prove, aims T4's requests at the positions where
the scales differ, and sorts whole answers by length, which needs no logprobs at all.

What it needs. The steering module of revision 3 or later. The file is in the derived
format, which revision 2's module does not read: it refuses with "tensor inventory
differs from layers 4 through 36" when the model is built, minutes into the boot.
patches/pp2-steering/stage_check.py says so in one line before a boot, and
recipe/serve_vllm_node.sh runs it on each node.

How it is kept off a serving lane. It is written in the derived (v2) format with the
verdict "TRACER ...". The module refuses such a file unless the boot sets
PP2_STEERING_ALLOW_UNVERIFIED=1, and the file name must contain TRACER. A lane that has
it loaded answers scale-1 requests with nothing: test boots only, router stripping on.

    python3 verify/make_tracer_sidecar.py --like <published sidecar> --out <dir>/TRACER-steering.safetensors
    python3 patches/pp2-steering/stage_check.py --sidecar <dir>/TRACER-steering.safetensors --allow-unverified
    sha256sum <dir>/TRACER-steering.safetensors
    # boot with VLLM_STEERING_SIDECAR=<that file> VLLM_STEERING_SHA256=<that sha>
    #           EXTRA_ENV='PP2_STEERING_ALLOW_UNVERIFIED=1'
    python3 verify/steering_probe.py --url http://<rank0>:8001 --refusal-prompts <file> \
        --only T4,T8 --out results/steering-probe-tracer.json

--like copies the layer range and the vector size from the published sidecar. Without
it, give --hidden and --layers. --selftest writes one to a temporary file, loads it
through the module and checks what it does to a batch of rows; it needs torch and
safetensors and no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent / "patches" / "pp2-steering"
V2_FORMAT = "pp2-steering-v2"
TINY = 1e-20  # the write vector's one entry: there, and below anything BF16 can show


def header_of(path: str) -> dict:
    """The safetensors header, read without the library."""
    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        return json.loads(handle.read(size))


def like(path: str) -> tuple[list[int], int]:
    """(layers, vector size) of a published (v1) or derived (v2) sidecar."""
    header = header_of(path)
    layers, sizes = set(), set()
    for name, entry in header.items():
        match = re.fullmatch(r"layers\.(\d+)\.(direction|(?:attn|shared)\.(?:read|write))", name)
        if match:
            layers.add(int(match.group(1)))
            sizes.add(int(entry["shape"][-1]))
    if not layers or len(sizes) != 1:
        raise SystemExit(f"{path}: no steering vectors of one size found (sizes {sorted(sizes)})")
    return sorted(layers), sizes.pop()


def build(layers: list[int], hidden: int, gain: float) -> tuple[dict, dict[str, str]]:
    import torch

    tensors = {}
    for layer in layers:
        for site in ("attn", "shared"):
            read = torch.zeros(hidden, dtype=torch.float32)
            read[0] = 1.0
            write = torch.zeros(hidden, dtype=torch.float32)
            write[0] = TINY
            tensors[f"layers.{layer}.{site}.read"] = read
            tensors[f"layers.{layer}.{site}.write"] = write
            tensors[f"layers.{layer}.{site}.row_gain"] = torch.full((hidden,), gain, dtype=torch.float32)
    metadata = {
        "format": V2_FORMAT,
        "verdict": "TRACER: not an abliteration. Scales both hooked outputs of every steered layer by "
                   f"{gain:g} at scale 1. For verify/steering_probe.py on a test boot only.",
        "tracer_gain": repr(gain),
        "layers": f"{layers[0]}-{layers[-1]}",
        "written_by": "verify/make_tracer_sidecar.py",
    }
    return tensors, metadata


def write(path: str, layers: list[int], hidden: int, gain: float) -> str:
    from safetensors.torch import save_file

    if "TRACER" not in os.path.basename(path):
        raise SystemExit("the output file name must contain TRACER, so nobody takes it for a real sidecar")
    if not 0.0 < gain < 1.0:
        raise SystemExit("--gain must be between 0 and 1 (the format wants a positive gain)")
    tensors, metadata = build(layers, hidden, gain)
    save_file(tensors, path, metadata=metadata)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest


def selftest() -> int:
    import torch

    sys.path.insert(0, str(MODULE_DIR))
    sys.dont_write_bytecode = True
    import pp2_steering as ps

    failures = []

    def check(name: str, good: bool, detail: str) -> None:
        print(f"{'ok  ' if good else 'FAIL'}  {name:<52} {detail}")
        if not good:
            failures.append(name)

    with tempfile.TemporaryDirectory() as work:
        path = os.path.join(work, "TRACER-selftest.safetensors")
        gain, hidden, layers = 1 / 64, 5120, list(range(4, 37))
        digest = write(path, layers, hidden, gain)
        check("written", os.path.getsize(path) > 0, f"{os.path.getsize(path)} bytes, sha256 {digest[:16]}")
        check("layer range and vector size can be read back", like(path) == (layers, hidden),
              f"layers {layers[0]}-{layers[-1]}, {hidden}")

        metadata = ps.sidecar_metadata(path)
        try:
            ps.check_verdict(metadata, False)
            refused = False
        except ValueError:
            refused = True
        check("the module refuses it on an ordinary boot", refused, "its verdict does not say REBUILT")
        try:
            ps.check_verdict(metadata, True)
            allowed = True
        except ValueError:
            allowed = False
        check("...and loads it with PP2_STEERING_ALLOW_UNVERIFIED=1", allowed, "experiments only")

        operators = ps.load_sidecar(path, torch.device("cpu"), allow_unverified=True)
        sites = sum(len(by_site) for by_site in operators.values())
        check("one operator per steered output", sorted(operators) == layers and sites == 2 * len(layers),
              f"{sites} operators on layers {min(operators)}-{max(operators)}")

        generator = torch.Generator().manual_seed(1017)
        rows = torch.randn(6, hidden, generator=generator).to(torch.bfloat16)
        scale = torch.tensor([0.0, 1.0, 0.0, 0.5, 1.0, 0.0])
        out = ps.steer(rows.clone(), operators[20]["attn"], scale.unsqueeze(1), (scale == 0).unsqueeze(1))
        stock = bool(torch.equal(out[scale == 0], rows[scale == 0]))
        check("rows at scale 0 come back bit for bit", stock, "3 of 6 rows")
        full = out[scale == 1].float() / rows[scale == 1].float()
        check("rows at scale 1 are the input times the gain",
              bool(((full - gain).abs() <= 0.02 * gain).float().mean() > 0.99),
              f"ratio {float(full.median()):.6f} (gain {gain:.6f})")
        half = out[scale == 0.5].float() / rows[scale == 0.5].float()
        want = 1 + 0.5 * (gain - 1)
        check("rows at scale 0.5 are halfway", bool(((half - want).abs() <= 0.02).float().mean() > 0.99),
              f"ratio {float(half.median()):.4f} (expected {want:.4f})")
        ratio = float(out[1].float().norm() / rows[1].float().norm())
        check("a steered row keeps almost nothing of the output", ratio < 0.02, f"norm ratio {ratio:.4f}")

        try:
            write(os.path.join(work, "steering.safetensors"), layers, hidden, gain)
            named = False
        except SystemExit:
            named = True
        check("a file name without TRACER is refused", named, "so it cannot pass for a real sidecar")
    print(f"\nTRACER-SELFTEST {'PASS' if not failures else 'FAIL'}: {len(failures)} check(s) failed")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--like", help="a published or derived sidecar to copy the layer range and vector size from")
    parser.add_argument("--hidden", type=int, help="vector size (without --like)")
    parser.add_argument("--layers", help="steered layers as first-last, e.g. 4-36 (without --like)")
    parser.add_argument("--gain", type=float, default=1 / 64, help="what a steered output is multiplied by at scale 1")
    parser.add_argument("--out", help="file to write; its name must contain TRACER")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.out:
        parser.error("--out is required")
    if args.like:
        layers, hidden = like(args.like)
    elif args.hidden and args.layers:
        first, last = (int(v) for v in args.layers.split("-"))
        layers, hidden = list(range(first, last + 1)), args.hidden
    else:
        parser.error("give --like, or both --hidden and --layers")
    digest = write(args.out, layers, hidden, args.gain)
    print(f"wrote {args.out}: layers {layers[0]}-{layers[-1]}, {2 * len(layers)} operators of size {hidden}, "
          f"gain {args.gain:g}")
    print(f"sha256 {digest}")
    print("load it only on a test boot: VLLM_STEERING_SIDECAR=<this file> VLLM_STEERING_SHA256=<this sha> "
          "EXTRA_ENV='PP2_STEERING_ALLOW_UNVERIFIED=1'")
    print("it needs the steering module of revision 3 or later (revision 2's reads only the published format); "
          f"check what is staged: python3 patches/pp2-steering/stage_check.py --sidecar {args.out} --allow-unverified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
