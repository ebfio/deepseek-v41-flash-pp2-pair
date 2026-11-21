#!/usr/bin/env python3
"""The fused steering kernel on a GPU: is it right, and what do the hooks cost?

Needs one idle CUDA device with torch and Triton (the vLLM image has both). No model, no
engine, no weights: it works on random activations, with the real sidecar's operators
when --sidecar is given. Run it before VLLM_STEERING_KERNEL=fused goes on a lane, on a
box whose GPU is not serving:

    docker run --rm --gpus '"device=0"' --entrypoint python3 \
        -v "$PWD:/repo:ro" -v "$SIDECAR:/sidecar.safetensors:ro" \
        vllm/vllm-openai:v0.34.0 \
        /repo/verify/steering_kernel_check.py --sidecar /sidecar.safetensors --json -

1. Correctness, against pp2_steering.steer() (the torch path): rows at scale 0 must come
   back bit-identical, and every other value must equal the torch path's or differ from it
   by one unit in the last place (the two sum the dot product in a different order).
2. Cost of a decode step: the hooks of each pipeline stage, on [rows, hidden] BF16, inside
   a CUDA graph that is replayed --replays times. Microseconds per replay, torch path and
   fused, with every row at scale 0 and with every row at scale 1. This is the latency the
   hooks add to one forward pass of that stage; in the engine the same kernels sit inside
   the model's own graph, some of them on the auxiliary stream, so read it as an upper
   bound that is close.
3. Cost of a prefill step: the same hooks eagerly on --prefill-rows rows. Milliseconds per
   pass and the peak of extra memory.

Exit status 0 when part 1 passes. Parts 2 and 3 are measurements, not verdicts: they are
this GPU's, not a GB300's unless it runs on one.

--interpret runs part 1 only, on the CPU, in Triton's interpreter (TRITON_INTERPRET=1 must
be set before Python starts). The interpreter truncates where the GPU rounds, so there the
share of equal values means nothing; the one-ulp bound still holds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "patches" / "pp2-steering"))
sys.dont_write_bytecode = True

STAGES = {0: range(0, 20), 1: range(20, 40)}  # the pair's 20/20 split


def ulp_of(reference, dtype):
    """One unit in the last place of `dtype` at each reference value."""
    import torch

    mantissa_bits = 8 if dtype == torch.bfloat16 else 24
    _, exponent = torch.frexp(reference.float().abs().clamp_min(1e-30))
    return torch.ldexp(torch.ones_like(reference, dtype=torch.float32), exponent - mantissa_bits)


def synthetic_operators(ps, hidden: int, device, generator):
    """Operators shaped like the published sidecar's: layers 4..36, a row gain on attention."""
    import torch

    layers = {}
    for layer in range(4, 37):
        raw = torch.randn(hidden, generator=generator)
        direction = (raw / raw.norm()).to(device)
        gain = (0.02 * torch.randn(hidden, generator=generator)).to(device)
        layers[layer] = {
            "attn": ps.SiteSteering(direction, -3.2 * direction, gain),
            "shared": ps.SiteSteering(direction, -3.2 * direction),
        }
    return layers


def correctness(ps, kernel, layers, device, generator, interpret: bool) -> tuple[list[dict], bool]:
    import torch

    first = layers[min(layers)]
    hidden = int(first["attn"].read.shape[0])
    scale = torch.tensor([0.0, 1.0, 0.5, 0.0, 0.25, 1.0, 0.0, 0.75, 1.0, 0.0], device=device).unsqueeze(1)
    stock = scale.view(-1) == 0
    rows, ok = [], True
    for dtype in (torch.bfloat16, torch.float32):
        for name, site in (("attention, " + ("row gain" if first["attn"].gain_delta is not None else "no gain"), first["attn"]),
                           ("shared expert, " + ("row gain" if first["shared"].gain_delta is not None else "no gain"), first["shared"])):
            x = torch.randn(scale.shape[0], hidden, generator=generator).to(device=device, dtype=dtype)
            reference = ps.steer(x, site, scale)
            fused = x.clone()
            kernel.launch(fused, site.read, site.write, site.gain_delta, scale)
            if device.type == "cuda":
                torch.cuda.synchronize()
            difference = (fused.float() - reference.float()).abs()
            if dtype == torch.bfloat16:
                worst = float((difference / ulp_of(reference, dtype)).max())
                within = worst <= 1.0
            else:  # FP32 values near zero make a per-value ulp meaningless: bound it by the row
                worst = float((difference / reference.float().abs().amax(dim=1, keepdim=True)).max())
                within = worst <= 1e-5
            equal_share = float((fused[~stock] == reference[~stock]).float().mean())
            row = {
                "dtype": str(dtype).replace("torch.", ""),
                "site": name,
                "stock_rows_untouched": bool(torch.equal(fused[stock], x[stock])),
                "steered_rows_changed": not bool(torch.equal(fused[~stock], x[~stock])),
                "worst": worst,
                "worst_unit": "ulp" if dtype == torch.bfloat16 else "of the row's largest value",
                "within_bound": within,
                "equal_share": equal_share,
            }
            # A BF16 value differs only when the two FP32 results straddle a rounding
            # boundary, which is rare; FP32 values differ in the last bit wherever the
            # order of the sum matters, so only the bound is judged there.
            row["ok"] = (
                row["stock_rows_untouched"]
                and row["steered_rows_changed"]
                and within
                and (interpret or dtype != torch.bfloat16 or equal_share >= 0.999)
            )
            ok &= row["ok"]
            rows.append(row)
    return rows, ok


def stage_sites(layers, stage: int):
    return [site for layer, sites in sorted(layers.items()) if layer in STAGES[stage] for site in sites.values()]


def decode_cost(ps, kernel, layers, device, rows_list, replays: int) -> list[dict]:
    """Per stage and batch size: one CUDA graph holding that stage's hooks, replayed."""
    import torch

    out = []
    for stage in STAGES:
        sites = stage_sites(layers, stage)
        if not sites:
            continue
        hidden = int(sites[0].read.shape[0])
        for rows in rows_list:
            x = torch.randn(rows, hidden, device=device).to(torch.bfloat16)
            token_scale = torch.zeros(rows, 1, device=device)
            stock_rows = torch.ones(rows, 1, dtype=torch.bool, device=device)
            holder: list = []

            def torch_path():
                holder.clear()
                for site in sites:
                    holder.append(ps.steer(x, site, token_scale, stock_rows))

            def fused_path():
                for site in sites:
                    kernel.launch(x, site.read, site.write, site.gain_delta, token_scale)

            entry = {"stage": stage, "rows": rows, "hooks": len(sites),
                     "torch_kernels": sum(site.kernels for site in sites), "fused_kernels": len(sites)}
            for name, body in (("torch", torch_path), ("fused", fused_path)):
                body()  # warm: compiles and allocates outside the capture
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    body()
                for value in (0.0, 1.0):
                    token_scale.fill_(value)
                    stock_rows.fill_(value == 0.0)
                    # The fused path edits x in place, so at scale 1 x grows from replay to
                    # replay until it overflows. That does not change what a replay costs.
                    x.normal_()
                    for _ in range(50):
                        graph.replay()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    for _ in range(replays):
                        graph.replay()
                    torch.cuda.synchronize()
                    entry[f"{name}_us_scale{int(value)}"] = (time.perf_counter() - started) / replays * 1e6
                del graph
            out.append(entry)
    return out


def prefill_cost(ps, kernel, layers, device, rows_list, passes: int) -> list[dict]:
    import torch

    out = []
    for stage in STAGES:
        sites = stage_sites(layers, stage)
        if not sites:
            continue
        hidden = int(sites[0].read.shape[0])
        for rows in rows_list:
            x = torch.randn(rows, hidden, device=device).to(torch.bfloat16)
            token_scale = torch.zeros(rows, 1, device=device)
            stock_rows = torch.ones(rows, 1, dtype=torch.bool, device=device)
            entry = {"stage": stage, "rows": rows, "hooks": len(sites)}
            for name in ("torch", "fused"):
                for value in (0.0, 1.0):
                    token_scale.fill_(value)
                    stock_rows.fill_(value == 0.0)
                    x.normal_()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                    before = torch.cuda.memory_allocated(device)
                    started = time.perf_counter()
                    for _ in range(passes):
                        for site in sites:
                            if name == "torch":
                                ps.steer(x, site, token_scale, stock_rows)
                            else:
                                kernel.launch(x, site.read, site.write, site.gain_delta, token_scale)
                    torch.cuda.synchronize()
                    entry[f"{name}_ms_scale{int(value)}"] = (time.perf_counter() - started) / passes * 1e3
                    entry[f"{name}_peak_mib_scale{int(value)}"] = (
                        torch.cuda.max_memory_allocated(device) - before
                    ) / 2**20
            out.append(entry)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--sidecar", help="steering sidecar, published or derived (default: random operators)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden", type=int, default=5120, help="without --sidecar")
    parser.add_argument("--decode-rows", default="6,24,96", help="rows of a decode step: streams x (1 + drafts)")
    parser.add_argument("--prefill-rows", default="8192,16384")
    parser.add_argument("--replays", type=int, default=2000)
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--interpret", action="store_true", help="part 1 only, on the CPU, in Triton's interpreter")
    parser.add_argument("--json", help="write the results as JSON to this file ('-' = stdout)")
    args = parser.parse_args()

    import torch

    import pp2_steering as ps

    if args.interpret:
        if os.environ.get("TRITON_INTERPRET") != "1":
            print("--interpret needs TRITON_INTERPRET=1 in the environment", file=sys.stderr)
            return 2
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            print("no CUDA device here: this check needs a GPU (or --interpret for part 1 on a CPU)", file=sys.stderr)
            return 2
        device = torch.device(args.device)
    kernel = ps.load_kernel()
    generator = torch.Generator().manual_seed(1017)
    if args.sidecar:
        layers = ps.load_sidecar(args.sidecar, device, allow_unverified=True)
        source = f"{args.sidecar} (sha256 {ps.file_sha256(args.sidecar)[:16]})"
    else:
        layers = synthetic_operators(ps, args.hidden, device, generator)
        source = "random operators shaped like the published sidecar"
    import triton

    where = "CPU, Triton interpreter" if args.interpret else torch.cuda.get_device_name(device)
    capability = "" if args.interpret else " sm_%d%d" % torch.cuda.get_device_capability(device)
    print(f"{where}{capability}; torch {torch.__version__}, Triton {triton.__version__}; {source}")

    rows, ok = correctness(ps, kernel, layers, device, generator, args.interpret)
    print("\n1. fused kernel against the torch path")
    for row in rows:
        print(f"  {'ok' if row['ok'] else 'FAIL':<4} {row['dtype']:<8} {row['site']:<26} stock rows untouched: "
              f"{row['stock_rows_untouched']}; worst difference {row['worst']:.3g} {row['worst_unit']}; "
              f"equal to the torch path on {row['equal_share']:.5f} of the steered values")
    result = {"device": where + capability, "sidecar": source, "correctness": rows, "correct": ok}
    if not args.interpret:
        decode = decode_cost(ps, kernel, layers, device, [int(v) for v in args.decode_rows.split(",")], args.replays)
        print(f"\n2. one decode step's hooks inside a CUDA graph, microseconds per replay ({args.replays} replays)")
        print(f"  {'stage':>5} {'rows':>5} {'hooks':>5} | {'torch':>6} kernels  scale 0  scale 1 | {'fused':>6} kernels  scale 0  scale 1")
        for e in decode:
            print(f"  {e['stage']:>5} {e['rows']:>5} {e['hooks']:>5} | {e['torch_kernels']:>14} {e['torch_us_scale0']:>8.0f} "
                  f"{e['torch_us_scale1']:>8.0f} | {e['fused_kernels']:>14} {e['fused_us_scale0']:>8.0f} {e['fused_us_scale1']:>8.0f}")
        both = [sum(e[key] for e in decode if e["rows"] == decode[0]["rows"]) for key in ("torch_us_scale0", "fused_us_scale0")]
        print(f"  both stages at {decode[0]['rows']} rows, every row at scale 0: torch {both[0]:.0f} us, fused {both[1]:.0f} us "
              "per token round trip")
        prefill = prefill_cost(ps, kernel, layers, device, [int(v) for v in args.prefill_rows.split(",")], args.passes)
        print(f"\n3. one prefill step's hooks, eager, milliseconds per pass ({args.passes} passes); peak extra memory in MiB")
        print(f"  {'stage':>5} {'rows':>6} | torch: scale 0  scale 1  peak | fused: scale 0  scale 1  peak")
        for e in prefill:
            print(f"  {e['stage']:>5} {e['rows']:>6} | {e['torch_ms_scale0']:>14.2f} {e['torch_ms_scale1']:>8.2f} "
                  f"{max(e['torch_peak_mib_scale0'], e['torch_peak_mib_scale1']):>5.0f} | {e['fused_ms_scale0']:>14.2f} "
                  f"{e['fused_ms_scale1']:>8.2f} {max(e['fused_peak_mib_scale0'], e['fused_peak_mib_scale1']):>5.0f}")
        result.update(decode=decode, prefill=prefill)
    print(f"\nSTEERING-KERNEL {'PASS' if ok else 'FAIL'}: part 1 "
          f"{'holds' if ok else 'does not hold'} on {where}"
          + ("" if not args.interpret else " (interpreter: the share of equal values is not judged)"))
    if args.json:
        text = json.dumps(result, indent=2)
        if args.json == "-":
            print(text)
        else:
            Path(args.json).write_text(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
