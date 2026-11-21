#!/usr/bin/env python3
"""Which float edit was the abliterated checkpoint built with, and can the steered lane apply it?

Per-request steering edits two activations per layer (the attention module's output and
the shared expert's output). At scale 1 that equals an edit of the two matrices that write
those outputs. This script compares the stock and the abliterated checkpoint on disk and
answers three questions, per layer:

  1. Which tensors were rewritten, and are they exactly the two writers the hooks sit on?
  2. REBUILD. Which float edit, quantised back onto the abliterated checkpoint's own grid,
     reproduces its bytes? Candidates: the published sidecar's operator, the same direction
     with other norm rules and strengths, and (--derive, or when those fail) an operator
     fitted to the bytes.
  3. What the lane has to apply to equal that float edit, written as a steering sidecar
     (--derive OUT), and how much of the float edit the checkpoint itself still carries.

Why the bytes and not the dequantised weights: an abliterated checkpoint of a quantised
model is the stock checkpoint dequantised, edited in float and quantised again. A value
only changes when its edit exceeds half its quantisation step, so the difference between
the two dequantised checkpoints is a thresholded copy of the float edit. Read as a
matrix it is weaker than the edit that made it and its dominant direction is bent, even
when the candidate operator is exactly right (the "requantised view" columns below show
that view; revision 2 of this script judged by it, which was wrong). Quantising a
candidate's float result onto the checkpoint's grid and comparing values is exact
instead: the agreement J is the share of values, among those either side changed, that
come out identical. An exact candidate scores 0.97 to 1.00 (below 1 because the sidecar
holds a BF16 copy of the direction the builder had in FP32); 3 % off in strength about
0.89; a direction 15 degrees off 0.2 to 0.4. 0.95 and above is reported as rebuilt.

CPU and disk only, read-only. Needs torch. Run it in the serving image so no GPU and no
service is touched (about half a minute per matrix on 8 threads; --layers for a first look):

    docker run --rm --entrypoint python3 -e CUDA_VISIBLE_DEVICES= \
        -v /models/derived/dsv41-1017-official:/stock:ro -v <abliterated dir>:/ablit:ro \
        -v "$SIDECAR:/sidecar.safetensors:ro" -v "$PWD:/repo" \
        vllm/vllm-openai:v0.34.0 /repo/verify/sidecar_vs_weights.py \
        --stock /stock --ablit /ablit --sidecar /sidecar.safetensors --threads 8 \
        --json /repo/results/sidecar-vs-weights.json --derive /repo/results/derived-steering.safetensors

    --names FILE   only these tensor names (one per line); skips the byte scan of both trees
    --layers 4,20  only these layers
    --no-fit       do not fit an operator to the bytes when the candidates fail
    --selftest     build synthetic checkpoints the way a real one is built and check every verdict

Verdict line, and exit status 0 only for the first two:

    REBUILT-AS-PUBLISHED   the published sidecar's operator, as the lane applies it, rebuilds the bytes
    REBUILT                another operator rebuilds them; --derive wrote it; load that file instead
    OPEN                   nothing tried rebuilds them, or tensors outside the two hook sites changed

Quantised weights are FP8 E4M3 payloads with block scales, value = payload x scale, the
scale tensor [ceil(N / bn), ceil(K / bk)] expanded in natural order, E8M0 scales read as
2^(code - 127): the layout vLLM v0.34.0 itself reads these tensors with
(vllm/models/deepseek_v41/amd/mono/weights942.py:33-41 for DeepSeek V4.1's dense MXFP8
linears, vllm/v1/attention/ops/rocm_aiter_mla_sparse.py:1684-1709 for the general case).
A scale tensor is looked for under <weight>_scale, <weight>_scale_inv and, for a name
ending in .weight, <name>.scale; the pairing used is printed per family.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys
import tempfile
import time

import torch

FLOATS = {"F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}
INTS = {"U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "I32": torch.int32,
        "I64": torch.int64, "BOOL": torch.bool}
MANTISSA_BITS = {"BF16": 7, "F16": 10, "F32": 23, "F64": 52}
LAYER = re.compile(r"(^|\.)layers\.(\d+)\.")
CHUNK = 64 << 20
REBUILT, CLOSE = 0.95, 0.80
SITES = ("attention", "shared expert")
SITE_PATTERNS = {"attention": r"attn|attention", "shared expert": r"shared_expert"}
SITE_KEYS = {"attention": "attn", "shared expert": "shared"}
# name, what the builder did, whether the lane can apply the float result exactly
VARIANTS = (
    ("plain", "x - a d (d.x)"),
    ("plain+gain", "the same, then the published per-row gain"),
    ("plain+rownorm", "the same, then each row restored to its stock norm"),
    ("rownormalised", "rows normalised, d removed from the unit rows, renormalised, row norms restored"),
    ("colnormalised", "columns normalised, d removed from the unit columns, renormalised, column norms restored"),
)
V2_FORMAT = "pp2-steering-v2"


def e4m3_table() -> torch.Tensor:
    values = []
    for byte in range(256):
        sign = -1.0 if byte >> 7 else 1.0
        exponent, mantissa = (byte >> 3) & 0xF, byte & 0x7
        if exponent == 0xF and mantissa == 0x7:
            values.append(float("nan"))
        elif exponent == 0:
            values.append(sign * (mantissa / 8.0) * 2.0**-6)
        else:
            values.append(sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
    return torch.tensor(values, dtype=torch.float32)


E4M3 = e4m3_table()


def e4m3_exponent(a: torch.Tensor) -> torch.Tensor:
    """Exponent of the binade |a| sits in, clamped to E4M3's range (subnormals share -6)."""
    _, ex = torch.frexp(a)
    return torch.where(a < 2.0**-6, torch.full_like(ex, -6), ex - 1).clamp(max=8)


def e4m3_round(y: torch.Tensor) -> torch.Tensor:
    """Nearest E4M3 value to y: ties to even, saturating at 448. Equal to PyTorch's
    float8_e4m3fn cast (checked in --selftest where that dtype exists)."""
    a = y.abs()
    step = torch.ldexp(torch.ones_like(a), e4m3_exponent(a) - 3)
    return torch.sign(y) * (torch.round(a / step) * step).clamp(max=448.0)


def float_round(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x.to(dtype).to(torch.float32)


# ------------------------------------------------------------------- reading checkpoints


def read_header(path: str) -> tuple[dict[str, dict], dict]:
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    base = 8 + length
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = {"file": path, "dtype": meta["dtype"], "shape": list(meta["shape"]),
                     "start": base + start, "end": base + end}
    return out, header.get("__metadata__") or {}


def index_tree(root: str) -> dict[str, dict]:
    files = sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))
    if not files:
        raise SystemExit(f"no .safetensors files in {root}")
    tensors: dict[str, dict] = {}
    for name in files:
        for key, meta in read_header(os.path.join(root, name))[0].items():
            if key in tensors:
                raise SystemExit(f"{key} appears in two shards of {root}")
            tensors[key] = meta
    return tensors


def same_bytes(a: dict, b: dict) -> bool:
    if a["dtype"] != b["dtype"] or a["shape"] != b["shape"]:
        return False
    remaining = a["end"] - a["start"]
    if remaining != b["end"] - b["start"]:
        return False
    with open(a["file"], "rb") as fa, open(b["file"], "rb") as fb:
        fa.seek(a["start"])
        fb.seek(b["start"])
        while remaining:
            step = min(CHUNK, remaining)
            if fa.read(step) != fb.read(step):
                return False
            remaining -= step
    return True


def raw(meta: dict) -> torch.Tensor:
    with open(meta["file"], "rb") as handle:
        handle.seek(meta["start"])
        data = bytearray(handle.read(meta["end"] - meta["start"]))
    dtype = FLOATS.get(meta["dtype"]) or INTS.get(meta["dtype"]) or torch.uint8
    if not data:
        return torch.zeros(meta["shape"], dtype=dtype)
    return torch.frombuffer(data, dtype=dtype).reshape(meta["shape"])


def scale_names(name: str) -> list[str]:
    """Where a weight's block scales may be stored, most specific first."""
    candidates = [name + "_scale", name + "_scale_inv"]
    if name.endswith(".weight"):
        candidates.append(name[: -len(".weight")] + ".scale")
    return candidates


def scale_of(name: str, tree: dict[str, dict]) -> str | None:
    return next((c for c in scale_names(name) if c in tree), None)


class Stored:
    """A tensor as numbers plus the grid it is stored on."""

    def __init__(self, values, kind, how, scale=None, dtype=None):
        self.values, self.kind, self.how, self.scale, self.dtype = values, kind, how, scale, dtype

    def round(self, x: torch.Tensor, rows=None) -> torch.Tensor:
        """x quantised onto this tensor's grid (rows: the row index x covers)."""
        if self.kind == "e4m3":
            scale = self.scale if rows is None else self.scale[rows]
            return e4m3_round(x / scale) * scale
        return float_round(x, self.dtype)

    def cells(self) -> tuple[torch.Tensor, torch.Tensor]:
        """[lo, hi]: the float values that round to each stored value."""
        v = self.values
        if self.kind == "e4m3":
            code = (v / self.scale).abs()
            exponent = e4m3_exponent(code)
            step = torch.ldexp(torch.ones_like(code), exponent - 3) * self.scale
            mantissa, _ = torch.frexp(code)
            inner = torch.where((mantissa == 0.5) & (code > 2.0**-6), step / 4, step / 2)
            outer = torch.where(code >= 448.0, torch.full_like(step, float("inf")), step / 2)
        else:
            bits = MANTISSA_BITS[next(k for k, t in FLOATS.items() if t == self.dtype)]
            mantissa, ex = torch.frexp(v.abs())
            step = torch.ldexp(torch.ones_like(v), ex - 1 - bits)
            inner = torch.where(mantissa == 0.5, step / 4, step / 2)
            outer = step / 2
        inner = torch.where(v == 0, outer, inner)
        lo = torch.where(v >= 0, v - inner, v - outer)
        hi = torch.where(v >= 0, v + outer, v + inner)
        return lo, hi


def expand_scale(factors: torch.Tensor, shape: list[int]) -> tuple[torch.Tensor | None, str]:
    """Block scales expanded to the weight's shape, the way vLLM expands them."""
    if factors.numel() == 1:
        return factors.reshape([1] * len(shape)).expand(shape), "one scale"
    if factors.dim() == 1 and factors.numel() == shape[0]:
        return factors.reshape([-1] + [1] * (len(shape) - 1)).expand(shape), "one scale per row"
    if factors.dim() != len(shape) or any(b > n for b, n in zip(factors.shape, shape)):
        return None, f"scale shape {list(factors.shape)} does not tile weight shape {shape}"
    blocks = [math.ceil(n / b) for n, b in zip(shape, factors.shape)]
    for axis, block in enumerate(blocks):
        if block > 1:
            factors = factors.repeat_interleave(block, dim=axis)
        factors = factors.narrow(axis, 0, shape[axis])
    return factors.contiguous(), "blocks of " + " x ".join(str(b) for b in blocks)


def read_stored(name: str, tree: dict[str, dict], payload: str) -> tuple[Stored | None, str]:
    weight = tree[name]
    dtype = weight["dtype"]
    scale_name = scale_of(name, tree)
    if dtype in FLOATS and scale_name is None:
        values = raw(weight).to(torch.float32)
        return Stored(values, "float", f"{dtype} as stored", dtype=FLOATS[dtype]), ""
    if not (dtype == "F8_E4M3" or (dtype in ("U8", "I8") and payload == "e4m3")):
        return None, f"weight dtype {dtype} is not read here"
    values = E4M3[raw(weight).reshape(-1).view(torch.uint8).long()].reshape(weight["shape"])
    how = "FP8 E4M3"
    if scale_name is None:
        return Stored(values, "e4m3", how + ", no scale tensor", scale=torch.ones_like(values)), ""
    scale = tree[scale_name]
    if scale["dtype"] in ("F8_E8M0", "U8"):
        factors = torch.pow(2.0, raw(scale).view(torch.uint8).to(torch.float32) - 127.0)
        how += f" x 2^(code-127) [{scale['dtype']}]"
    elif scale["dtype"] in FLOATS:
        factors = raw(scale).to(torch.float32)
        how += f" x {scale['dtype']} scale"
    else:
        return None, f"scale dtype {scale['dtype']} of {scale_name} is not read here"
    expanded, layout = expand_scale(factors, weight["shape"])
    if expanded is None:
        return None, f"{scale_name}: {layout}"
    return Stored(values * expanded, "e4m3", f"{how}, {layout}", scale=expanded), ""


def family(name: str) -> tuple[str, int | None]:
    match = LAYER.search(name)
    if not match:
        return name, None
    return name[: match.start(2)] + "{L}" + name[match.end(2):], int(match.group(2))


def site_of(fam: str, patterns: dict[str, str]) -> str | None:
    for site in ("shared expert", "attention"):
        if re.search(patterns[site], fam, re.IGNORECASE):
            return site
    return None


def load_sidecar(path: str) -> dict[int, dict]:
    """The published (v1) sidecar: layer -> alpha, direction (unit), attn_row_gain."""
    entries: dict[int, dict] = {}
    for name, meta in read_header(path)[0].items():
        match = re.fullmatch(r"layers\.(\d+)\.(alpha|direction|attn_row_gain)", name)
        if match:
            entries.setdefault(int(match.group(1)), {})[match.group(2)] = raw(meta).to(torch.float32)
    out = {}
    for layer, entry in entries.items():
        if len(entry) == 3:
            entry["direction"] = entry["direction"] / entry["direction"].norm()
            entry["alpha"] = float(entry["alpha"])
            out[layer] = entry
    return out


# ------------------------------------------------------------------- the requantised view


def top_left_singular(delta: torch.Tensor, iters: int = 12, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(delta.shape[0], generator=generator)
    for _ in range(iters):
        v = delta @ (delta.T @ v)
        v = v / v.norm().clamp_min(1e-30)
    return v


def requantised_view(stock: torch.Tensor, ablit: torch.Tensor, entry: dict) -> dict:
    """The two dequantised matrices compared directly. Biased by rounding (module docstring):
    kept because it is what a plain weight diff shows."""
    direction, alpha, gain = entry["direction"], entry["alpha"], entry["attn_row_gain"]
    delta = ablit - stock
    sd, ad = direction @ stock, direction @ ablit
    generator = torch.Generator().manual_seed(0)
    randoms = torch.randn(8, stock.shape[0], generator=generator)
    randoms = randoms / randoms.norm(dim=1, keepdim=True)
    control = (randoms @ ablit).norm(dim=1) / (randoms @ stock).norm(dim=1).clamp_min(1e-30)
    v = top_left_singular(delta)
    predicted = stock - alpha * torch.outer(direction, sd)
    row_change = ablit.norm(dim=1) / predicted.norm(dim=1).clamp_min(1e-30) - 1.0
    centred = torch.stack([row_change - row_change.mean(), (gain - 1.0) - (gain - 1.0).mean()])
    denominator = float(centred[0].norm() * centred[1].norm())
    return {
        "cos": abs(float(v @ direction)),
        "rho": float(ad.norm() / sd.norm().clamp_min(1e-30)),
        "control": float(control.mean()),
        "alpha_fit": float(((sd - ad) @ sd) / (sd @ sd).clamp_min(1e-30)),
        "alpha": alpha,
        "gain_r": float(centred[0] @ centred[1]) / denominator if denominator else 0.0,
        "changed": float((delta != 0).float().mean()),
    }


# ------------------------------------------------------------------- rebuild


class Site:
    """One writer matrix [hidden, in], stock and abliterated, and the candidate edits."""

    def __init__(self, stock: Stored, ablit: Stored, entry: dict) -> None:
        self.S, self.A, self.grid = stock.values, ablit.values, ablit
        self.d, self.alpha, self.gain = entry["direction"], entry["alpha"], entry["attn_row_gain"]
        S = self.S
        self.n_row = S.norm(dim=1).clamp_min(1e-30)
        self.n_col = S.norm(dim=0).clamp_min(1e-30)
        self.p = S.T @ self.d                       # what every row reads out along d
        self.q = S.T @ (self.d / self.n_row)        # the same on unit rows
        self.pc = self.p / self.n_col               # cosine of each column with d
        strong = torch.topk(self.d.abs(), min(384, S.shape[0])).indices
        spread = torch.arange(0, S.shape[0], max(1, S.shape[0] // 256))
        self.subset = torch.unique(torch.cat([strong, spread]))

    def build(self, variant: str, alpha: float, rows: torch.Tensor) -> torch.Tensor:
        """Rows of the float matrix the builder would have produced."""
        S, d = self.S[rows], self.d[rows].unsqueeze(1)
        if variant == "rownormalised":
            unit = S / self.n_row[rows].unsqueeze(1) - alpha * d * self.q.unsqueeze(0)
            return unit * (self.n_row[rows] / unit.norm(dim=1).clamp_min(1e-30)).unsqueeze(1)
        x = S - alpha * d * self.p.unsqueeze(0)
        if variant == "plain+gain":
            x = x * self.gain[rows].unsqueeze(1)
        elif variant == "plain+rownorm":
            x = x * (self.n_row[rows] / x.norm(dim=1).clamp_min(1e-30)).unsqueeze(1)
        elif variant == "colnormalised":
            x = x / torch.sqrt((1.0 + (alpha * alpha - 2.0 * alpha) * self.pc**2).clamp_min(1e-12)).unsqueeze(0)
        elif variant != "plain":
            raise ValueError(variant)
        return x

    def operator(self, variant: str, alpha: float) -> dict:
        """The same float edit as an output-side operator: x' = (x + write (read.x)) * row_gain."""
        d = self.d
        if variant == "rownormalised":
            unit_norm = torch.cat([
                (self.S[rows] / self.n_row[rows].unsqueeze(1) - alpha * d[rows].unsqueeze(1) * self.q).norm(dim=1)
                for rows in torch.arange(self.S.shape[0]).split(1024)
            ])
            return {"read": d / self.n_row, "write": -alpha * self.n_row * d, "row_gain": 1.0 / unit_norm.clamp_min(1e-30)}
        out = {"read": d.clone(), "write": -alpha * d}
        if variant == "plain+gain":
            out["row_gain"] = self.gain.clone()
        elif variant == "plain+rownorm":
            norm = torch.cat([self.build("plain", alpha, rows).norm(dim=1)
                              for rows in torch.arange(self.S.shape[0]).split(1024)])
            out["row_gain"] = self.n_row / norm.clamp_min(1e-30)
        return out

    def apply(self, op: dict, rows: torch.Tensor) -> torch.Tensor:
        """Rows of S edited by an output-side operator (what the lane computes at scale 1)."""
        x = self.S[rows] + op["write"][rows].unsqueeze(1) * (self.S.T @ op["read"]).unsqueeze(0)
        if "row_gain" in op:
            x = x * op["row_gain"][rows].unsqueeze(1)
        return x

    def agreement(self, make, rows: torch.Tensor | None = None) -> tuple[float, float]:
        """J: of the values the candidate or the checkpoint changed, the share that agree.
        Second value: the share of values the checkpoint changed."""
        agree = touched = changed = total = 0
        chunks = [rows] if rows is not None else list(torch.arange(self.S.shape[0]).split(512))
        for chunk in chunks:
            S, A = self.S[chunk], self.A[chunk]
            pred = self.grid.round(make(chunk), chunk)
            either = (A != S) | (pred != S)
            agree += int(((pred == A) & either).sum())
            touched += int(either.sum())
            changed += int((A != S).sum())
            total += A.numel()
        return (agree / touched if touched else 1.0), changed / max(total, 1)

    RATIOS = [0.60 + 0.05 * i for i in range(17)]     # strengths tried, relative to the published alpha

    def score(self, variant: str, alpha: float) -> float:
        return self.agreement(lambda rows: self.build(variant, alpha, rows), self.subset)[0]

    def scan(self) -> dict[str, tuple[float, float, bool]]:
        """Per rule, the strength that rebuilds best on a row subset:
        (alpha, J, the best strength sits at the edge of the range tried)."""
        coarse = {name: [self.score(name, self.alpha * r) for r in self.RATIOS] for name, _ in VARIANTS}
        top = max(max(values) for values in coarse.values())
        out = {}
        for name, values in coarse.items():
            best = max(range(len(values)), key=values.__getitem__)
            edge = best in (0, len(values) - 1)
            if values[best] < top - 0.15:               # not in contention: no need to refine
                out[name] = (self.alpha * self.RATIOS[best], values[best], edge)
                continue
            lo = self.alpha * self.RATIOS[max(best - 1, 0)]
            hi = self.alpha * self.RATIOS[min(best + 1, len(values) - 1)]
            for _ in range(9):
                a, b = lo + (hi - lo) / 3, hi - (hi - lo) / 3
                if self.score(name, a) < self.score(name, b):
                    lo = a
                else:
                    hi = b
            alpha = (lo + hi) / 2
            out[name] = (alpha, self.score(name, alpha), edge)
        return out

    def retained(self, make) -> tuple[float, float]:
        """(share of the float edit the checkpoint carries, size of what it lost or added,
        both relative to the float edit)."""
        dot = energy = miss = 0.0
        for chunk in torch.arange(self.S.shape[0]).split(512):
            edit = (make(chunk) - self.S[chunk]).double()
            actual = (self.A[chunk] - self.S[chunk]).double()
            dot += float((edit * actual).sum())
            energy += float((edit * edit).sum())
            miss += float(((actual - edit) ** 2).sum())
        return dot / max(energy, 1e-300), math.sqrt(miss / max(energy, 1e-300))

    def distance(self, make_a, make_b) -> float:
        """|edit a - edit b| / |edit b|, Frobenius."""
        diff = base = 0.0
        for chunk in torch.arange(self.S.shape[0]).split(512):
            a, b = make_a(chunk).double(), make_b(chunk).double()
            diff += float(((a - b) ** 2).sum())
            base += float(((b - self.S[chunk].double()) ** 2).sum())
        return math.sqrt(diff / max(base, 1e-300))

    # ---- an operator fitted to the bytes, for when no candidate rebuilds them ---------

    def fit(self, iters: int) -> tuple[dict, dict]:
        """A rank-one output-side operator consistent with the rounding cell of every value:
        alternating projections between "rank one" and "inside the cells"."""
        lo, hi = self.grid.cells()
        lo, hi = lo - self.S, hi - self.S                 # cells for the edit itself
        cols = torch.arange(0, self.S.shape[1], max(1, self.S.shape[1] // 2048))
        lo_c, hi_c = lo[:, cols], hi[:, cols]
        Z = (self.A - self.S)[:, cols]
        u = top_left_singular(Z)
        outside = 1.0
        for step in range(iters):
            m = Z.T @ u
            model = torch.outer(u, m)
            if step % 25 == 24:
                now = float(((model < lo_c) | (model > hi_c)).float().mean())
                if outside - now < 2e-5 * max(1.0, step / 100):
                    break
                outside = now
            Z = torch.minimum(torch.maximum(model, lo_c), hi_c)
            u = Z @ (Z.T @ u)
            u = u / u.norm().clamp_min(1e-30)
        m = (self.A - self.S).T @ u                       # every column's read-out, u fixed
        for _ in range(40):
            m = torch.minimum(torch.maximum(torch.outer(u, m), lo), hi).T @ u
        S64 = self.S.double()
        if S64.shape[1] <= S64.shape[0]:
            gram = S64.T @ S64
            ridge = 1e-10 * float(gram.diagonal().mean())
            v = S64 @ torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype), m.double())
        else:
            gram = S64 @ S64.T
            ridge = 1e-10 * float(gram.diagonal().mean())
            v = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], dtype=gram.dtype), S64 @ m.double())
        residual = float((S64.T @ v - m.double()).norm() / m.double().norm().clamp_min(1e-300))
        op = {"read": v.float(), "write": u.clone()}
        unit = lambda x: x / x.norm().clamp_min(1e-30)  # noqa: E731
        info = {
            "outside_cells": outside, "read_fit": residual, "iterations": step + 1,
            "write_cos_d": abs(float(unit(u) @ self.d)),
            "write_cos_rowweighted_d": abs(float(unit(u) @ unit(self.n_row * self.d))),
            "read_cos_d": abs(float(unit(v.float()) @ self.d)),
            "read_cos_rowweighted_d": abs(float(unit(v.float()) @ unit(self.d / self.n_row))),
        }
        return op, info


def rebuild_site(site: Site, shipped: str, fit_iters: int, force_fit: bool) -> dict:
    """Everything the rebuild says about one matrix."""
    out: dict = {"alpha_published": site.alpha}
    as_shipped = lambda rows: site.build(shipped, site.alpha, rows)  # noqa: E731
    out["shipped_variant"] = shipped
    out["shipped_J"], out["changed"] = site.agreement(as_shipped)
    scans = site.scan()
    out["scan"] = {name: {"alpha": a, "J_subset": j, "edge": e} for name, (a, j, e) in scans.items()}
    top = max(j for _, j, _ in scans.values())
    # Variants the bytes cannot tell apart are settled in the listed order: simplest first.
    name = next(n for n, _ in VARIANTS if scans[n][1] >= top - 0.004)
    alpha = scans[name][0]
    best = lambda rows: site.build(name, alpha, rows)  # noqa: E731
    out["best_variant"], out["best_alpha"], out["best_edge"] = name, alpha, scans[name][2]
    out["best_J"], _ = site.agreement(best)
    out["tied"] = [n for n, _ in VARIANTS if n != name and scans[n][1] >= top - 0.004]
    op = site.operator(name, alpha)
    applied = lambda rows: site.apply(op, rows)  # noqa: E731
    out["operator_J"], _ = site.agreement(applied)
    out["operator_exact"] = name != "colnormalised"
    out["retained"], out["lost"] = site.retained(best)
    out["shipped_vs_best"] = site.distance(as_shipped, best)
    if name == "colnormalised":
        gamma = 1.0 / torch.sqrt((1.0 + (alpha * alpha - 2.0 * alpha) * site.pc**2).clamp_min(1e-12))
        out["column_gain"] = {"min": float(gamma.min()), "share_over_1pct": float(((gamma - 1).abs() > 0.01).float().mean()),
                              "missing": site.distance(applied, best)}
    out["operator"] = op
    out["source"] = f"published direction, {name}, alpha {alpha:.4f}"
    if force_fit or (out["best_J"] < REBUILT and fit_iters > 0):
        fitted, info = site.fit(fit_iters)
        out["fit"] = info
        out["fit_J"], _ = site.agreement(lambda rows: site.apply(fitted, rows))
        if out["fit_J"] > max(out["best_J"], out["operator_J"]) or force_fit:
            out["operator"], out["operator_J"], out["operator_exact"] = fitted, out["fit_J"], True
            out["source"] = "fitted to the bytes"
            out["retained"], out["lost"] = site.retained(lambda rows: site.apply(fitted, rows))
            out["shipped_vs_best"] = site.distance(as_shipped, lambda rows: site.apply(fitted, rows))
    return out


# ------------------------------------------------------------------- writing a v2 sidecar


def write_safetensors(path: str, tensors: dict[str, tuple[str, list[int], bytes]],
                      metadata: dict[str, str] | None = None) -> None:
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    offset, blobs = 0, []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def pack(tensor: torch.Tensor, dtype: str) -> tuple[str, list[int], bytes]:
    tensor = tensor.contiguous()
    if dtype == "BF16":
        return dtype, list(tensor.shape), tensor.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    if dtype == "F32":
        return dtype, list(tensor.shape), tensor.to(torch.float32).numpy().tobytes()
    return dtype, list(tensor.shape), tensor.numpy().tobytes()


def write_v2(path: str, operators: dict[tuple[int, str], dict], notes: dict[str, str]) -> None:
    tensors = {}
    for (layer, site), op in sorted(operators.items()):
        prefix = f"layers.{layer}.{SITE_KEYS[site]}"
        tensors[f"{prefix}.read"] = pack(op["read"], "F32")
        tensors[f"{prefix}.write"] = pack(op["write"], "F32")
        if "row_gain" in op:
            tensors[f"{prefix}.row_gain"] = pack(op["row_gain"], "F32")
    write_safetensors(path, tensors, {"format": V2_FORMAT, **notes})


# ------------------------------------------------------------------- the analysis


def analyse(stock_root: str, ablit_root: str, sidecar_path: str, names: list[str] | None = None,
            payload: str = "e4m3", quiet: bool = False, patterns: dict[str, str] | None = None,
            layers: set[int] | None = None, fit_iters: int = 400, force_fit: bool = False,
            derive: str | None = None) -> dict:
    patterns = patterns or SITE_PATTERNS

    def say(text: str = "") -> None:
        if not quiet:
            print(text, flush=True)

    started = time.time()
    stock, ablit = index_tree(stock_root), index_tree(ablit_root)
    sidecar = load_sidecar(sidecar_path)
    only = sorted(set(stock) ^ set(ablit))
    if names is None:
        shared = sorted(set(stock) & set(ablit))
        total = sum(stock[n]["end"] - stock[n]["start"] for n in shared)
        say(f"comparing {len(shared)} tensors ({total / 2**30:.1f} GiB per tree) byte for byte ...")
        names = [n for n in shared if not same_bytes(stock[n], ablit[n])]
    else:
        names = [n for n in names if n in stock and n in ablit]
    changed = set(names)
    say(f"{len(names)} tensors differ; the sidecar covers layers {min(sidecar)}-{max(sidecar)} ({len(sidecar)} layers)")

    # A weight and its scale tensor are one matrix: group them under the weight.
    scales = {}
    for name in sorted(set(stock) & set(ablit)):
        paired = scale_of(name, stock)
        if paired is not None and (name in changed or paired in changed):
            scales[paired] = name
    matrices = sorted((changed - set(scales)) | set(scales.values()))
    families: dict[str, dict] = {}
    for name in matrices:
        fam, layer = family(name)
        entry = families.setdefault(fam, {"family": fam, "site": site_of(fam, patterns), "layers": {}, "tensors": 0})
        entry["layers"][layer] = name
        entry["tensors"] += (name in changed) + (scale_of(name, stock) in changed)
    by_site = {site: [f for f in families.values() if f["site"] == site] for site in SITES}

    problems: list[str] = []
    operators: dict[tuple[int, str], dict] = {}
    rows_out = []
    say()
    say("requantised view = the two dequantised matrices compared directly (weaker and bent by rounding);")
    say("J = share of changed values a float candidate reproduces on the checkpoint's own grid")
    for fam in sorted(families.values(), key=lambda f: (f["site"] is None, f["family"])):
        site_name = fam["site"]
        role = f"the {site_name} writer" if site_name else "NOT at a steering hook"
        say()
        say(f"{fam['family']}  ({fam['tensors']} tensors in {len(fam['layers'])} layers; {role})")
        if site_name is None:
            problems.append(f"{fam['family']}: {len(fam['layers'])} rewritten matrices are not at a steering hook, "
                            "so the steered lane does not reproduce them")
            continue
        if len(by_site[site_name]) != 1:
            problems.append(f"{len(by_site[site_name])} rewritten families match the {site_name} site by name; "
                            "pass --attention-pattern / --shared-pattern so exactly one does")
            continue
        shipped = "plain+gain" if site_name == "attention" else "plain"
        say(f"  {'layer':>5}  {'changed':>7}  | {'cos':>5} {'alpha~':>6} of {'alpha':>6} | "
            f"{'as shipped':>10} | {'best rule':<14} {'alpha*':>7} {'J':>6} | {'operator':>8} {'carries':>7}  note")
        covered = set()
        for layer, name in sorted(fam["layers"].items(), key=lambda kv: (kv[0] is None, kv[0])):
            row = {"family": fam["family"], "site": site_name, "layer": layer, "name": name}
            rows_out.append(row)
            if layer is None or layer not in sidecar:
                row["skip"] = "layer not in the sidecar"
                problems.append(f"{name}: rewritten, but the sidecar has no layer {layer}")
                say(f"  {str(layer):>5}  rewritten, and the sidecar has no such layer")
                continue
            if layers is not None and layer not in layers:
                continue
            s, why_s = read_stored(name, stock, payload)
            a, why_a = read_stored(name, ablit, payload)
            if s is None or a is None:
                row["skip"] = why_s or why_a
                problems.append(f"{name}: {row['skip']}")
                say(f"  {layer:>5}  not read: {row['skip']}")
                continue
            hidden = sidecar[layer]["direction"].shape[0]
            if s.values.dim() != 2 or s.values.shape[0] != hidden or s.values.shape != a.values.shape:
                row["skip"] = f"shape {list(s.values.shape)} is not [hidden={hidden}, in]"
                problems.append(f"{name}: {row['skip']}")
                say(f"  {layer:>5}  {row['skip']}")
                continue
            row["read_as"], row["scale"] = a.how, scale_of(name, ablit)
            row["view"] = requantised_view(s.values, a.values, sidecar[layer])
            site = Site(s, a, sidecar[layer])
            result = rebuild_site(site, shipped, fit_iters, force_fit)
            operators[(layer, site_name)] = result.pop("operator")
            row.update(result)
            covered.add(layer)
            notes = []
            if result["tied"]:
                notes.append("= " + ", ".join(result["tied"]))
            if result["best_edge"]:
                notes.append("alpha* at the edge of the scan")
            if "fit_J" in result:
                notes.append(f"fitted to the bytes J {result['fit_J']:.3f}")
            if "column_gain" in result:
                notes.append(f"column gain not applicable by the lane: {result['column_gain']['missing']:.3f} of the edit")
            view = row["view"]
            say(f"  {layer:>5}  {result['changed']:>7.3f}  | {view['cos']:>5.3f} {view['alpha_fit']:>6.3f} of {view['alpha']:>6.3f} | "
                f"{result['shipped_J']:>10.3f} | {result['best_variant']:<14} {result['best_alpha']:>7.4f} {result['best_J']:>6.3f} | "
                f"{result['operator_J']:>8.3f} {result['retained']:>7.3f}  {'; '.join(notes)}")
        done = [r for r in rows_out if r["family"] == fam["family"] and "best_J" in r]
        fam["read_as"] = done[0]["read_as"] if done else None
        if done:
            say(f"  read as: {done[0]['read_as']} (scale tensor {done[0]['scale']})")
        missing = sorted(set(sidecar) - set(fam["layers"]) if layers is None else set())
        if missing:
            problems.append(f"{fam['family']}: sidecar layers {missing} were not rewritten in the checkpoint")
    for site_name in SITES:
        if not by_site[site_name]:
            problems.append(f"no rewritten family matches the {site_name} site by name")
    if only:
        problems.append(f"{len(only)} tensor names exist in only one tree, e.g. {only[:3]}")

    done = [r for r in rows_out if "best_J" in r]
    summary: dict = {}
    for site_name in SITES:
        rows = [r for r in done if r["site"] == site_name]
        if not rows:
            continue
        median = lambda key: sorted(r[key] for r in rows)[len(rows) // 2]  # noqa: E731
        rules = sorted({r["best_variant"] for r in rows if "fit_J" not in r or r["source"] != "fitted to the bytes"})
        summary[site_name] = {
            "layers": len(rows),
            "shipped_J_min": min(r["shipped_J"] for r in rows), "shipped_J_median": median("shipped_J"),
            "operator_J_min": min(r["operator_J"] for r in rows), "operator_J_median": median("operator_J"),
            "best_J_min": min(r["best_J"] for r in rows),
            "rules": rules, "sources": sorted({r["source"].split(", alpha")[0] for r in rows}),
            "alpha_star_min": min(r["best_alpha"] for r in rows), "alpha_star_max": max(r["best_alpha"] for r in rows),
            "alpha_star_median": median("best_alpha"), "alpha_published_median": median("alpha_published"),
            "retained_median": median("retained"), "lost_median": median("lost"),
            "shipped_vs_best_median": median("shipped_vs_best"),
            "inexact": sum(not r["operator_exact"] for r in rows),
            "view_median": {key: sorted(r["view"][key] for r in rows)[len(rows) // 2]
                            for key in ("cos", "rho", "control", "alpha_fit", "alpha", "gain_r")},
        }
    say()
    for site_name, s in summary.items():
        say(f"{site_name} writer, {s['layers']} layers:")
        v = s["view_median"]
        say(f"  requantised view, medians (what revision 2 of this script reported): cos {v['cos']:.3f}, rho {v['rho']:.3f} "
            f"(control {v['control']:.2f}), alpha~ {v['alpha_fit']:.3f} of {v['alpha']:.3f}, gain r {v['gain_r']:.2f}")
        say(f"  the published operator as the lane applies it: J median {s['shipped_J_median']:.3f}, min {s['shipped_J_min']:.3f}")
        say(f"  best rule with the published direction: {', '.join(s['rules']) or '-'}; alpha* {s['alpha_star_median']:.4f} "
            f"(range {s['alpha_star_min']:.4f} to {s['alpha_star_max']:.4f}; published median {s['alpha_published_median']:.4f}); "
            f"J min {s['best_J_min']:.3f}")
        say(f"  the operator the lane would apply ({'; '.join(s['sources'])}): J median {s['operator_J_median']:.3f}, "
            f"min {s['operator_J_min']:.3f}")
        say(f"  the published operator's edit differs from that one by {s['shipped_vs_best_median']:.2f} of its size (median)")
        say(f"  the checkpoint carries {s['retained_median']:.2f} of that float edit; what rounding dropped or added is "
            f"{s['lost_median']:.2f} of its size")
        if s["inexact"]:
            say(f"  in {s['inexact']} layers the best rule has a per-column gain the lane cannot apply (wo_b's input is a "
                "quantised activation on the decode path); the operator written is the projection without it")

    if not done and not problems:
        problems.append("nothing was compared")
    floor_shipped = min((r["shipped_J"] for r in done), default=0.0)
    floor_operator = min((r["operator_J"] for r in done), default=0.0)
    if any(not r["operator_exact"] for r in done):
        problems.append("a per-column gain is part of the build and the lane cannot apply it")
    if not problems and floor_shipped >= REBUILT:
        verdict = "REBUILT-AS-PUBLISHED"
        detail = (f"the published sidecar's operator rebuilds every rewritten matrix (J >= {floor_shipped:.3f}); "
                  "at scale 1 the lane applies the float edit this checkpoint was built with")
    elif not problems and floor_operator >= REBUILT:
        verdict = "REBUILT"
        detail = (f"the operators in the table rebuild every rewritten matrix (J >= {floor_operator:.3f}); the published "
                  f"sidecar as shipped does not (J >= {floor_shipped:.3f}). Load the derived sidecar, not the published one")
    else:
        verdict = "OPEN"
        if done and floor_operator < REBUILT:
            band = ("close: the rule and strength are identified; what is left is a small gain or the builder's "
                    "arithmetic" if floor_operator >= CLOSE else "not rebuilt")
            problems.append(f"lowest J of the best operator found is {floor_operator:.3f} ({band})")
        detail = "; ".join(problems)
    if derive and operators:
        notes = {"verdict": verdict, "stock": os.path.abspath(stock_root), "abliterated": os.path.abspath(ablit_root),
                 "published_sidecar": os.path.abspath(sidecar_path),
                 "sites": json.dumps({f"{r['layer']}.{SITE_KEYS[r['site']]}": [r["source"], round(r["operator_J"], 4)] for r in done})}
        write_v2(derive, operators, notes)
        say()
        say(f"wrote {derive}: {len(operators)} operators (read, write, optional row gain per layer and site), "
            f"format {V2_FORMAT}" + ("" if verdict.startswith("REBUILT") else "  [verdict is OPEN: do not serve from it]"))
    say()
    say(f"SIDECAR-VS-WEIGHTS {verdict}: {detail}  [{time.time() - started:.0f} s]")
    return {"verdict": verdict, "detail": detail, "problems": problems, "differing": len(names), "summary": summary,
            "families": [{k: v for k, v in f.items() if k != "layers"} | {"layers": sorted(k for k in f["layers"] if k is not None)}
                         for f in families.values()],
            "matrices": rows_out, "only_in_one_tree": only, "sidecar_layers": sorted(sidecar), "operators": operators}


# ---------------------------------------------------------------------------- self-test


def quantise_mx(matrix: torch.Tensor, block: tuple[int, int] = (32, 32)) -> tuple[torch.Tensor, torch.Tensor]:
    """A matrix quantised the way a builder would: (E4M3 bytes, E8M0 codes), block maxima
    scaled into the top binade."""
    rows, cols = matrix.shape
    b0, b1 = block
    peak = matrix.abs().reshape(rows // b0, b0, cols // b1, b1).amax(dim=(1, 3)).clamp_min(1e-30)
    exponent = torch.ceil(torch.log2(peak / 448.0))
    scale = torch.pow(2.0, exponent).repeat_interleave(b0, 0).repeat_interleave(b1, 1)
    values = e4m3_round(matrix / scale)
    table = torch.nan_to_num(E4M3, nan=float("inf"))
    order = torch.argsort(table)
    codes = order[torch.searchsorted(table[order], values.reshape(-1))].to(torch.uint8).reshape(rows, cols)
    codes = torch.where(values == 0, torch.zeros_like(codes), codes)   # +0, never -0
    return codes, (exponent + 127).to(torch.uint8)


def synthetic(work: str, tag: str, build, sidecar_of=None, hidden: int = 512, layers=(4, 5, 6),
              in_attn: int = 384, in_shared: int = 256, scale_suffix: str = ".scale",
              scale_dtype: str = "F8_E8M0", extra=None, block: tuple[int, int] = (32, 32),
              along: float = 1.0) -> dict:
    """A stock tree, an abliterated tree built ON THE STOCK GRID (dequantise, edit in float,
    quantise again: how a real one is made) and a published-style sidecar.
    build(S, d, site, layer) -> float matrix. `along`: how strongly the writers write along
    the direction, relative to a random direction (small = most edits fall below rounding)."""
    stock_dir, ablit_dir = os.path.join(work, tag, "stock"), os.path.join(work, tag, "ablit")
    os.makedirs(stock_dir)
    os.makedirs(ablit_dir)
    stock: dict = {}
    ablit: dict = {}
    sidecar: dict = {}
    truth: dict = {}
    for layer in layers:
        d = torch.randn(hidden)
        d = d / d.norm()
        rowscale = torch.exp(0.25 * torch.randn(hidden))        # rows of different norm
        for site, n_in in (("attn.wo_b", in_attn), ("ffn.shared_experts.w2", in_shared)):
            name = f"layers.{layer}.{site}.weight"
            master = torch.randn(hidden, n_in)
            master = master - (1.0 - along) * torch.outer(d, d @ master)
            master = rowscale.unsqueeze(1) * master
            aligned = torch.randperm(n_in)[:12]                 # a few columns that point along d
            master[:, aligned] += 6.0 * along * d.unsqueeze(1) * torch.rand(12)
            codes, exps = quantise_mx(master, block)
            S = E4M3[codes.long()] * torch.pow(2.0, exps.float() - 127).repeat_interleave(block[0], 0).repeat_interleave(block[1], 1)
            X = build(S, d, site, layer)
            truth[(layer, site)] = (S, X, d)
            acodes, aexps = quantise_mx(X, block)
            scale_key = name[: -len(".weight")] + ".scale" if scale_suffix == ".scale" else name + scale_suffix
            for tree, c, e in ((stock, codes, exps), (ablit, acodes, aexps)):
                tree[name] = pack(c, "F8_E4M3")
                tree[scale_key] = pack(e, scale_dtype)
        same = pack(torch.randn(hidden, 16).to(torch.bfloat16), "BF16")
        stock[f"layers.{layer}.attn.norm.weight"] = ablit[f"layers.{layer}.attn.norm.weight"] = same
        if extra:
            extra(stock, ablit, layer, d)
        entry = sidecar_of(truth, layer, d) if sidecar_of else {"alpha": 3.0, "gain": torch.ones(hidden)}
        sidecar[f"layers.{layer}.alpha"] = pack(torch.tensor(float(entry["alpha"])), "F32")
        sidecar[f"layers.{layer}.direction"] = pack(entry.get("direction", d), "BF16")
        sidecar[f"layers.{layer}.attn_row_gain"] = pack(entry["gain"], "BF16")
    for index in (0, 1):   # two shards per tree
        pick = lambda tree: {n: v for i, (n, v) in enumerate(sorted(tree.items())) if i % 2 == index}  # noqa: E731
        write_safetensors(os.path.join(stock_dir, f"model-{index:05d}-of-00002.safetensors"), pick(stock))
        write_safetensors(os.path.join(ablit_dir, f"model-{index:05d}-of-00002.safetensors"), pick(ablit))
    sidecar_path = os.path.join(work, tag, "sidecar.safetensors")
    write_safetensors(sidecar_path, sidecar)
    return {"stock": stock_dir, "ablit": ablit_dir, "sidecar": sidecar_path, "truth": truth,
            "derived": os.path.join(work, tag, "derived.safetensors")}


def row_normalised_build(S: torch.Tensor, d: torch.Tensor, alpha: float) -> torch.Tensor:
    n = S.norm(dim=1, keepdim=True)
    unit = S / n
    unit = unit - alpha * torch.outer(d, d @ unit)
    return n * unit / unit.norm(dim=1, keepdim=True)


def symmetric_sidecar(truth: dict, layer: int, d: torch.Tensor) -> dict:
    """What a symmetric sidecar made from a row-normalised build at 3.0 looks like: the
    strength of the real operator along d, and its row gain."""
    S, _, _ = truth[(layer, "attn.wo_b")]
    n = S.norm(dim=1)
    unit = S / n.unsqueeze(1) - 3.0 * torch.outer(d, (d / n) @ S)
    return {"alpha": 3.0 * float((n * d) @ d) * float((d / n) @ d), "gain": 1.0 / unit.norm(dim=1)}


def selftest() -> int:
    torch.manual_seed(11)
    hidden = 512
    work = tempfile.mkdtemp(prefix="sidecar-vs-weights-")
    problems: list[str] = []

    if hasattr(torch, "float8_e4m3fn"):
        y = (torch.rand(400_000) * 2 - 1) * torch.exp(torch.randn(400_000) * 3)
        y = torch.cat([y, torch.tensor([0.0, 448.0, 470.0, -500.0, 2.0**-6, 2.0**-9, 2.0**-10, 3e-4])])
        if not torch.equal(e4m3_round(y), y.to(torch.float8_e4m3fn).float()):
            problems.append("e4m3_round differs from torch's float8_e4m3fn cast")
        decoded = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        if not torch.equal(torch.nan_to_num(decoded, nan=9e9), torch.nan_to_num(E4M3, nan=9e9)):
            problems.append("the E4M3 table differs from torch's")

    def scenario(tag: str, build, sidecar_of=None, fit_iters: int = 400, force_fit: bool = False, **trees) -> dict:
        made = synthetic(work, tag, build, sidecar_of, **trees)
        result = analyse(made["stock"], made["ablit"], made["sidecar"], quiet=True, derive=made["derived"],
                         fit_iters=fit_iters, force_fit=force_fit)
        result.update(truth=made["truth"], derived=made["derived"])
        return result

    def col_normalised(S, d, alpha):
        n = S.norm(dim=0, keepdim=True)
        unit = S / n
        unit = unit - alpha * torch.outer(d, d @ unit)
        return n * unit / unit.norm(dim=0, keepdim=True)

    def check(tag: str, result: dict, verdict: str, rule: str | None = None, alpha: float | None = None,
              shipped_low: bool | None = None, exact: bool = True) -> None:
        rows = [r for r in result["matrices"] if "best_J" in r]
        line = (f"{tag:<34} {result['verdict']:<21} " + (
            f"shipped J {min(r['shipped_J'] for r in rows):.3f}  best {rows[0]['best_variant']} "
            f"alpha* {rows[0]['best_alpha']:.3f}  operator J {min(r['operator_J'] for r in rows):.3f}  "
            f"carries {rows[0]['retained']:.2f}  view cos {rows[0]['view']['cos']:.3f} "
            f"alpha~ {rows[0]['view']['alpha_fit']:.3f}/{rows[0]['view']['alpha']:.3f}" if rows else ""))
        bad = []
        if result["verdict"] != verdict:
            bad.append(f"verdict {result['verdict']} (expected {verdict}): {result['detail'][:200]}")
        if rule and any(r["best_variant"] != rule for r in rows):
            bad.append(f"rules {sorted({r['best_variant'] for r in rows})} (expected {rule})")
        if alpha and any(abs(r["best_alpha"] - alpha) > 0.01 * alpha for r in rows):
            bad.append(f"alpha* {[round(r['best_alpha'], 3) for r in rows]} (expected {alpha})")
        if shipped_low is not None and any((r["shipped_J"] < CLOSE) != shipped_low for r in rows):
            bad.append(f"shipped J {[round(r['shipped_J'], 3) for r in rows]}")
        if verdict.startswith("REBUILT") and exact:
            # The operator written to the derived sidecar must be the float build itself.
            written, _ = read_header(result["derived"])
            for (layer, site), (S, X, _) in result["truth"].items():
                key = f"layers.{layer}.{'attn' if site.startswith('attn') else 'shared'}"
                read, write = raw(written[key + ".read"]), raw(written[key + ".write"])
                applied = S + torch.outer(write, read @ S)
                if key + ".row_gain" in written:
                    applied = applied * raw(written[key + ".row_gain"]).unsqueeze(1)
                error = float((applied - X).norm() / (X - S).norm())
                if error > 0.02:
                    bad.append(f"derived operator for {key} is {error:.3f} of the edit away from the float build")
        print(("ok    " if not bad else "FAIL  ") + line)
        problems.extend(f"{tag}: {b}" for b in bad)

    plain = lambda a: (lambda S, d, site, layer: S - a * torch.outer(d, d @ S))  # noqa: E731

    # 1. The published operator exactly (projection; the attention gain is 1 to BF16 precision).
    check("published operator, as shipped", scenario("asis", plain(3.0)), "REBUILT-AS-PUBLISHED", "plain", 3.0, False)
    # 1b. The same, where the writers write little along d: most edits fall below the rounding
    #     step. The dequantised view reads a weaker, bent edit; the bytes still rebuild.
    r = scenario("deadzone", plain(3.0), along=0.06)
    check("published operator, weak writers", r, "REBUILT-AS-PUBLISHED", "plain", 3.0, False)
    rows = [x for x in r["matrices"] if "view" in x]
    middle = lambda values: sorted(values)[len(values) // 2]  # noqa: E731
    if not (middle([x["view"]["alpha_fit"] / x["view"]["alpha"] for x in rows]) < 0.95
            and middle([x["retained"] for x in rows]) < 0.95):
        problems.append("deadzone: expected the requantised view and the carried share to read below 0.95: "
                        + str([(round(x["view"]["alpha_fit"], 2), round(x["retained"], 2)) for x in rows]))
    # 2. Same direction, the build used another strength than the sidecar says.
    check("same direction, alpha 3.0 vs 3.2", scenario("alpha", plain(3.0), lambda t, l, d: {"alpha": 3.2, "gain": torch.ones(hidden)}),
          "REBUILT", "plain", 3.0)
    # 3. Row-normalised build at 3.0; the sidecar is the symmetric approximation of it.
    r = scenario("rownorm", lambda S, d, site, layer: row_normalised_build(S, d, 3.0), symmetric_sidecar)
    check("row-normalised build, symmetric sidecar", r, "REBUILT", "rownormalised", 3.0, True)
    view = [x["view"] for x in r["matrices"] if "view" in x]
    if not all(v["cos"] < 0.99 and v["alpha_fit"] < 0.95 * v["alpha"] for v in view):
        problems.append("rownorm: the requantised view should read weaker and bent here")
    # 4. Column-normalised build: identified, and flagged as not fully applicable.
    check("column-normalised build", scenario("colnorm", lambda S, d, site, layer: col_normalised(S, d, 3.0)),
          "OPEN", "colnormalised", 3.0, exact=False)
    # 5. A direction 15 degrees from the sidecar's: only an operator fitted to the bytes rebuilds it.
    def rotated(truth, layer, d):
        e = torch.randn(hidden)
        e = e - (e @ d) * d
        return {"alpha": 3.0, "gain": torch.ones(hidden), "direction": 0.966 * d + 0.2588 * e / e.norm()}
    check("other direction, fitted to the bytes", scenario("fit", plain(3.0), rotated, fit_iters=1500), "REBUILT", None, None, True)
    # 6. An oblique build with weights unrelated to the matrix: fitted as well.
    metric = {}
    def oblique(S, d, site, layer):
        D = metric.setdefault(layer, torch.exp(0.3 * torch.randn(hidden)))
        return S - 3.0 * torch.outer(D * d, (d / D) @ S)
    check("oblique build, fitted to the bytes", scenario("oblique", oblique, fit_iters=1500), "REBUILT", None, None, True)
    # 7. Two directions per matrix: nothing with one direction rebuilds it.
    def two(S, d, site, layer):
        e = torch.randn(hidden)
        e = e - (e @ d) * d
        e = e / e.norm()
        return S - 3.0 * torch.outer(d, d @ S) - 3.0 * torch.outer(e, e @ S)
    check("two directions", scenario("rank2", two, fit_iters=300), "OPEN", exact=False)
    # 8. Something else rewritten too (routed experts): never REBUILT, whatever the sites say.
    def routed(stock, ablit, layer, d):
        w = torch.randn(hidden, 64)
        stock[f"layers.{layer}.ffn.experts.3.w2.weight"] = pack(w.to(torch.bfloat16), "BF16")
        ablit[f"layers.{layer}.ffn.experts.3.w2.weight"] = pack((w - torch.outer(d, d @ w)).to(torch.bfloat16), "BF16")
    check("a third family rewritten", scenario("extra", plain(3.0), extra=routed), "OPEN", "plain", 3.0, exact=False)
    # 9. Other storage conventions: U8 scale codes named weight_scale on a [N, K/32] grid.
    check("U8 scales, weight_scale, 1 x 32", scenario("u8", plain(3.0), scale_suffix="_scale", scale_dtype="U8", block=(1, 32)),
          "REBUILT-AS-PUBLISHED", "plain", 3.0)
    # 10. --names gives the same answer as the byte scan.
    names_file = [n for n in index_tree(os.path.join(work, "asis", "stock")) if "wo_b" in n or "shared" in n]
    listed = analyse(os.path.join(work, "asis", "stock"), os.path.join(work, "asis", "ablit"),
                     os.path.join(work, "asis", "sidecar.safetensors"), names=names_file, quiet=True)
    if listed["verdict"] != "REBUILT-AS-PUBLISHED":
        problems.append(f"--names: {listed['verdict']}")
    print(f"\nSIDECAR-VS-WEIGHTS-SELFTEST {'PASS' if not problems else 'FAIL'}" + "".join(f"\n  {p}" for p in problems))
    return 0 if not problems else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--stock", help="directory of the stock checkpoint (*.safetensors)")
    parser.add_argument("--ablit", help="directory of the abliterated checkpoint")
    parser.add_argument("--sidecar", help="the published steering sidecar (.safetensors)")
    parser.add_argument("--names", help="file with the differing tensor names, one per line")
    parser.add_argument("--layers", help="comma-separated layers to analyse (default: all)")
    parser.add_argument("--payload", choices=("e4m3", "integer"), default="e4m3",
                        help="how to read U8/I8 weights (default e4m3)")
    parser.add_argument("--attention-pattern", default=SITE_PATTERNS["attention"],
                        help="regex naming the tensor family that writes the attention output")
    parser.add_argument("--shared-pattern", default=SITE_PATTERNS["shared expert"],
                        help="regex naming the tensor family that writes the shared-expert output")
    parser.add_argument("--derive", help="write the operators that rebuild the checkpoint as a v2 steering sidecar")
    parser.add_argument("--fit", action="store_true", help="fit an operator to the bytes even where a candidate rebuilds them")
    parser.add_argument("--no-fit", action="store_true", help="never fit an operator to the bytes")
    parser.add_argument("--fit-iters", type=int, default=1500)
    parser.add_argument("--threads", type=int, help="CPU threads for torch")
    parser.add_argument("--json", help="write the full result as JSON")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.selftest:
        return selftest()
    if not (args.stock and args.ablit and args.sidecar):
        parser.error("--stock, --ablit and --sidecar are required")
    names = None
    if args.names:
        with open(args.names, encoding="utf-8") as handle:
            names = [line.strip() for line in handle if line.strip()]
    layers = {int(x) for x in args.layers.split(",")} if args.layers else None
    result = analyse(args.stock, args.ablit, args.sidecar, names, args.payload,
                     patterns={"attention": args.attention_pattern, "shared expert": args.shared_pattern},
                     layers=layers, fit_iters=0 if args.no_fit else args.fit_iters, force_fit=args.fit,
                     derive=args.derive)
    if args.json:
        result.pop("operators", None)
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=1)
    return 0 if result["verdict"].startswith("REBUILT") else 1


if __name__ == "__main__":
    sys.exit(main())
