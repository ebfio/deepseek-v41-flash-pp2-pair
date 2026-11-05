#!/usr/bin/env python3
"""GSM8K width slice — parameterised offset/limit/max_tokens.

The 100-question gate (gsm8k_gate.py) is the golden instrument and stays fixed
at q1-100 / 2048 tokens. This variant exists to widen the sample so a two-flip
delta can be read as a rate with an interval instead of noise.

  python3 gsm8k_width.py --offset 100 --limit 400 --max-tokens 4096
  python3 gsm8k_width.py --offset 86  --limit 1   --max-tokens 8192   # q87 rerun

0-shot, temp 0, C4. Scoring is identical to the gate: the target's '#### N'
integer vs the last integer in the model's content.
"""
import argparse
import concurrent.futures
import json
import math
import re
import urllib.request

BASE = "http://localhost:8001"
URL = (
    "https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k"
    "&config=main&split=test&offset={offset}&length={length}"
)
SUFFIX = "\n\nReason step by step, then give the final numeric answer on a line starting with '#### '."


def fetch(offset, length):
    # The datasets-server rows endpoint caps length at 100, so page.
    rows = []
    fetched = 0
    while fetched < length:
        want = min(100, length - fetched)
        try:
            with urllib.request.urlopen(  # noqa: S310 - fixed https dataset endpoint
                URL.format(offset=offset + fetched, length=want), timeout=60
            ) as r:
                d = json.load(r)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"dataset fetch failed at offset {offset + fetched}: {e}"
            ) from e
        batch = d["rows"]
        if not batch:
            break
        for row in batch:
            q = row["row"]["question"]
            m = re.search(r"####\s*([\-0-9,\.]+)", row["row"]["answer"])
            target = m.group(1).replace(",", "").rstrip(".") if m else None
            rows.append((q, target))
        fetched += len(batch)
    return rows


def last_int(text):
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text.replace(",", ""))
    if not nums:
        return None
    v = nums[-1].rstrip(".")
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except ValueError:
        return None


def ask(idx, q, target, max_tokens):
    body = json.dumps(
        {
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": q + SUFFIX}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
    ).encode()
    try:
        r = urllib.request.urlopen(  # noqa: S310 - localhost engine endpoint
            urllib.request.Request(
                BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"}
            ),
            timeout=900,
        )
        out = json.load(r)
        content = out["choices"][0]["message"].get("content") or ""
        pred = last_int(content)
        ok = target is not None and pred is not None and float(pred) == float(target)
        return idx, ok, target, pred, out["usage"]["completion_tokens"]
    except Exception as e:  # noqa: BLE001
        return idx, None, target, None, str(e)[:80]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


ap = argparse.ArgumentParser()
ap.add_argument("--offset", type=int, default=100)
ap.add_argument("--limit", type=int, default=400)
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()

rows = fetch(a.offset, a.limit)
assert len(rows) == a.limit, f"expected {a.limit} rows, got {len(rows)}"
print(
    f"q{a.offset + 1}-q{a.offset + a.limit}  max_tokens={a.max_tokens}", flush=True
)

results = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
    futs = [ex.submit(ask, i, q, t, a.max_tokens) for i, (q, t) in enumerate(rows)]
    for done, f in enumerate(concurrent.futures.as_completed(futs), start=1):
        i, ok, target, pred, info = f.result()
        results[i] = (ok, target, pred, info)
        if done % 50 == 0:
            known = [v[0] for v in results.values() if v[0] is not None]
            print(
                f"progress {done}/{a.limit} running_acc={sum(known) / max(1, len(known)):.3f}",
                flush=True,
            )

n = a.limit
correct = sum(1 for v in results.values() if v[0] is True)
errors = sum(1 for v in results.values() if v[0] is None)
lo, hi = wilson(correct, n)
print(
    f"GSM8K q{a.offset + 1}-q{a.offset + n}: {correct}/{n} = {correct / n:.3f} "
    f"(95% CI {lo:.3f}-{hi:.3f}) errors={errors}"
)
for i in sorted(results):
    ok, target, pred, info = results[i]
    if ok is not True:
        print(f"MISS q{a.offset + i + 1}: target={target} pred={pred} ctok={info}")
