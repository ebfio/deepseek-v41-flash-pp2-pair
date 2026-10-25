#!/usr/bin/env python3
"""GSM8K 100-question slice on the PP2 lane (golden re-bank gate).

0-shot, temp 0, max_tokens 2048 (reasoning model needs budget). Exact-match:
the target's '#### N' integer vs the last integer in the model's content.
C4 concurrency (soak-validated band). Prints running accuracy + the misses.
"""
import json
import re
import urllib.request
import concurrent.futures

BASE = "http://localhost:8001"
N = 100
URL_ROWS = (
    "https://datasets-server.huggingface.co/rows?dataset=openai%2Fgsm8k"
    "&config=main&split=test&offset=0&length=100"
)


def fetch():
    with urllib.request.urlopen(URL_ROWS, timeout=60) as r:
        d = json.load(r)
    rows = []
    for row in d["rows"]:
        q = row["row"]["question"]
        a = row["row"]["answer"]
        m = re.search(r"####\s*([\-0-9,\.]+)", a)
        target = m.group(1).replace(",", "").rstrip(".") if m else None
        rows.append((q, target))
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


def ask(i, q, target):
    body = json.dumps(
        {
            "model": "deepseek-v4.1-flash",
            "messages": [
                {
                    "role": "user",
                    "content": q + "\n\nReason step by step, then give the final numeric answer on a line starting with '#### '.",
                }
            ],
            "temperature": 0,
            "max_tokens": 2048,
        }
    ).encode()
    try:
        r = urllib.request.urlopen(
            urllib.request.Request(
                BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"}
            ),
            timeout=600,
        )
        out = json.load(r)
        content = out["choices"][0]["message"].get("content") or ""
        pred = last_int(content)
        ok = target is not None and pred is not None and float(pred) == float(target)
        return i, ok, target, pred, out["usage"]["completion_tokens"]
    except Exception as e:
        return i, None, target, None, str(e)[:80]


rows = fetch()
assert len(rows) >= N, "expected >= %d rows, got %d" % (N, len(rows))
rows = rows[:N]
print("fetched %d questions" % len(rows), flush=True)
results = {}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    futs = [ex.submit(ask, i, q, t) for i, (q, t) in enumerate(rows)]
    done = 0
    for f in concurrent.futures.as_completed(futs):
        i, ok, target, pred, info = f.result()
        results[i] = (ok, target, pred, info)
        done += 1
        if done % 20 == 0:
            so_far = [v[0] for v in results.values()]
            known = [x for x in so_far if x is not None]
            print(
                "progress %d/%d running_acc=%.2f" % (done, N, sum(known) / max(1, len(known))),
                flush=True,
            )
correct = sum(1 for v in results.values() if v[0] is True)
errors = sum(1 for v in results.values() if v[0] is None)
print("GSM8K %d-slice: accuracy=%.2f (%d/%d) errors=%d" % (N, correct / N, correct, N, errors))
for i in sorted(results):
    ok, target, pred, info = results[i]
    if ok is not True:
        print("MISS q%d: target=%s pred=%s ctok=%s" % (i, target, pred, info))
