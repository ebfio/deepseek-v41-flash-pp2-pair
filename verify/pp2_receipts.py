#!/usr/bin/env python3
"""PP2 receipts: (1) 24-prompt parity vs the TP2 refs; (2) counter-based
acceptance + step time over a measurement window (launch-counted law: steps
from Prometheus deltas, never tok/s x assumed acceptance)."""
import hashlib
import json
import time
import urllib.request

import os
import sys

# The corpus and the references used to be /tmp/accept_sweep.py and /tmp/tp2_refs.json,
# which were never committed. Both now live next to this script; see verify/corpus.py.
HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.environ.get("PP2_CORPUS", os.path.join(HERE, "corpus.py"))
URL = os.environ.get("PP2_URL", "http://127.0.0.1:8001").rstrip("/")
MODEL = os.environ.get("PP2_MODEL", "deepseek-v4.1-flash")
src = open(CORPUS).read()
ns = {}
exec(compile(src.split("COMBOS")[0], "corpus", "exec"), ns)
CLASSES = ns["CLASSES"]
SEP = "\n<<<SPLIT>>>\n"
EMPTY = hashlib.md5(SEP.encode()).hexdigest()[:12]
# PP2_REFS: the reference file ([class, index, md5, tokens] rows). PP2_RECORD_REFS=1 writes
# it from this lane instead of comparing against it. Without a file, parity is unmeasured.
REFS = os.environ.get("PP2_REFS", os.path.join(HERE, "refs", "parity_refs.json"))
RECORD = os.environ.get("PP2_RECORD_REFS", "") == "1"
refs = {}
if not RECORD and os.path.exists(REFS):
    refs = {(c, i): (h, n) for c, i, h, n in json.load(open(REFS))}


def one(prompt, max_tokens=1536):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 0.95,
        }
    ).encode()
    t0 = time.time()
    r = urllib.request.urlopen(
        urllib.request.Request(
            URL + "/v1/chat/completions",
            body,
            {"Content-Type": "application/json"},
        ),
        timeout=900,
    )
    d = json.load(r)
    dt = time.time() - t0
    m = d["choices"][0]["message"]
    txt = (m.get("reasoning_content") or m.get("reasoning") or "") + SEP + (
        m.get("content") or ""
    )
    h = hashlib.md5(txt.encode()).hexdigest()[:12]
    n = d["usage"]["completion_tokens"]
    return h, n, dt, d["choices"][0].get("finish_reason")


def spec_counters():
    with urllib.request.urlopen(URL + "/metrics", timeout=10) as r:
        text = r.read().decode()
    drafts = accepted = 0.0
    steps = None
    for line in text.splitlines():
        if line.startswith("vllm:spec_decode_num_drafts_total{"):
            drafts = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
            # label variants: bare or with engine/model; take the max seen
            v = float(line.rsplit(" ", 1)[1])
            accepted = max(accepted, v)
        elif line.startswith("vllm:num_steps_total"):
            steps = float(line.rsplit(" ", 1)[1])
    return drafts, accepted, steps


# --- (2) counter window on ONE fixed prompt (steps counted by the engine) ---
d0, a0, s0 = spec_counters()
h, n, dt, fin = one(
    CLASSES["prose"][0] if "prose" in CLASSES else list(CLASSES.values())[0][0],
    512,
)
d1, a1, s1 = spec_counters()
steps = (s1 - s0) if s0 is not None and s1 is not None else None
print("WINDOW: ctok=%d wall=%.1fs client_tok/s=%.1f" % (n, dt, n / dt))
print(
    "WINDOW: drafts_delta=%d accepted_delta=%d steps_delta=%s"
    % (d1 - d0, a1 - a0, steps)
)
if steps:
    print("WINDOW: step_time=%.2f ms acceptance=%.2f" % (1000 * dt / steps, (a1 - a0) / steps))

# --- (1) parity sweep ---
same = diff = 0
rows = []
for cls, prompts in CLASSES.items():
    for i, p in enumerate(prompts):
        h, n, dt, fin = one(p)
        key = (cls, i)
        if RECORD:
            rows.append((cls, i, h, n))
            print("%s[%d] recorded %s tok/s=%.1f n=%d" % (cls, i, h, n / dt, n))
            continue
        if key not in refs:
            continue
        rh, rn = refs[key]
        if h == EMPTY:
            verdict = "EMPTY!"
            diff += 1
        elif h == rh:
            verdict = "same"
            same += 1
        else:
            verdict = "diff"
            diff += 1
        rows.append((cls, i, verdict, n, rn, n / dt))
        print("%s[%d] %s tok/s=%.1f n=%d ref_n=%d" % (cls, i, verdict, n / dt, n, rn))
if RECORD:
    os.makedirs(os.path.dirname(REFS) or ".", exist_ok=True)
    json.dump([list(r) for r in rows], open(REFS, "w"), indent=1)
    print("PARITY: recorded %d references to %s" % (len(rows), REFS))
elif not refs:
    print("PARITY: unmeasured (no references at %s; record them with PP2_RECORD_REFS=1)" % REFS)
    sys.exit(3)
else:
    print("PARITY: %d same, %d diff (of %d) vs %s" % (same, diff, same + diff, REFS))
