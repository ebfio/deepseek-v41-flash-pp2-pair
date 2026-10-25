#!/usr/bin/env python3
"""PP2 soak: C1/C2/C4/C8 aggregate tok/s + rank0 host memory under load.

Gate: PP pipelines batches across stages, so C2 aggregate should
approach ~2x C1 (vs TP2's C2 ~= C1-per-step). free -g on rank 0 under load:
both Engram tables + page cache on one Grace; no swap, no compaction storms.
"""
import json
import subprocess
import time
import urllib.request

src = open("/tmp/accept_sweep.py").read()
ns = {}
exec(compile(src.split("COMBOS")[0], "corpus", "exec"), ns)
CLASSES = ns["CLASSES"]
# Round-robin the corpus: realistic mix, no prompt-identity confound in the
# aggregate (every level runs the same prompt multiset).
PROMPTS = [p for ps in CLASSES.values() for p in ps]


def one(prompt, max_tokens=512):
    body = json.dumps(
        {
            "model": "deepseek-v4.1-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.95,
        }
    ).encode()
    r = urllib.request.urlopen(
        urllib.request.Request(
            "http://127.0.0.1:8001/v1/chat/completions",
            body,
            {"Content-Type": "application/json"},
        ),
        timeout=900,
    )
    d = json.load(r)
    return d["usage"]["completion_tokens"]


def free_g():
    out = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.startswith("Mem:"):
            f = line.split()
            return int(f[2]), int(f[3]), int(f[4])  # used, free, avail
    return None


def level(c, n_batches):
    """c concurrent streams; each stream runs n_batches prompts round-robin."""
    import concurrent.futures

    results = []

    def stream(sid):
        tot = 0
        for j in range(n_batches):
            tot += one(PROMPTS[(sid * n_batches + j) % len(PROMPTS)])
        return tot

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        futs = [ex.submit(stream, i) for i in range(c)]
        for f in futs:
            results.append(f.result())
    wall = time.time() - t0
    total = sum(results)
    mem = free_g()
    print(
        "C%d: aggregate=%.1f tok/s total=%d wall=%.1fs per-stream=%s mem(used,free,avail)=%s"
        % (
            c,
            total / wall,
            total,
            wall,
            [round(x / wall, 1) for x in results],
            mem,
        ),
        flush=True,
    )


mem0 = free_g()
print("baseline mem(used,free,avail)=%s" % (mem0,), flush=True)
for c in (1, 2, 4, 8):
    level(c, n_batches=3)
mem1 = free_g()
swap = subprocess.run(
    ["bash", "-c", "free -g | awk '/Swap:/{print $3}'"], capture_output=True, text=True
).stdout.strip()
compaction = subprocess.run(
    ["bash", "-c", "grep -c compact /proc/vmstat || true"],
    capture_output=True,
    text=True,
).stdout.strip()
print("post mem=%s swap_used_g=%s compact_events=%s" % (mem1, swap, compaction), flush=True)
