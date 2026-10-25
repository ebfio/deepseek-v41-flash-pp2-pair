#!/usr/bin/env python3
"""Agent-scale prefix-cache probe.

2K-token system prompt, 8 turns growing the context to ~20K+. Queries count
TOKENS (metrics/stats.py:117-121); block size 128; SWA replay discards hits
<= its window. Healthy signature: per-turn hits ~= previous total minus the
partial block minus the replay window; misses on the most recent assistant
turn (re-tokenization) are expected and healthy.
"""
import json
import time
import urllib.request

BASE = "http://localhost:8001"

# ~2K tokens of byte-stable system context: a fake ops manual.
SECTION = (
    "Fleet runbook section {n}: host {host} runs service {svc} on port {port}. "
    "The service holds {gb} GiB of weights in page cache and exports metrics at "
    "/metrics. Its health check is a 200 on /v1/models. On failure, restart via "
    "the systemd unit, wait for graph capture, then run the parity script. "
    "Never restart during a deploy window. Log rotation is weekly. The on-call "
    "rotation includes seven engineers across three time zones. Escalation goes "
    "to the platform lead after two failed restarts. "
)
HOSTS = ["erebus", "hades", "proteus", "acheron", "selene", "thea", "nemesis"]
SVCS = ["vllm-dsv41", "vllm-glm53", "asr-qwen3", "omnivoice", "exporter"]
SYSTEM = "You are Aline, the fleet operations agent. Use the runbook below.\n\n" + "".join(
    SECTION.format(
        n=i, host=HOSTS[i % 7], svc=SVCS[i % 5], port=8000 + (i % 40), gb=40 + (i % 60)
    )
    for i in range(90)
)

TURNS = [
    "Summarize the restart policy from the runbook in two paragraphs, then list every host that runs vllm-dsv41.",
    "Now explain the escalation path in detail. If erebus failed twice at 3am, walk through exactly what happens, step by step, in about 400 words.",
    "Write a short incident report template based on this runbook, with six fields and guidance for each. Then fill in a sample for a hypothetical asr-qwen3 crash.",
    "Draft the weekly log-rotation review checklist: ten items, each with a one-line rationale grounded in the runbook.",
    "Compose an on-call handover note covering the rotation, the escalation policy, and the three most fragile services, with reasoning.",
    "Based on everything above, write a one-page fleet health briefing as you would send it, then annotate it with which runbook sections each claim comes from.",
    "Now audit your own briefing: three risks it missed, three things it overstated, and a corrected closing paragraph.",
    "Finally, produce the complete consolidated runbook quick-reference card: every host, service, port, and policy, organized in tables with a short usage note.",
]


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    q = h = None
    for line in text.splitlines():
        if line.startswith("vllm:prefix_cache_queries_total"):
            q = float(line.rsplit(" ", 1)[1])
        elif line.startswith("vllm:prefix_cache_hits_total"):
            h = float(line.rsplit(" ", 1)[1])
    return q, h


def turn(msgs):
    t0 = time.time()
    body = json.dumps(
        {
            "model": "deepseek-v4.1-flash",
            "messages": msgs,
            "temperature": 0,
            "max_tokens": 1400,
        }
    ).encode()
    r = urllib.request.urlopen(
        urllib.request.Request(
            BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"}
        ),
        timeout=600,
    )
    out = json.load(r)
    dt = time.time() - t0
    m = out["choices"][0]["message"]
    content = m.get("content") or ""
    ptok = out["usage"]["prompt_tokens"]
    ctok = out["usage"]["completion_tokens"]
    return dt, ptok, ctok, content


msgs = [{"role": "system", "content": SYSTEM}]
q0, h0 = metrics()
print("system prompt ~%d chars" % len(SYSTEM))
for i, t in enumerate(TURNS, 1):
    msgs.append({"role": "user", "content": t})
    dt, ptok, ctok, content = turn(msgs)
    msgs.append({"role": "assistant", "content": content})
    q1, h1 = metrics()
    dq, dh = q1 - q0, h1 - h0
    rate = dh / dq if dq else float("nan")
    print(
        "turn %d: ptok=%d ctok=%d dq=%.0f dh=%.0f hit_rate=%.2f ttft~=%.1fs tok/s=%.1f"
        % (i, ptok, ctok, dq, dh, rate, dt, ctok / dt),
        flush=True,
    )
    q0, h0 = q1, h1
