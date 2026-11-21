#!/usr/bin/env python3
"""Self-test of verify/decode_rate.py against a toy lane whose answers are known.

    python3 verify/decode_rate_selftest.py

The toy streams a fixed number of tokens per speculative round at a fixed time per round,
after a fixed delay to the first token, and counts its rounds in the same Prometheus
counters vLLM exports. The script has to read those three numbers back.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "decode_rate.py"


class Toy:
    def __init__(self, ttft: float, round_s: float, per_round: int, counters: bool = True, stranger: int = 0,
                 broken: str = "", cache: str = "usage"):
        self.ttft, self.round_s, self.per_round, self.has_counters, self.stranger = ttft, round_s, per_round, counters, stranger
        self.broken = broken  # "refuses": HTTP 400; "stream-error": an error object as the first event
        # How this lane reports prompt tokens served from its prefix cache: in each
        # response's usage, in a counter only, or not at all. "ignores-salt" is a lane
        # whose cache serves a prompt it has seen whatever cache_salt says.
        self.cache = cache
        self.seen: set[str] = set()
        self.cached_total = 0
        self.last_body: dict = {}
        self.lock = threading.Lock()
        self.drafts = self.accepted = self.generated = 0

    def metrics(self) -> str:
        if not self.has_counters:
            return "# nothing\n"
        with self.lock:
            return (f'vllm:spec_decode_num_drafts_total{{engine="0"}} {float(self.drafts)}\n'
                    f'vllm:spec_decode_num_drafts_created{{engine="0"}} 1.7e9\n'
                    f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {float(5 * self.drafts)}\n'
                    f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {float(self.accepted)}\n'
                    f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="0"}} 1e9\n'
                    f'vllm:generation_tokens_total{{engine="0"}} {float(self.generated)}\n'
                    + ("" if self.cache == "none" else
                       f'vllm:prompt_tokens_cached_total{{engine="0"}} {float(self.cached_total)}\n'
                       f'vllm:prompt_tokens_cached_by_source_total{{engine="0",source="x"}} 1e9\n'))


def serve(toy: Toy):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args) -> None:
            pass

        def do_GET(self) -> None:  # noqa: N802
            payload = (json.dumps({"data": [{"id": "toy"}]}) if self.path == "/v1/models" else toy.metrics()).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            total = int(body["max_tokens"])
            if toy.broken == "refuses":
                payload = json.dumps({"error": {"message": "steering_scale must be in [0, 1]", "code": 400}}).encode()
                self.send_response(400)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()

            def event(payload) -> None:
                self.wfile.write(b"data: " + (payload if isinstance(payload, bytes) else json.dumps(payload).encode()) + b"\n\n")
                self.wfile.flush()

            time.sleep(toy.ttft)
            if toy.broken == "stream-error":
                event({"error": {"message": "the engine died mid-request", "code": 500}})
                event(b"[DONE]")
                return
            sent = 0
            while sent < total:
                count = min(toy.per_round, total - sent)
                if sent:
                    time.sleep(toy.round_s)
                with toy.lock:
                    toy.drafts += 1
                    toy.accepted += count - 1
                    toy.generated += count + toy.stranger
                sent += count
                event({"choices": [{"delta": {"reasoning_content": "x " * count}, "finish_reason": None}]})
            event({"choices": [{"delta": {}, "finish_reason": "length"}]})
            words = len(body["messages"][0]["content"].split())
            usage = {"completion_tokens": sent, "prompt_tokens": words}
            with toy.lock:
                toy.last_body = body
                key = body["messages"][0]["content"] + ("" if toy.cache == "ignores-salt" else body.get("cache_salt", ""))
                cached = words if key in toy.seen else 0
                toy.seen.add(key)
                toy.cached_total += cached
            if toy.cache in ("usage", "ignores-salt"):
                usage["prompt_tokens_details"] = {"cached_tokens": cached}
            event({"choices": [], "usage": usage})
            event(b"[DONE]")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def run(toy: Toy, *extra: str) -> tuple[int, dict, str]:
    server, url = serve(toy)
    try:
        done = subprocess.run([sys.executable, "-B", str(SCRIPT), "--url", url, "--repeat", "3", "--json", "-",
                               *(() if "--tokens" in extra else ("--tokens", "240")), *extra],
                              capture_output=True, text=True, timeout=300,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    finally:
        server.shutdown()
    start = done.stdout.find('{\n  "url"')
    data = json.loads(done.stdout[start:]) if start >= 0 else {}
    return done.returncode, data, done.stdout + done.stderr


def main() -> int:
    failures = []

    def check(name: str, good: bool, detail: str) -> None:
        print(f"{'ok  ' if good else 'FAIL'}  {name:<46} {detail}")
        if not good:
            failures.append(name)

    def middle(data: dict, index: int) -> dict:
        runs = data["runs"][index]
        return sorted(runs, key=lambda r: r["tok_s_aggregate"])[len(runs) // 2]

    # 4 tokens per 10 ms round after 0.20 s: 400 tok/s per stream, whatever the first token cost.
    status, data, text = run(Toy(0.20, 0.010, 4), "--streams", "1,4")
    one, four = (middle(data, 0), middle(data, 1)) if data else ({}, {})
    check("one stream: tokens per second after the first", status == 0 and 330 <= one.get("tok_s_per_stream", 0) <= 410,
          f"{one.get('tok_s_per_stream', 0):.0f} tok/s (the toy makes 400; 240 tokens in 0.79 s would read 304)")
    check("one stream: time to first token", 0.19 <= one.get("ttft_s", 0) <= 0.30, f"{one.get('ttft_s', 0):.3f} s (0.20)")
    check("one stream: tokens per round", abs(one.get("tokens_per_round", 0) - 4.0) < 0.01,
          f"{one.get('tokens_per_round', 0):.2f} (4)")
    check("one stream: milliseconds per round", 9.5 <= one.get("ms_per_round", 0) <= 12.5,
          f"{one.get('ms_per_round', 0):.2f} ms (10, over 59 of 60 rounds)")
    check("four streams: aggregate and per-stream", 1250 <= four.get("tok_s_aggregate", 0) <= 1640
          and 330 <= four.get("tok_s_per_stream", 0) <= 410,
          f"{four.get('tok_s_aggregate', 0):.0f} aggregate, {four.get('tok_s_per_stream', 0):.0f} per stream (1600, 400)")
    check("four streams: per round is per stream", abs(four.get("tokens_per_round", 0) - 4.0) < 0.01
          and 9.5 <= four.get("ms_per_round", 0) <= 13.0,
          f"{four.get('tokens_per_round', 0):.2f} tokens, {four.get('ms_per_round', 0):.2f} ms")
    # Nothing accepted: the same round time yields a quarter of the speed.
    status, data, text = run(Toy(0.05, 0.010, 1), "--streams", "1")
    one = middle(data, 0) if data else {}
    check("no draft accepted: 1 token per round", abs(one.get("tokens_per_round", 0) - 1.0) < 0.01
          and 80 <= one.get("tok_s_per_stream", 0) <= 102,
          f"{one.get('tokens_per_round', 0):.2f} tokens per round, {one.get('tok_s_per_stream', 0):.0f} tok/s (1, 100)")
    status, data, text = run(Toy(0.05, 0.002, 4, counters=False), "--streams", "1")
    check("a lane without the counters", status == 0 and "per-round numbers unmeasured" in text, "says so, still reports tok/s")
    status, data, text = run(Toy(0.05, 0.002, 4, stranger=3), "--streams", "1")
    check("other traffic on the lane", status == 0 and "other traffic" in text, "says so")
    status, data, text = run(Toy(0.05, 0.002, 4), "--streams", "1", "--context-words", "3000")
    words = middle(data, 0).get("prompt_tokens") or 0 if data else 0
    check("a long context in front of the prompt", status == 0 and 3000 <= words <= 3200, f"prompt of {words} words reached the lane")
    check("...and its prefill is reported cold", "prefill cold in every run" in text, "each request has its own cache_salt")
    # Tokens per round depend on the text, so two lanes are compared on one prompt.
    toy = Toy(0.05, 0.002, 4)
    status, data, text = run(toy, "--streams", "1", "--train-prompt", "--tokens", "64")
    body = toy.last_body
    check("the README's train prompt, as the README sent it",
          status == 0 and body.get("messages", [{}])[0].get("content", "").startswith("A train leaves at 60 mph.")
          and body.get("top_p") == 0.95 and body.get("max_tokens") == 64 and body.get("temperature") == 0
          and "train prompt" in text, f"top_p {body.get('top_p')}, max_tokens {body.get('max_tokens')}")
    one = middle(data, 0) if data else {}
    check("tokens over wall time is reported beside it", 0 < one.get("tok_s_with_ttft", 0) < one.get("tok_s_per_stream", 0),
          f"{one.get('tok_s_with_ttft', 0):.0f} with the first token's latency, {one.get('tok_s_per_stream', 0):.0f} after it")
    status, data, text = run(Toy(0.05, 0.002, 4, cache="ignores-salt"), "--streams", "1")
    check("a prefill that was not cold is said so", status == 0 and "prefill NOT cold" in text, "from the responses' usage")
    status, data, text = run(Toy(0.05, 0.002, 4, cache="counter"), "--streams", "1")
    check("cold prefill, lane with the counter only", status == 0 and "prefill cold in every run" in text,
          "from vllm:prompt_tokens_cached")
    status, data, text = run(Toy(0.05, 0.002, 4, cache="none"), "--streams", "1")
    check("a lane that reports no cached tokens", status == 0 and "does not report cached prompt tokens" in text, "says so")
    status, data, text = run(Toy(0.01, 0.002, 4, broken="refuses"), "--streams", "1")
    check("a lane that refuses the request", status == 2 and "HTTP 400" in text and "must be in [0, 1]" in text,
          "exit 2, with the lane's own message")
    status, data, text = run(Toy(0.01, 0.002, 4, broken="stream-error"), "--streams", "1")
    check("an error inside the stream", status == 2 and "the engine died mid-request" in text and "Traceback" not in text,
          "exit 2, with the lane's own message")
    print(f"\nDECODE-RATE-SELFTEST {'PASS' if not failures else 'FAIL'}: {len(failures)} check(s) failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
