#!/usr/bin/env python3
"""Self-test of verify/steering_probe.py against toy servers. No GPU, no vLLM.

    python3 verify/steering_probe_selftest.py
    python3 verify/steering_probe_selftest.py --against <another revision's steering_probe.py>

The second form runs that other probe against three of these toy lanes (a sidecar that
matters at few positions, a tracer, one disturbed request among the references) and
prints what it makes of them. With revision 4's probe it prints the three outcomes the
pair had on 2026-11-20.

Each toy server speaks just enough of the OpenAI API and is either correct or broken in
one specific way. The probe has to pass the correct one, also when it is as noisy as a
real engine, and fail each broken one in the test that is meant to catch it. Run this
before trusting a probe verdict on a real lane, and again after any change to the probe.

The toy is built to have the properties that defeated revision 2 of the probe on the
pair: its first generated token has probability 1 at any scale (a reasoning model's
does), a cache hit changes the output a little with no steering involved (the engine
recomputes the hit's last window), and one variant never repeats itself exactly.

And the two that left revision 3 undecided there: requests in flight together change
each other's output with no steering involved (one variant moves a position in twelve by
about as much as steering does), and a token computed in a decode step is not exactly
the token its context gets when prefilled.

And the three that left revision 4 undecided there. The vocabulary is larger than a
top-20 list, and steering can put tokens on top that the stock list does not name at
all: at a share of the positions only (a sidecar that matters at few positions), or at
every position with answers that stop after 15 tokens and have no text (a tracer
sidecar). Revision 4's distance read such a change as nothing. One request in a run can
be disturbed (a hiccup), which used to raise every prompt's bar. And other clients can
be on the lane, which the engine's counters show.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROBE = Path(__file__).resolve().parent / "steering_probe.py"
VOCAB = 48  # tokens the stock model chooses among at a position
DECISIVE, TRACER = 1000, 2000  # first ids of the tokens steering can put on top instead
TRACER_STOP = 15  # a tracer row at scale 1 ends after this many tokens
GENERATION_MARK = 99  # the id a tokenised chat prompt ends with; no token the toy generates or tokenises to
HARMFUL_MARK = "[H]"
BLOCK = 16


def unit(*parts) -> float:
    """A fixed pseudo-random number in [0, 1) from the arguments."""
    digest = hashlib.sha256(json.dumps(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def tokenize(text: str) -> list[int]:
    return [100 + int.from_bytes(hashlib.sha256(w.encode()).digest()[:4], "big") % 50000 for w in text.split()] + [
        GENERATION_MARK
    ]


class Toy:
    """A two-stage 'model'. Each stage adds its own direction to the token being computed,
    scaled, and leaves its mark on the context that later tokens attend to."""

    def __init__(self, mode: str = "correct", debug: bool = True, weights: str = "stock", drift: float = 0.0,
                 noise: float = 0.0, usage_details: bool = True, has_tokenize: bool = True, window: int = 0,
                 flaky_text: bool = False, burst_noise: float = 0.0, decode_gap: float = 0.0,
                 module: bool = True, effect: float = 2.0, decisive: float = 0.0, tracer: bool = False,
                 tracer_noise: float = 0.0, hiccup: int = 0, counters: bool = True) -> None:
        self.mode, self.debug, self.weights, self.drift = mode, debug, weights, drift
        self.noise, self.usage_details, self.has_tokenize, self.window = noise, usage_details, has_tokenize, window
        self.flaky_text = flaky_text
        # burst_noise: with other requests in flight, one generated position in twelve is
        # pushed around by this much (logit units), steering or not. decode_gap: a token
        # computed in a decode step differs this much from the same context prefilled.
        # module=False: an engine without the steering module (the field is ignored).
        self.burst_noise, self.decode_gap, self.module = burst_noise, decode_gap, module
        # effect: how hard steering pushes the stock candidates around (logit units).
        # decisive: the share of positions at which steering instead puts other tokens on
        # top, none of them in the stock list. tracer: it does that at every position, and
        # a row at scale 1 stops after TRACER_STOP tokens with no text; tracer_noise makes
        # such a row's output differ from one request to the next. hiccup: the n-th
        # generation request, once, is disturbed at every position.
        self.effect, self.decisive, self.tracer, self.tracer_noise = effect, decisive, tracer, tracer_noise
        self.hiccup, self.counters = hiccup, counters
        self.lock = threading.Lock()
        self.cache: dict[tuple, tuple[float, float]] = {}
        self.hits = 0
        self.queries = 0
        self.inflight: dict[int, tuple[float, float]] = {}
        self.counter = 0
        self.finished = 0
        self.generated_tokens = 0
        self.steps = 0.0
        self.random = random.Random(7)

    def validate(self, body: dict) -> str | None:
        xargs = body.get("vllm_xargs") or {}
        scale = xargs.get("steering_scale", body.get("steering_scale"))
        if not self.module:
            return None
        if self.mode != "no-validation" and scale is not None:
            if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not 0 <= scale <= 1:
                return "steering_scale must be a number in [0, 1]"
        if "steering_ranks" in xargs and not self.debug:
            return "steering_ranks is a debug knob and is off on this server"
        return None

    def scales(self, body: dict) -> tuple[float, float]:
        xargs = body.get("vllm_xargs") or {}
        scale = xargs.get("steering_scale", body.get("steering_scale"))
        scale = float(scale) if isinstance(scale, (int, float)) else 0.0
        scale = min(max(scale, 0.0), 1.0)
        mask = xargs.get("steering_ranks")
        s0 = scale if mask is None or int(mask) & 1 else 0.0
        s1 = scale if mask is None or int(mask) & 2 else 0.0
        if self.mode == "noop" or not self.module:
            return 0.0, 0.0
        if self.mode == "rank1-missing":
            return s0, 0.0
        return s0, s1

    def run(self, ids: list[int], body: dict, harmful: bool) -> dict:
        own = self.scales(body)
        with self.lock:
            self.counter += 1
            ticket = self.counter
            self.inflight[ticket] = own
        try:
            return self.generate(ticket, own, ids, body, harmful)
        finally:
            with self.lock:
                self.inflight.pop(ticket, None)

    def generate(self, ticket: int, own: tuple[float, float], ids: list[int], body: dict, harmful: bool) -> dict:
        time.sleep(0.006)  # let the rest of a burst arrive
        with self.lock:
            peers = [v for k, v in sorted(self.inflight.items()) if k != ticket]
        crowded = bool(peers)  # other requests are in flight: the batch is shared
        others = [v for v in peers if v != own]
        if self.mode == "row-swap" and others:
            own = others[-1]  # this request's rows read another request's scale
        blocks = (len(ids) // BLOCK) * BLOCK
        with self.lock:
            key = (body.get("cache_salt"), own if self.mode != "unsalted-cache" else None, tuple(ids[:blocks]))
            self.queries += len(ids)
            # Like the engine: a hit no longer than the replay window is recomputed in full.
            hit = self.mode != "cache-off" and blocks > self.window and key in self.cache
            if hit:
                self.hits += blocks
                prefix = self.cache[key]
            else:
                if blocks:
                    self.cache[key] = own
                prefix = own
        # The scales each context token was computed under: a cached prefix keeps the
        # scales of the request that computed it.
        computed = [prefix] * blocks + [own] * (len(ids) - blocks) if hit else [own] * len(ids)
        generated: list[int] = []
        profile = []
        limit = min(int(body.get("max_tokens") or 16), 26)
        stops = self.tracer and sum(own) >= 2.0  # a tracer row at scale 1 runs out early
        if stops:
            limit = min(limit, TRACER_STOP)
        for step in range(limit):
            current = own if step == 0 or self.mode != "decode-unsteered" else (0.0, 0.0)
            if others and step and unit("crossed", ticket, step) < 0.3:
                if self.mode == "row-swap-some":
                    current = others[-1]  # three decode rows in ten read another request's scale
                elif self.mode == "stage1-swap-some":
                    current = (own[0], others[-1][1])  # ...on the second stage only
            context = hashlib.sha256(json.dumps(ids + generated).encode()).hexdigest()[:20]
            mean0 = sum(c[0] for c in computed) / len(computed)
            mean1 = sum(c[1] for c in computed) / len(computed)
            first = not generated and ids[-1] == GENERATION_MARK
            strength = (current[0] + current[1]) / 2
            logits: dict[int, float] = {}
            for token in range(VOCAB):
                value = -3 * unit(self.weights, context, token)
                value += self.drift * unit("boot", context, token)  # another boot, last digits differ
                value += self.effect * current[0] * (2 * unit("stage0", context, token) - 1)
                value += self.effect * current[1] * (2 * unit("stage1", context, token) - 1)
                value += 0.75 * self.effect * mean0 * (2 * unit("kv0", context, token) - 1)
                value += 0.75 * self.effect * mean1 * (2 * unit("kv1", context, token) - 1)
                if hit:
                    value += 0.01 * (2 * unit("replayed window", context, token) - 1)
                if self.noise:
                    value += self.noise * (2 * unit("noise", ticket, step, token) - 1)
                if step and self.decode_gap:
                    value += self.decode_gap * (2 * unit("decode path", context, token) - 1)
                if crowded and self.burst_noise and unit("burst", ticket, step) < 1 / 12:
                    value += self.burst_noise * (2 * unit("burst direction", ticket, step, token) - 1)
                if ticket == self.hiccup:
                    value += 4.0 * (2 * unit("hiccup", step, token) - 1)
                if first and token == 0:
                    value += 14.0  # the reasoning model's first token: certain at any scale
                logits[token] = value
            if strength and self.decisive and unit("decision", context) < self.decisive:
                # A position where the steered model wants other tokens altogether. Each
                # stage contributes half; at scale 1 none of the stock candidates is left
                # in the top 20.
                for token in range(DECISIVE, DECISIVE + VOCAB):
                    logits[token] = -9 + 16 * strength - 3 * unit("decisive", context, token)
            if strength and self.tracer:
                for token in range(TRACER, TRACER + VOCAB):
                    value = 22 * strength - 3 * unit("tracer", context, token, round(strength, 3))
                    if self.tracer_noise:
                        value += self.tracer_noise * (2 * unit("tracer noise", ticket, step, token) - 1)
                    logits[token] = value
            norm = math.log(sum(math.exp(v) for v in logits.values()))
            ranked = sorted(((token, value - norm) for token, value in logits.items()), key=lambda item: -item[1])
            profile.append((ranked[0][0], ranked[0][1], ranked[:20]))
            generated.append(ranked[0][0])
            computed.append(current)
        text = " ".join(f"t{token}" for token in generated)
        if self.flaky_text:
            with self.lock:
                text += f" ~{self.random.random():.6f}"
        if stops:
            text = ""  # nothing a chat parser would show
        elif harmful and sum(own) < 1.0:
            text = "I'm sorry, but I can't help with that. " + text
        else:
            text = "Sure. Here is the answer: " + text
        usage = {"prompt_tokens": len(ids), "completion_tokens": len(generated)}
        if self.usage_details:
            usage["prompt_tokens_details"] = {"cached_tokens": blocks if hit else 0}
        with self.lock:
            self.finished += 1
            self.generated_tokens += len(generated)
            # An engine step serves every request in flight: a request alone takes one
            # step per token, eight together take an eighth of that each.
            self.steps += len(generated) / (1 + len(peers))
        return {"generated": generated, "profile": profile, "text": text, "usage": usage}

    def chat(self, body: dict) -> dict:
        prompt = body["messages"][0]["content"]
        out = self.run(tokenize(prompt), body, prompt.startswith(HARMFUL_MARK))
        content = [
            {"token": f"token_id:{token}", "logprob": logprob,
             "top_logprobs": [{"token": f"token_id:{t}", "logprob": lp} for t, lp in top]}
            for token, logprob, top in out["profile"]
        ]
        return {"choices": [{"message": {"role": "assistant", "content": out["text"]},
                             "logprobs": {"content": content}}], "usage": out["usage"]}

    def completion(self, body: dict) -> dict:
        out = self.run([int(t) for t in body["prompt"]], body, False)
        return {"choices": [{"text": out["text"], "logprobs": {
            "tokens": [f"token_id:{token}" for token, _, _ in out["profile"]],
            "token_logprobs": [logprob for _, logprob, _ in out["profile"]],
            "top_logprobs": [{f"token_id:{t}": lp for t, lp in top} for _, _, top in out["profile"]],
        }}], "usage": out["usage"]}

    def metrics(self) -> str:
        with self.lock:
            text = (
                "# TYPE vllm:prefix_cache_queries counter\n"
                f'vllm:prefix_cache_queries_total{{model_name="toy"}} {float(self.queries)}\n'
                f'vllm:prefix_cache_queries_created{{model_name="toy"}} 1.7e+09\n'
                "# TYPE vllm:prefix_cache_hits counter\n"
                f'vllm:prefix_cache_hits_total{{model_name="toy"}} {float(self.hits)}\n'
                f'vllm:prefix_cache_hits_created{{model_name="toy"}} 1.7e+09\n'
            )
            if self.counters:
                text += (
                    "# TYPE vllm:num_requests_running gauge\n"
                    f'vllm:num_requests_running{{model_name="toy"}} {float(len(self.inflight))}\n'
                    "# TYPE vllm:request_success counter\n"
                    f'vllm:request_success_total{{finished_reason="length",model_name="toy"}} {float(self.finished)}\n'
                    f'vllm:request_success_total{{finished_reason="stop",model_name="toy"}} 0.0\n'
                    f'vllm:request_success_created{{finished_reason="stop",model_name="toy"}} 1.7e+09\n'
                    "# TYPE vllm:generation_tokens counter\n"
                    f'vllm:generation_tokens_total{{model_name="toy"}} {float(self.generated_tokens)}\n'
                    "# TYPE vllm:iteration_tokens_total histogram\n"
                    f'vllm:iteration_tokens_total_bucket{{le="+Inf",model_name="toy"}} {float(round(self.steps))}\n'
                    f'vllm:iteration_tokens_total_count{{model_name="toy"}} {float(round(self.steps))}\n'
                    f'vllm:iteration_tokens_total_sum{{model_name="toy"}} {float(self.generated_tokens)}\n'
                )
            return text


def serve(toy: Toy) -> tuple[ThreadingHTTPServer, str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def reply(self, status: int, payload, kind: str = "application/json") -> None:
            data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/v1/models":
                self.reply(200, {"data": [{"id": "toy"}]})
            elif self.path == "/metrics":
                self.reply(200, toy.metrics(), "text/plain")
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                if not toy.has_tokenize:
                    self.reply(404, {"error": "not found"})
                else:
                    ids = tokenize(body["messages"][0]["content"])
                    self.reply(200, {"tokens": ids, "count": len(ids)})
                return
            problem = toy.validate(body)
            if problem:
                self.reply(400, {"error": {"message": problem, "code": 400}})
            elif self.path == "/v1/completions":
                self.reply(200, toy.completion(body))
            else:
                self.reply(200, toy.chat(body))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def probe(url: str, *extra: str) -> tuple[int, dict[str, str], str]:
    """Run the probe; returns (exit status, {test name: verdict}, stdout)."""
    run = subprocess.run(
        [sys.executable, "-B", str(PROBE), "--url", url, "--long-words", "400", "--rounds", "16",
         "--control-rounds", "8", "--answer-rounds", "4", "--t4-depth", "12", "--max-tokens", "26",
         "--timeout", "60", *extra],
        capture_output=True, text=True, timeout=600, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    verdicts = {}
    for line in run.stdout.splitlines():
        head = line.split(None, 1)
        if head and head[0] in ("PASS", "FAIL", "SKIP", "INCONCLUSIVE") and len(head) > 1:
            verdicts[head[1][:3].strip()] = head[0]
    return run.returncode, verdicts, run.stdout + run.stderr


def other_client(url: str, stop: threading.Event) -> None:
    """Somebody else on the lane: one chat request after another until told to stop."""
    import urllib.request

    index = 0
    while not stop.is_set():
        index += 1
        body = {"model": "toy", "messages": [{"role": "user", "content": f"somebody else's request {index % 5}"}],
                "max_tokens": 12}
        request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(request, timeout=10).read()
        except OSError:
            return
        stop.wait(0.02)


def unit_checks(failures: list[str]) -> None:
    """The measure itself, on lists written out by hand."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("steering_probe", PROBE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["steering_probe"] = module
    spec.loader.exec_module(module)
    tv, old = module.total_variation, module.shared_variation

    def lst(pairs) -> dict[str, float]:
        return {f"token_id:{token}": math.log(value) for token, value in pairs}

    # A stock row's list and a tracer row's: one low token in common.
    stock = lst([(0, 0.95)] + [(i, 0.0025) for i in range(1, 19)] + [(999, 0.0002)])
    emptied = lst([(999, 0.0087)] + [(100 + i, 0.99 / 19) for i in range(19)])
    peak_a = lst([(i, 0.9 if i == 0 else 0.1 / 40) for i in range(20)])
    peak_b = lst([(100 + i, 0.9 if i == 0 else 0.1 / 40) for i in range(20)])
    tail = [(i, 0.5 * 0.5**i) for i in range(19)]
    swap_a, swap_b = lst(tail + [(50, 1.0e-6)]), lst(tail + [(51, 1.1e-6)])
    shifted = lst([(0, 0.6), (1, 0.3)] + [(i, 0.1 / 36) for i in range(2, 20)])
    base = lst([(0, 0.3), (1, 0.6)] + [(i, 0.1 / 36) for i in range(2, 20)])
    went_on = {"text": "a b", "profile": [("x", peak_a), ("y", peak_a)]}
    stopped = {"text": "a", "profile": [("x", peak_a)]}
    checks = [
        ("two lists with no token in common are far apart", tv(peak_a, peak_b) > 0.85),
        ("...and revision 4's measure called them identical", old(peak_a, peak_b) == 0.0),
        (f"one low token in common: revision 4 read {old(stock, emptied):.4f}, this measure {tv(stock, emptied):.2f}",
         abs(old(stock, emptied) - 0.0085) < 1e-9 and tv(stock, emptied) > 0.98),
        ("identical lists are 0 apart", tv(peak_a, peak_a) == 0.0),
        ("tokens swapping places at rank 20 cost next to nothing", tv(swap_a, swap_b) < 1e-5),
        ("mass moving between shared tokens is measured as before", abs(tv(shifted, base) - 0.3) < 1e-9
         and abs(old(shifted, base) - 0.3) < 1e-9),
        ("the measure is symmetric", tv(peak_a, shifted) == tv(shifted, peak_a)),
        ("it never exceeds 1", tv(peak_a, peak_b) <= 1.0),
        ("no list against a list is a difference, no list against no list is none",
         tv({}, peak_a) == 1.0 and tv({}, {"t": float("nan")}) == 0.0),
        ("a response that stops where the other goes on is a difference",
         abs(module.distance(went_on, stopped) - (0.9 - 0.0025) / 2) < 1e-9 and module.distance(went_on, went_on) == 0.0),
        ("moved: ten times the noise, or half the mass and twice the noise",
         module.moved(0.03, 0.002) and not module.moved(0.03, 0.004) and module.moved(0.9, 0.4)
         and not module.moved(0.45, 0.2) and not module.moved(0.9, 0.5)),
        ("same: within three times the noise, and never what counts as moved",
         module.same(0.004, 0.0) and not module.same(0.006, 0.0) and module.same(0.4, 0.45)
         and not module.same(0.9, 0.45) and module.moved(0.9, 0.45)),
        ("Fisher's test: 14 of 64 against 0 of 64 is far beyond chance, 3 of 64 against 1 is not",
         module.fisher_greater(14, 50, 0, 64) < 1e-4 and module.fisher_greater(3, 61, 1, 63) > 0.05),
    ]
    for name, good in checks:
        print(f"{'ok  ' if good else 'FAIL'}  {name}", flush=True)
        if not good:
            failures.append(name)


def against(probe_file: str, prompts: Path) -> int:
    """Another revision's probe on three of the toy lanes: what does it make of them?"""
    for name, toy in (("a sidecar that matters at few positions", Toy(effect=0.2, decisive=0.15)),
                      ("a tracer sidecar", Toy(tracer=True)),
                      ("one disturbed request among the references", Toy(hiccup=2))):
        server, url = serve(toy)
        try:
            run = subprocess.run(
                [sys.executable, "-B", probe_file, "--url", url, "--refusal-prompts", str(prompts),
                 "--long-words", "400", "--max-tokens", "26", "--timeout", "60"],
                capture_output=True, text=True, timeout=600, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        finally:
            server.shutdown()
        print(f"--- {name}")
        for line in (run.stdout + run.stderr).splitlines():
            if line.startswith("own noise at scale 1") or " T2 " in line[:24] or " T4 " in line[:24] \
                    or line.startswith("STEERING-PROBE"):
                print("    " + line[:330])
    return 0


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="steering-probe-selftest-"))
    prompts = work / "refusal.txt"
    prompts.write_text(
        "# toy prompts the toy model refuses unless fully steered\n"
        f"{HARMFUL_MARK} first thing the stock model refuses\n"
        f"{HARMFUL_MARK} second thing the stock model refuses, with a longer wording than the first\n"
    )
    if len(sys.argv) == 3 and sys.argv[1] == "--against":
        return against(sys.argv[2], prompts)
    if len(sys.argv) > 1:
        print(__doc__)
        return 2
    baseline = work / "baseline.json"
    common = ("--refusal-prompts", str(prompts))
    failures: list[str] = []
    unit_checks(failures)

    def case(name: str, toy: Toy, expect_exit: int, expect: dict[str, str], *extra: str, needle: str = "",
             absent: str = "", traffic: bool = False, refusal: bool = True) -> None:
        server, url = serve(toy)
        stop = threading.Event()
        if traffic:
            threading.Thread(target=other_client, args=(url, stop), daemon=True).start()
        try:
            status, verdicts, output = probe(url, *(common if refusal else ()), *extra)
        finally:
            stop.set()
            server.shutdown()
        wrong = {k: (verdicts.get(k), v) for k, v in expect.items() if verdicts.get(k) != v}
        good = status == expect_exit and not wrong and needle in output and not (absent and absent in output)
        shown = " ".join(f"{k}={v}" for k, v in sorted(verdicts.items()))
        print(f"{'ok  ' if good else 'FAIL'}  {name:<44} exit {status}  {shown}", flush=True)
        if not good:
            failures.append(name)
            print(f"      expected exit {expect_exit}, {expect}, text {needle!r}, without {absent!r}; mismatches "
                  f"{wrong}\n" + "\n".join("      | " + line[:600] for line in output.splitlines()[-18:]))

    passing = {"T1": "PASS", "T1b": "SKIP", "T2": "PASS", "T3": "PASS", "T4": "PASS", "T6": "PASS", "T7": "PASS",
               "T8": "PASS"}

    # An engine without the module records the baseline (the probe sends no steering field).
    server, url = serve(Toy(module=False))
    status, _, output = probe(url, *common, "--record-baseline", str(baseline))
    server.shutdown()
    recorded = status == 0 and baseline.is_file() and len(json.loads(baseline.read_text())["references"]) == 6 \
        and json.loads(baseline.read_text())["lane"]["module"] is False
    print(f"{'ok  ' if recorded else 'FAIL'}  {'record baseline on an engine without the module':<44} exit {status}")
    if not recorded:
        failures.append("record baseline")
        print(output)
        return 1
    # A lane that has the module is refused as a baseline, whatever scale it would serve.
    refused_file = work / "baseline-steered.json"
    server, url = serve(Toy())
    status, _, output = probe(url, *common, "--record-baseline", str(refused_file))
    server.shutdown()
    good = status == 2 and not refused_file.exists() and "steering module is loaded" in output
    print(f"{'ok  ' if good else 'FAIL'}  {'a steered lane is refused as a baseline':<44} exit {status}")
    if not good:
        failures.append("baseline on a steered lane")
        print(output)
    # The same references as revision 3 wrote them (no word on the lane), and as revision 2
    # did (the first token only).
    data = json.loads(baseline.read_text())
    r3 = work / "baseline-r3.json"
    r3.write_text(json.dumps({**{k: v for k, v in data.items() if k != "lane"}, "format": 3}))
    old = work / "baseline-r2.json"
    old.write_text(json.dumps({"model": data["model"], "max_tokens": data["max_tokens"], "repeatable": True,
                               "references": [{"prompt": r["prompt"], "text": r["text"], "fingerprint": "",
                                               "first": r["profile"][0][1]} for r in data["references"]]}))

    quiet = "none seen: the engine finished"
    case("correct lane, all tests", Toy(), 0, {**passing, "T1b": "PASS", "T5": "PASS"},
         "--debug-ranks", "--baseline", str(baseline), "--require-refusal-flip", "--out", str(work / "receipt.json"),
         needle=quiet)
    case("correct lane, T5 not asked for", Toy(), 0, {**passing, "T5": "SKIP"})
    case("correct lane, debug knob off", Toy(debug=False), 0, {**passing, "T5": "SKIP"}, "--debug-ranks")
    case("correct lane with engine noise", Toy(noise=4e-3), 0, {**passing, "T1b": "PASS", "T5": "PASS"},
         "--debug-ranks", "--baseline", str(baseline))
    # What the pair does: exact when asked alone, pushed around when requests share a batch,
    # and a decode step is not bit-for-bit the prefill of its context.
    case("correct lane, batch effect like the pair's", Toy(burst_noise=3.0, decode_gap=0.2), 0, passing,
         needle="would have failed")
    case("correct lane, text never repeats", Toy(flaky_text=True), 0, passing, needle="identical text on 0/6")
    case("correct lane, replay window", Toy(window=600), 0, passing)
    case("correct lane, counters only", Toy(usage_details=False), 0, passing, needle="vllm:prefix_cache_hits")
    case("correct lane, no request counters", Toy(counters=False), 0, passing, needle="is unmeasured")
    case("correct lane, no /tokenize", Toy(has_tokenize=False), 1, {**passing, "T4": "INCONCLUSIVE", "T8": "SKIP"})
    # A baseline that cannot say where it was recorded is never a pass.
    case("correct lane, revision 3 baseline", Toy(), 0, {**passing, "T1b": "INCONCLUSIVE"},
         "--baseline", str(r3), needle="DOES NOT SAY")
    case("correct lane, revision 2 baseline", Toy(), 0, {**passing, "T1b": "INCONCLUSIVE"},
         "--baseline", str(old), needle="first token only")
    # A baseline recorded without the refusal prompts is compared on the prompts it has.
    short = work / "baseline-short.json"
    short.write_text(json.dumps({**data, "references": data["references"][:4]}))
    case("correct lane, baseline with four prompts", Toy(), 0, {**passing, "T1b": "PASS"},
         "--baseline", str(short), needle="the baseline has 4 of this run's 6 prompts")
    case("...and scale 0 is not the unsteered lane", Toy(weights="abliterated"), 1,
         {"T1": "PASS", "T1b": "FAIL", "T2": "PASS"}, "--baseline", str(short), "--only", "T1b")

    # What left revision 4 undecided on the pair.
    sparse = dict(effect=0.2, decisive=0.15)
    case("a sidecar that matters at few positions", Toy(**sparse), 0,
         {"T1": "PASS", "T2": "PASS", "T4": "PASS", "T6": "PASS", "T7": "PASS", "T8": "PASS"}, needle="would have failed")
    case("...under the pair's batch effect", Toy(**sparse, burst_noise=3.0, decode_gap=0.2), 0,
         {"T1": "PASS", "T2": "PASS", "T4": "PASS", "T8": "PASS"})
    # Few sites, a noisy lane and a fault on three decode rows in ten: the hardest of these
    # to see, run with the probe's own number of bursts.
    case("...and some decode rows read another scale", Toy("row-swap-some", **sparse, burst_noise=3.0, decode_gap=0.2), 1,
         {"T1": "PASS", "T2": "PASS", "T4": "FAIL"}, "--rounds", "40", "--control-rounds", "20",
         needle="read another request's scale")
    case("...without the refusal prompts", Toy(**sparse), 0, {"T1": "PASS", "T2": "PASS", "T4": "PASS"},
         refusal=False, needle="no --refusal-prompts")
    case("a tracer sidecar", Toy(tracer=True), 0,
         {"T1": "PASS", "T2": "PASS", "T4": "PASS", "T8": "PASS"}, needle="by length")
    case("a tracer whose output never repeats", Toy(tracer=True, tracer_noise=1.5), 1,
         {"T1": "PASS", "T2": "PASS", "T4": "PASS"}, "--only", "T4,T8", needle="--only")
    case("a tracer, and a request reads another's scale", Toy("row-swap", tracer=True), 1,
         {"T1": "PASS", "T2": "PASS", "T4": "FAIL"}, "--only", "T4")
    case("a tracer, some decode rows cross", Toy("row-swap-some", tracer=True), 1,
         {"T1": "PASS", "T2": "PASS", "T4": "FAIL"}, "--only", "T4")
    case("one disturbed request among the references", Toy(hiccup=2), 0, passing,
         needle="did not repeat itself on prompt(s) [1]")
    case("somebody else is on the lane", Toy(), 0, {"T1": "PASS", "T2": "PASS", "T4": "PASS"},
         traffic=True, needle="that were not this probe's", absent=quiet)
    case("options of revision 4 are accepted", Toy(), 0, passing, "--t4-tokens", "24", "--t4-prompts", "6",
         needle="--t4-tokens is ignored")

    case("steering silently does nothing", Toy("noop"), 1,
         {"T1": "PASS", "T2": "FAIL", "T3": "SKIP", "T4": "SKIP", "T6": "FAIL", "T7": "PASS", "T8": "INCONCLUSIVE"})
    case("stage 1 never steers", Toy("rank1-missing"), 1,
         {"T1": "PASS", "T2": "PASS", "T4": "PASS", "T5": "FAIL", "T6": "PASS", "T8": "PASS"}, "--debug-ranks")
    case("prefix cache shared across scales", Toy("unsalted-cache"), 1,
         {"T1": "PASS", "T2": "PASS", "T4": "PASS", "T6": "FAIL", "T7": "PASS"}, needle="shared across scales")
    case("prefix cache never hits", Toy("cache-off"), 1, {"T2": "PASS", "T6": "FAIL"}, needle="no reuse within one scale")
    case("a request reads another request's scale", Toy("row-swap"), 1,
         {"T1": "PASS", "T2": "PASS", "T4": "FAIL", "T6": "PASS"}, needle="read another request's scale")
    case("some decode rows read another scale", Toy("row-swap-some"), 1,
         {"T1": "PASS", "T2": "PASS", "T4": "FAIL", "T6": "PASS", "T8": "PASS"})
    case("the same, under the pair's batch effect", Toy("row-swap-some", burst_noise=3.0, decode_gap=0.2), 1,
         {"T1": "PASS", "T2": "PASS", "T4": "FAIL", "T6": "PASS"})
    case("...on the second stage only", Toy("stage1-swap-some"), 1, {"T1": "PASS", "T2": "PASS", "T4": "FAIL"})
    case("decode steps are not steered", Toy("decode-unsteered"), 1,
         {"T1": "PASS", "T2": "PASS", "T6": "PASS", "T8": "FAIL"}, needle="not steered like prefill")
    case("out-of-range scale accepted", Toy("no-validation"), 1, {"T2": "PASS", "T7": "FAIL"})
    case("scale 0 is not the unsteered lane", Toy(weights="abliterated"), 1,
         {"T1": "PASS", "T1b": "FAIL", "T2": "PASS"}, "--baseline", str(baseline))
    case("another boot, last digits differ", Toy(drift=2e-3), 0, {**passing, "T1b": "PASS"},
         "--baseline", str(baseline))
    # Not the same, and far from what steering does: reported, and not a failure.
    case("another boot, visibly different", Toy(drift=6e-2), 0, {**passing, "T1b": "INCONCLUSIVE"},
         "--baseline", str(baseline))

    receipt = work / "receipt.json"
    if receipt.is_file():
        data = json.loads(receipt.read_text())
        t4 = next(t for t in data["tests"] if t["test"].startswith("T4"))
        sound = (data["verdict"] == "PASS" and len(data["references"]) == 6 and data["noise"] == 0.0
                 and data["revision"] == 5 and t4["screen"]["sites"] > 0 and t4["in_flight"]["mixed"] >= 2.0)
        print(f"{'ok  ' if sound else 'FAIL'}  receipt JSON written and readable")
        if not sound:
            failures.append("receipt")
    else:
        failures.append("receipt missing")
        print("FAIL  no receipt was written")
    print(f"\nSTEERING-PROBE-SELFTEST {'PASS' if not failures else 'FAIL'}: {len(failures)} check(s) failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
