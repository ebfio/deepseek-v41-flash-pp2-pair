#!/usr/bin/env python3
"""Decode speed of a lane, taken apart: time to first token, tokens per second after it,
and what one speculative round yields and costs.

Standard library only; nothing is restarted and nothing but requests is sent:

    python3 verify/decode_rate.py --url http://<rank0>:8001 --streams 1,4 --tokens 1024

For each concurrency it streams that many chat completions of the same prompt (each with
its own cache_salt, so none is served from the prefix cache) and reports

    ttft        seconds to the first streamed token (median over streams)
    tok/s       per stream: (tokens - 1) / (time from the first token to the last);
                aggregate: all tokens after the first / the time they took together.
                The first chunk can carry a whole round's tokens, so both read high by
                at most one round in --tokens (under 1 % at 1024).
    per round   from the engine's own counters over the run: tokens per speculative
                round (1 = nothing accepted beyond the verified token) and milliseconds
                per round. A round is one draft-and-verify step of one request.

tok/s falls when a round gets slower or when it yields fewer tokens; this tells which.
"128 tokens in 0.70 s" does not: it has the first token's latency in it and no rounds.
The round counter is vllm:spec_decode_num_drafts, which counts a request's steps that
carried drafts. Where the engine skips drafting on some steps, their tokens still count,
so both per-round numbers then read high; vllm:spec_decode_num_draft_tokens / drafts in
the JSON says how many drafts a round carried.

Tokens per round depend on the text: a lane drafts a worked arithmetic answer far better
than an essay. Compare two lanes, or a lane with a published number, on the same prompt
only. --train-prompt sends the prompt this repository's README numbers are on (the one
in verify/texts_prod.py, with its top_p 0.95; use --tokens 512 as the README did), and
--prompt-file FILE sends the text of FILE. Without either, the prompt is the essay below.

--temperature T sends that temperature instead of 0, and --temperature server sends
neither temperature nor top_p, so the lane's own generation defaults apply: what a client
that sets nothing gets. Every published number of this repository is at temperature 0. A
sampled run takes another path through the drafter and the verifier, yields another
number of tokens per round, and its text differs from run to run: read the median and
the spread over --repeat, and compare two lanes at the same setting only.

--scale S sends steering_scale S with every request (a lane booted with steering).
--context-words N puts a ledger of about N words in front of the prompt, so that ttft
measures a cold prefill of that length (about 1.5 tokens a word; the script prints the
prompt's token count from the lane's own usage). Whether a run's prefill was cold is
reported, not assumed: from the usage of each response where the lane gives it
(prompt_tokens_details.cached_tokens), else from vllm:prompt_tokens_cached.
Run it on an otherwise idle lane: the counters are the engine's, so other traffic
during the run lands in the "per round" numbers (the script says when the counters
moved by more tokens than it received).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid

# verify/texts_prod.py: the prompt the README's single-stream tok/s and step time are on.
TRAIN_PROMPT = (
    "A train leaves at 60 mph. Two hours later a second train leaves the same station at 90 mph on a "
    "parallel track. At what distance does the second catch the first? Reason step by step before answering."
)
PROMPT = (
    "Write a detailed, numbered walk through the history of timekeeping, from sundials and "
    "water clocks to pendulum clocks, marine chronometers, quartz and atomic clocks. Give "
    "each device a paragraph: how it works, what limited its accuracy, what it made possible. "
    "Do not stop before you have covered at least twelve devices."
)


def context(words: int) -> str:
    """Numbered ledger lines, no two alike, about `words` words."""
    rng = random.Random(1017)
    things = ["harbor", "lantern", "granite", "willow", "copper", "meadow", "signal", "ember", "orchard", "ledger"]
    lines, count = [], 0
    while count < words:
        a, b = rng.sample(things, 2)
        line = (f"Entry {len(lines) + 1}: the {a} was logged at {rng.randrange(24):02d}:{rng.randrange(60):02d} "
                f"with {rng.randrange(2, 900)} units and stored beside the {b}.")
        lines.append(line)
        count += len(line.split())
    return "Here is a ledger you may ignore.\n" + "\n".join(lines) + "\n\n"


COUNTERS = {
    "drafts": "vllm:spec_decode_num_drafts",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens",
    "accepted": "vllm:spec_decode_num_accepted_tokens",
    "generated": "vllm:generation_tokens",
    "cached": "vllm:prompt_tokens_cached",
}


def counters(url: str, timeout: float) -> dict[str, float | None]:
    """Each counter summed over its label sets; None when the lane does not export it."""
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=timeout) as response:
            text = response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return dict.fromkeys(COUNTERS)
    found: dict[str, list[float]] = {key: [] for key in COUNTERS}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        sample = line.split("{", 1)[0].split(" ", 1)[0]
        for key, name in COUNTERS.items():
            if sample in (name, name + "_total"):
                found[key].append(float(line.rsplit(" ", 1)[1]))
    return {key: (sum(values) if values else None) for key, values in found.items()}


def stream_one(url: str, model: str, tokens: int, scale: float | None, timeout: float, prompt: str,
               top_p: float | None = None, temperature: float | None = 0) -> dict:
    """One streamed chat completion: {first, last, tokens, finish} with wall-clock times.
    temperature None sends no temperature: the lane's own default applies."""
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": uuid.uuid4().hex,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if top_p is not None:
        body["top_p"] = top_p
    if scale is not None:
        body["vllm_xargs"] = {"steering_scale": scale}
    request = urllib.request.Request(
        url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    sent = time.perf_counter()
    first = last = None
    completion_tokens = prompt_tokens = cached_tokens = None
    finish = None
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {exc.code} from /v1/chat/completions: {detail}") from None
    with response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("error"):
                raise RuntimeError(f"the lane answered with an error in the stream: {str(chunk['error'])[:300]}")
            if chunk.get("usage"):
                completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
                prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                details = chunk["usage"].get("prompt_tokens_details") or {}
                if details.get("cached_tokens") is not None:
                    cached_tokens = int(details["cached_tokens"])
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if any(delta.get(key) for key in ("content", "reasoning_content", "reasoning")):
                    last = time.perf_counter()
                    if first is None:
                        first = last
                finish = choice.get("finish_reason") or finish
    if first is None or completion_tokens is None:
        raise RuntimeError("the lane streamed no tokens or no usage (stream_options.include_usage)")
    return {"sent": sent, "first": first, "last": last, "tokens": int(completion_tokens), "finish": finish,
            "prompt_tokens": prompt_tokens, "cached_tokens": cached_tokens}


def measure(url: str, model: str, streams: int, tokens: int, scale: float | None, timeout: float,
            prompt: str, top_p: float | None = None, temperature: float | None = 0) -> dict:
    before = counters(url, timeout)
    with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
        futures = [pool.submit(stream_one, url, model, tokens, scale, timeout, prompt, top_p, temperature)
                   for _ in range(streams)]
        runs = [future.result() for future in futures]
    # The engine hands a step's output to the streams before it records that step's
    # counters (vllm/v1/engine/async_llm.py:919, then :944): let the last step land.
    time.sleep(0.3)
    after = counters(url, timeout)
    per_stream = [(r["tokens"] - 1) / (r["last"] - r["first"]) for r in runs if r["last"] > r["first"]]
    decode_start, decode_end = min(r["first"] for r in runs), max(r["last"] for r in runs)
    received = sum(r["tokens"] for r in runs)
    if decode_end <= decode_start:
        raise RuntimeError("every token arrived in one chunk: there is no decode interval to time "
                           "(is the lane streaming token by token?)")
    out = {
        "streams": streams,
        "prompt_tokens": runs[0]["prompt_tokens"],
        "tokens": received,
        "ttft_s": sorted(r["first"] - r["sent"] for r in runs)[len(runs) // 2],
        "tok_s_per_stream": sorted(per_stream)[len(per_stream) // 2] if per_stream else None,
        "tok_s_aggregate": (received - streams) / (decode_end - decode_start),
        # All tokens over all the time, the first token's latency included: what a
        # "tokens / wall seconds" number measures.
        "tok_s_with_ttft": sorted(r["tokens"] / (r["last"] - r["sent"]) for r in runs)[len(runs) // 2],
        "short": sum(r["tokens"] < tokens for r in runs),
    }
    delta = {key: (after[key] - before[key]) if None not in (after[key], before[key]) else None for key in COUNTERS}
    out["counters"] = delta
    # Prompt tokens that came from the prefix cache: per response where the lane says so,
    # else the counter's movement over the run (which other traffic moves too).
    if all(r["cached_tokens"] is not None for r in runs):
        out["cached_prompt_tokens"], out["cached_source"] = sum(r["cached_tokens"] for r in runs), "usage"
    elif delta["cached"] is not None:
        out["cached_prompt_tokens"], out["cached_source"] = int(round(delta["cached"])), "vllm:prompt_tokens_cached"
    else:
        out["cached_prompt_tokens"], out["cached_source"] = None, None
    if delta["drafts"]:
        generated = delta["generated"] if delta["generated"] is not None else received
        out["tokens_per_round"] = generated / delta["drafts"]
        out["accepted_per_round"] = (delta["accepted"] or 0.0) / delta["drafts"]
        # Each stream runs its own rounds side by side: rounds per stream over the decode time.
        out["ms_per_round"] = 1000.0 * (decode_end - decode_start) / (delta["drafts"] / streams)
        out["other_traffic"] = generated > 1.05 * received + 8
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--url", required=True, help="API base, e.g. http://10.0.42.51:8001")
    parser.add_argument("--model", help="served model name (default: first of /v1/models)")
    parser.add_argument("--streams", default="1,4", help="concurrencies to measure, in order")
    parser.add_argument("--tokens", type=int, default=1024, help="max_tokens of each stream")
    parser.add_argument("--repeat", type=int, default=3, help="runs per concurrency; the median is reported")
    parser.add_argument("--scale", type=float, help="steering_scale sent with every request")
    parser.add_argument("--context-words", type=int, default=0, help="words of ledger in front of the prompt")
    parser.add_argument("--train-prompt", action="store_true",
                        help="the README's prompt (verify/texts_prod.py) with its top_p 0.95; use --tokens 512")
    parser.add_argument("--prompt-file", help="send the text of this file as the prompt")
    parser.add_argument("--top-p", type=float, help="top_p sent with every request (default: not sent)")
    parser.add_argument("--temperature", default="0",
                        help="temperature sent with every request (default 0), or 'server': send neither "
                             "temperature nor top_p, so the lane's own generation defaults apply")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--json", help="write every run as JSON to this file ('-' = stdout)")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    model = args.model
    if model is None:
        with urllib.request.urlopen(url + "/v1/models", timeout=args.timeout) as response:
            model = json.loads(response.read())["data"][0]["id"]
    if args.train_prompt and args.prompt_file:
        parser.error("--train-prompt and --prompt-file are two different prompts")
    server_sampling = args.temperature.strip().lower() == "server"
    try:
        temperature = None if server_sampling else float(args.temperature)
    except ValueError:
        parser.error("--temperature takes a number or the word server")
    if temperature is not None and temperature < 0:
        parser.error("--temperature must not be negative")
    top_p, which = args.top_p, "the essay prompt"
    text = PROMPT
    if args.train_prompt:
        text, which = TRAIN_PROMPT, "the README's train prompt"
        if not server_sampling:  # with the lane's own defaults nothing is added to the request
            top_p = 0.95 if top_p is None else top_p
    elif args.prompt_file:
        with open(args.prompt_file, encoding="utf-8") as handle:
            text, which = handle.read().strip(), f"the prompt in {args.prompt_file}"
    prompt = (context(args.context_words) if args.context_words else "") + text
    settle = stream_one(url, model, 16, args.scale, args.timeout, prompt, top_p, temperature)  # settles the lane
    print(f"lane {model} at {url}; {which}, {settle['prompt_tokens']} prompt tokens, max_tokens {args.tokens}, "
          f"temperature {'not sent (the lane decides)' if temperature is None else f'{temperature:g}'}, "
          f"top_p {'not sent' if top_p is None else top_p}, "
          f"steering_scale {'not sent' if args.scale is None else args.scale}; median of {args.repeat} runs")
    if temperature != 0:
        print("sampled requests: the text differs from run to run and tokens per round with it. Read the "
              "median and the spread, and compare with another lane at this same setting only")
    print(f"{'streams':>7} {'ttft s':>7} {'tok/s per stream':>17} {'tok/s aggregate':>16} {'with ttft':>10} "
          f"{'tokens/round':>13} {'ms/round':>9}  notes")
    everything = []
    for streams in [int(v) for v in args.streams.split(",")]:
        runs = [measure(url, model, streams, args.tokens, args.scale, args.timeout, prompt, top_p, temperature)
                for _ in range(args.repeat)]
        everything.append(runs)
        mid = sorted(runs, key=lambda r: r["tok_s_aggregate"])[len(runs) // 2]
        spread = (max(r["tok_s_aggregate"] for r in runs) - min(r["tok_s_aggregate"] for r in runs)) / mid["tok_s_aggregate"]
        notes = [f"runs within {100 * spread:.1f}%"]
        cached = [r["cached_prompt_tokens"] for r in runs]
        if None in cached:
            notes.append("prefill: the lane does not report cached prompt tokens")
        elif not any(cached):
            notes.append("prefill cold in every run")
        else:
            notes.append(f"prefill NOT cold: {max(cached)} prompt tokens from the prefix cache in one run "
                         f"({runs[0]['cached_source']})")
        if mid["short"]:
            notes.append(f"{mid['short']} stream(s) stopped before max_tokens")
        if "tokens_per_round" not in mid:
            notes.append("no speculative-decoding counters on this lane: per-round numbers unmeasured")
        elif mid.get("other_traffic"):
            notes.append("the counters moved by more tokens than this run received: other traffic")
        show = lambda key, form: format(mid[key], form) if mid.get(key) is not None else "-"  # noqa: E731
        print(f"{streams:>7} {mid['ttft_s']:>7.3f} {show('tok_s_per_stream', '17.1f'):>17} {mid['tok_s_aggregate']:>16.1f} "
              f"{mid['tok_s_with_ttft']:>10.1f} {show('tokens_per_round', '13.2f'):>13} {show('ms_per_round', '9.2f'):>9}  "
              f"{'; '.join(notes)}")
    if args.json:
        payload = json.dumps({"url": url, "model": model, "tokens": args.tokens, "scale": args.scale, "prompt": which,
                              "top_p": top_p, "temperature": "server" if temperature is None else temperature,
                              "context_words": args.context_words, "runs": everything}, indent=2)
        if args.json == "-":
            print(payload)
        else:
            with open(args.json, "w", encoding="utf-8") as handle:
                handle.write(payload)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, urllib.error.URLError) as exc:
        print(f"DECODE-RATE ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
