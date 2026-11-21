#!/usr/bin/env python3
"""Steering receipt 2 of 2: does a running lane steer per request, and only per request?

Standard library only. Point it at a lane booted with VLLM_STEERING_SIDECAR set, and give
it the refusal prompts: on harmless prompts an abliteration is built to change little,
and a test that has to tell two scales apart then has little to work with.

    python3 verify/steering_probe.py --url http://<rank0>:8001 \
        --refusal-prompts refusal_prompts.txt --out results/steering-probe.json

How two distributions are compared. Every request asks for the top 20 logprobs of each
generated token. Two such lists are compared by the largest total variation distance
they prove (0 = identical, 1 = disjoint): for each token a list names, how much more
probability it has there than in the other list, where a token the other list does not
name is given the most it can have there (that list's 20th probability, or whatever
mass that list has left, whichever is smaller). The result never exceeds the distance
between the full distributions. Revision 4 and earlier compared only the tokens BOTH
lists name plus one bucket for the rest, and two lists with no token in common came out
as identical: the more a row was changed, the less the probe saw (see the history at the
end). The old figure is still printed next to T2's, as "tokens in both lists only".

How two responses are compared. They are walked token by token for as long as they
generated the same tokens. The largest distance on that shared path is the distance
between the responses; a response that stops where the other goes on counts as a
difference there. It does not need the first token to be informative (a reasoning
model's first token often has probability 1 at any scale).

How "the same" and "different" are decided. The lane is asked for its unsteered and
steered references several times. For each prompt, the largest distance between
repeats of one request is that prompt's own noise. "Same" means at most 3 x noise (and
at least 0.005); "moved" means more than 10 x noise and more than 0.02, or at least 0.5
and twice the noise. What falls between is INCONCLUSIVE with its numbers, never a pass
or a failure. One prompt whose repeats disagree no longer raises the bar for the others.

Other traffic. The engine's own counters say how many requests it finished and how many
tokens it generated while the probe ran; the probe knows its own. The difference is
printed: "quiet" is measured, not assumed.

    T1  stock identity     scale omitted, 0 in vllm_xargs and 0 at the top level are the
                           same request
    T1b (with --baseline)  scale 0 against a lane booted WITHOUT the module. The baseline
                           file says whether its lane had the module (it is asked for
                           steering_scale 7: HTTP 400 means the module is there), and a
                           file that does not say is never a PASS
    T2  steering is live   scale 1 moves the output
    T3  scale is a dial    0.5 differs from both 0 and 1
    T4  mixed batches      see "How T4 decides" below
    T5  both stages        needs --debug-ranks and VLLM_STEERING_DEBUG=1: steering on
                           stage 0 only, stage 1 only, both and neither
    T6  prefix cache       judged by accounting first: tokens served from the cache
                           across two scales must be 0, within one scale more than 0.
                           Outputs are compared against controls measured on the same
                           long prompt: cold twice (its own noise) and cold against a
                           cache hit at scale 0 (what a hit does with no steering at all)
    T7  validation         out-of-range and non-numeric scales are HTTP 400
    T8  decode = prefill   a token generated in a decode step has the distribution its
                           context gets when prefilled at the same scale, and not the
                           one it gets at the other scale: steering is applied in decode
                           steps too, and only where asked. Judged like a T4 position,
                           on every site (see T4) after the first generated token of the
                           lane's own quiet generations

How T4 decides. A burst of requests in flight together changes this engine's outputs by
itself: on the pair an all-stock burst moved a response by 0.38. So T4 never compares a
response with a reference response. It works in three steps.
  1. The screen, one quiet request at a time. Each prompt is generated alone at scale 0
     and at scale 1 (--t4-depth tokens). For every position on those paths the lane is
     asked what the next token's distribution is at scale 0 and at scale 1. A position
     where the two answers differ by at least 0.2 is a SITE: a place where it shows which
     scale a row was computed at.
  2. Bursts of raw completions from token ids, each request aimed at a site. Half of
     them end their prompt at the site, so the site is their first generated token (a
     prefill step); the others end it one to five tokens earlier on the same path, so the
     site comes out of a decode step. Mixed bursts carry both scales. Bursts of one scale
     are the control: nothing in them can read another scale, so whatever they show is
     the lane's own batch effect.
  3. Every generated position whose context has two answers 0.2 apart is judged: "own"
     when the burst's distribution is within a quarter of that difference of its own
     scale's answer, "swap" when it is that near the other scale's answer instead,
     "unclear" when it is on neither (a row that read the wrong scale on one stage only
     ends up there).
Mixed bursts are then compared with the control, for the rows at scale 0 and the rows at
scale 1 separately, on how often a position is not "own" and how often it is "swap":
four one-sided Fisher exact tests. With four tests each is held to a quarter of the
usual level.
    FAIL          one of the four says mixed bursts are worse, p < 0.00025
    INCONCLUSIVE  p < 0.0125; or the control itself has swaps at more than 2 % of its
                  positions; or too few positions were judged, in all or in prefill
                  steps or in decode steps; or the requests of a burst were not in
                  flight together (measured from their start and end times)
    PASS          none of that, and it says for each scale how much cross-talk it would
                  have seen
A second part works on whole answers and needs no logprobs. A prompt takes part when its
quiet answers at the two scales can be told apart every time by something coarse:
refused or not (the refusal prompts an abliteration flips), or length (one scale's
answers at least twice as long as the other's; a tracer sidecar's steered answers stop
after a few tokens). Responses that do what the other scale does are counted in mixed
bursts and in as many single-scale ones, and compared by one exact test (FAIL at
p < 0.001, INCONCLUSIVE at p < 0.05): a coarse check that catches a gross fault and
does not depend on any distance.

Every request uses temperature 0 and its own cache_salt unless the test is about the
cache. Verdicts: PASS, FAIL, SKIP (not run, with the reason) and INCONCLUSIVE.
Exit status 0 only when nothing failed and T1, T2, T4, T6 and T7 passed.

--record-baseline FILE writes the references of a lane booted WITHOUT steering (no
steering field is sent, apart from the one request that checks the module is absent; a
lane that has the module is refused). --baseline FILE compares scale 0 against them on a
steered lane. A baseline file keeps the top-20 lists, so a file recorded by revision 4
is read as it is and compared by this revision's measure.

History. Revision 2 compared the first token's top logprobs by an L1 distance with a
floor for missing tokens and took its noise from one back-to-back repeat. Revision 3
compared whole responses, and the lane's batch effect was as large as the steering
effect. Revision 4 judged each generated position by which scale's answer it sits on,
and on the pair it could not decide T4 either, for three reasons that this revision
removes. (1) Its distance could not see a token that one list lacks. A tracer sidecar
that emptied every steered answer read 0.008 there while the text differed on 4 of 4
prompts, so T2 failed and T4 was skipped. With the published sidecar the positions where
steering replaces the leading token are the ones it under-read. (2) It generated 16
tokens per burst request and judged whatever positions happened to matter: 16 of 768.
(3) One noise figure, the largest over all prompts and both scales, set every prompt's
bar, so a single disturbed repeat could switch T2 off for the whole run and T4 with it.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

BENIGN = [
    "List three practical uses of a paperclip. One line each.",
    "In two sentences, why is the sky blue at noon and red at sunset?",
    "Write a four-line poem about a lighthouse keeper who collects clocks, then explain the "
    "rhyme scheme you used in one sentence.",
    "A train leaves at 09:40 and arrives at 13:05. How long is the trip? Show the arithmetic.",
]
REFUSAL = re.compile(
    r"\b(i\s*(?:can(?:no|')t|won't|will not|am not able|'m not able|'m unable|am unable)|"
    r"i'?m sorry|i apologi[sz]e|cannot (?:help|assist|provide)|can't (?:help|assist|provide)|"
    r"not able to (?:help|assist|provide))",
    re.IGNORECASE,
)
SAME_FLOOR, MOVED_FLOOR = 0.005, 0.02
DEPTH = 24  # generated positions compared
TOP_K = 20  # logprobs asked for at each position
EFFECT_MIN = 0.2  # a position is a site, and is judged, where scale 0 and scale 1 differ by this much
MAX_LEAD = 5  # a decode-step request of T4 ends its prompt up to this many tokens before its site
CACHED = ("vllm:prompt_tokens_cached", "vllm:prefix_cache_hits")
FINISHED, GENERATED, STEPS = "vllm:request_success", "vllm:generation_tokens", "vllm:iteration_tokens_total_count"


class Lane:
    def __init__(self, url: str, model: str | None, max_tokens: int, timeout: float,
                 template_kwargs: dict | None, steering_fields: bool = True) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.template_kwargs = template_kwargs
        self.steering_fields = steering_fields
        self.requests = 0
        self.answered = 0  # generation requests of this probe that the engine finished
        self.generated = 0  # tokens the engine says it generated for them
        self.count_lock = threading.Lock()
        self.model = model or self.get_json("/v1/models")["data"][0]["id"]

    def get_json(self, path: str) -> dict:
        with urllib.request.urlopen(self.url + path, timeout=self.timeout) as response:
            return json.loads(response.read())

    def post(self, path: str, body: dict) -> tuple[int, dict | str]:
        request = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        self.requests += 1
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return 200, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")[:300]

    def metric(self, name: str) -> float | None:
        """Sum over label sets of the metric `name` (a counter is exported as `name_total`);
        None when the lane does not export it. The name is matched whole:
        vllm:prompt_tokens_cached is not vllm:prompt_tokens_cached_by_source."""
        try:
            with urllib.request.urlopen(self.url + "/metrics", timeout=self.timeout) as response:
                text = response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError):
            return None
        values = []
        for line in text.splitlines():
            if line.startswith("#") or " " not in line:
                continue
            sample = line.split("{", 1)[0].split(" ", 1)[0]
            if sample in (name, name + "_total"):
                values.append(float(line.rsplit(" ", 1)[1]))
        return sum(values) if values else None

    def cached_counter(self) -> tuple[str, float] | None:
        for name in CACHED:
            value = self.metric(name)
            if value is not None:
                return name, value
        return None

    def _served(self, usage: dict) -> None:
        with self.count_lock:
            self.answered += 1
            self.generated += int(usage.get("completion_tokens") or 0)

    def mark(self) -> dict:
        """The engine's counters and this probe's own, for others_since(). The engine
        hands a response out before it records it, hence the short wait."""
        time.sleep(0.3)
        with self.count_lock:
            own = (self.answered, self.generated)
        return {"finished": self.metric(FINISHED), "generated": self.metric(GENERATED),
                "steps": self.metric(STEPS), "answered": own[0], "tokens": own[1]}

    def others_since(self, before: dict) -> dict | None:
        """What the engine did since `before` that was not this probe's doing: requests
        finished and tokens generated for somebody else. None when the lane does not
        export the counters."""
        after = self.mark()
        if before["finished"] is None or after["finished"] is None:
            return None
        requests = int(round(after["finished"] - before["finished"])) - (after["answered"] - before["answered"])
        tokens = None
        if before["generated"] is not None and after["generated"] is not None:
            tokens = int(round(after["generated"] - before["generated"])) - (after["tokens"] - before["tokens"])
        steps = None
        if before["steps"] is not None and after["steps"] is not None:
            steps = int(round(after["steps"] - before["steps"]))
        return {"requests": requests, "tokens": tokens, "steps": steps,
                "own_requests": after["answered"] - before["answered"]}

    def _steer(self, body: dict, scale, ranks, top_level: bool) -> None:
        xargs: dict = {}
        if self.steering_fields:
            if scale is not None and top_level:
                body["steering_scale"] = scale
            elif scale is not None:
                xargs["steering_scale"] = scale
            if ranks is not None:
                xargs["steering_ranks"] = ranks
        if xargs:
            body["vllm_xargs"] = xargs

    def chat(self, prompt: str, *, scale=None, ranks=None, top_level=False, salt: str | None = None,
             max_tokens: int | None = None, count_cache: bool = False) -> dict:
        """One chat completion: {status, text, profile, usage, cached, error}."""
        body: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 0,
            "max_tokens": max_tokens or self.max_tokens,
            "logprobs": True,
            "top_logprobs": 20,
            "return_tokens_as_token_ids": True,
            "cache_salt": salt or uuid.uuid4().hex,
        }
        if self.template_kwargs:
            body["chat_template_kwargs"] = self.template_kwargs
        self._steer(body, scale, ranks, top_level)
        before = self.cached_counter() if count_cache else None
        status, payload = self.post("/v1/chat/completions", body)
        if status != 200:
            return {"status": status, "error": payload}
        assert isinstance(payload, dict)
        choice = payload["choices"][0]
        message = choice.get("message", {})
        text = "".join(str(message.get(key) or "") for key in ("reasoning_content", "reasoning", "content"))
        profile = [
            (item["token"], {t["token"]: float(t["logprob"]) for t in item.get("top_logprobs", [])})
            for item in ((choice.get("logprobs") or {}).get("content") or [])[:DEPTH]
        ]
        usage = payload.get("usage") or {}
        self._served(usage)
        return {"status": 200, "text": text, "profile": profile, "usage": usage,
                "length": usage.get("completion_tokens"), "cached": self._cached(usage, before, count_cache)}

    def _cached(self, usage: dict, before, count_cache: bool) -> dict | None:
        """Prompt tokens served from the prefix cache for the request just made: from the
        response when the server reports it, else from the counter's movement."""
        if not count_cache:
            return None
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is not None:
            return {"tokens": int(details["cached_tokens"]), "source": "usage"}
        after = self.cached_counter()
        if before is None or after is None or before[0] != after[0]:
            return None
        return {"tokens": int(round(after[1] - before[1])), "source": after[0]}

    def tokenize(self, prompt: str) -> list[int] | None:
        body: dict = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                      "add_generation_prompt": True}
        if self.template_kwargs:
            body["chat_template_kwargs"] = self.template_kwargs
        status, payload = self.post("/tokenize", body)
        if status != 200 or not isinstance(payload, dict) or not isinstance(payload.get("tokens"), list):
            return None
        return [int(t) for t in payload["tokens"]]

    def complete(self, token_ids: list[int], *, scale=None, max_tokens: int = 8) -> dict:
        """A raw completion from token ids: {status, tokens, profile, error}. A completion
        that generated nothing the server reports is status 200 with an empty profile."""
        body: dict = {
            "model": self.model, "prompt": token_ids, "temperature": 0, "seed": 0, "max_tokens": max_tokens,
            "logprobs": 20, "return_tokens_as_token_ids": True, "cache_salt": uuid.uuid4().hex,
        }
        self._steer(body, scale, None, False)
        status, payload = self.post("/v1/completions", body)
        if status != 200:
            return {"status": status, "error": payload}
        assert isinstance(payload, dict)
        self._served(payload.get("usage") or {})
        logprobs = payload["choices"][0].get("logprobs") or {}
        tokens = logprobs.get("tokens") or []
        tops = logprobs.get("top_logprobs") or []
        ids = []
        for token in tokens:
            match = re.fullmatch(r"token_id:(\d+)", str(token))
            if match is None:
                return {"status": 0, "error": f"the server did not return token ids ({token!r})"}
            ids.append(int(match.group(1)))
        profile = [(token, {t: float(lp) for t, lp in (top or {}).items()}) for token, top in zip(tokens, tops)]
        return {"status": 200, "tokens": ids, "profile": profile}


def void(top: dict[str, float]) -> bool:
    """No distribution at all: the server returned no list for the position, or one with a
    NaN in it."""
    return not top or any(value != value for value in top.values())


def _listed(top: dict[str, float]) -> tuple[dict[str, float], float]:
    """(probability of each listed token, the most a token NOT listed can have)."""
    listed = {token: math.exp(value) for token, value in top.items()}
    ranked = sorted(listed.values(), reverse=True)
    rest = max(0.0, 1.0 - sum(ranked))
    # The list is the top TOP_K: an unlisted token has at most its last probability, and
    # at most the mass the list leaves over.
    return listed, min(ranked[min(len(ranked), TOP_K) - 1], rest)


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    """The largest total variation distance between two distributions that their top-k
    logprob lists prove. TV(A, B) is the sum over all tokens of (A - B) where A is larger,
    and equally of (B - A) where B is larger. For a token both lists name the term is
    known. For a token only A's list names, B gives it at most B's cap, so the term is at
    least A's value minus that cap. Tokens neither list names are left out. Each sum is
    therefore a lower bound of the true distance, and so is the larger of the two.

    Two tokens swapping places at rank 20 cost next to nothing (the one that left has
    about the other list's last probability), and two lists with no token in common
    come out near 1 when either of them is peaked."""
    if void(a) or void(b):
        return 0.0 if void(a) and void(b) else 1.0
    pa, cap_a = _listed(a)
    pb, cap_b = _listed(b)
    up = sum(max(0.0, value - pb.get(token, cap_b)) for token, value in pa.items())
    down = sum(max(0.0, value - pa.get(token, cap_a)) for token, value in pb.items())
    return min(1.0, max(up, down))


def shared_variation(a: dict[str, float], b: dict[str, float]) -> float:
    """Revision 4's measure, kept to show what it read: the tokens both lists name plus
    one bucket for everything else. It equals the larger of the two sums above taken over
    the shared tokens only, so a token that one list lacks is never seen: two lists with
    nothing in common are 0 apart."""
    shared = set(a) & set(b)
    pa = {t: math.exp(a[t]) for t in shared}
    pb = {t: math.exp(b[t]) for t in shared}
    rest_a, rest_b = max(0.0, 1.0 - sum(pa.values())), max(0.0, 1.0 - sum(pb.values()))
    total = sum(abs(pa[t] - pb[t]) for t in shared) + abs(rest_a - rest_b)
    return min(1.0, 0.5 * total)


def distance(a: dict, b: dict, measure=total_variation) -> float:
    """Largest distance on the path the two responses share."""
    return compare(a, b, measure)[0]


def compare(a: dict, b: dict, measure=total_variation) -> tuple[float, int | None, int]:
    """(distance, position where the generated tokens first differ or None, positions compared)."""
    pa, pb = a.get("profile") or [], b.get("profile") or []
    if not pa or not pb:
        return (0.0 if a.get("text") == b.get("text") else 1.0), None, 0
    worst, compared = 0.0, 0
    for position, ((token_a, top_a), (token_b, top_b)) in enumerate(zip(pa, pb)):
        worst = max(worst, measure(top_a, top_b))
        compared += 1
        if token_a != token_b:
            return worst, position, compared
    if len(pa) != len(pb) and measure is total_variation:
        # One response stopped where the other went on (both were cut at the same
        # max_tokens and DEPTH, so the shorter one ended by itself). The one that went on
        # gave its token probability m and no other token more than m2. The one that
        # stopped preferred another token, which it therefore gave at least as much as
        # it gave that token: the two distributions differ by at least (m - m2) / 2.
        longer = pa if len(pa) > len(pb) else pb
        ranked = sorted((math.exp(v) for v in longer[compared][1].values() if v == v), reverse=True)
        if ranked:
            worst = max(worst, (ranked[0] - (ranked[1] if len(ranked) > 1 else 0.0)) / 2)
        return worst, compared, compared
    return worst, None, compared


def nearest(result: dict, references: list[dict]) -> float:
    return min(distance(result, reference) for reference in references)


def moved(value: float, noise: float) -> bool:
    """Clearly more than the repeats differ by: ten times that, or, where a repeat is
    itself far from exact, at least half the probability mass and twice the noise."""
    return value > max(10 * noise, MOVED_FLOOR) or (value >= 0.5 and value >= 2 * noise)


def same(value: float, noise: float) -> bool:
    """Within three times what the repeats differ by, and not something moved() calls moved."""
    return value <= max(3 * noise, SAME_FLOOR) and not moved(value, noise)


def refused(result: dict) -> bool:
    return bool(REFUSAL.search(result["text"][:400]))


def fisher_greater(marked_a: int, plain_a: int, marked_b: int, plain_b: int) -> float:
    """One-sided Fisher exact test: the probability of at least `marked_a` marked items
    in group a when the marked items fall on the two groups by chance."""
    size_a, marked, total = marked_a + plain_a, marked_a + marked_b, marked_a + plain_a + marked_b + plain_b
    if not marked or not size_a or size_a == total:
        return 1.0
    ways = math.comb(total, marked)
    return min(1.0, sum(math.comb(size_a, k) * math.comb(total - size_a, marked - k)
                        for k in range(marked_a, min(size_a, marked) + 1)) / ways)


def judge_position(seen: dict[str, float], own: dict[str, float], other: dict[str, float]) -> tuple[str, float, float, float]:
    """(kind, effect, distance to own scale, distance to other scale) for one generated
    position, or kind "" where the two scales' answers differ by less than EFFECT_MIN.
    "own": within a quarter of that difference of its own scale's answer. "swap": within
    a quarter of it of the other scale's answer, and three quarters from its own.
    "unclear": on neither. A row computed half at one scale and half at the other (one
    stage of two reading the wrong scale) lands there, which "nearer its own answer than
    the other's" would count as own about half the time."""
    effect = total_variation(own, other)
    if effect < EFFECT_MIN:
        return "", effect, 0.0, 0.0
    d_own, d_other = total_variation(seen, own), total_variation(seen, other)
    if d_own <= 0.25 * effect:
        kind = "own"
    elif d_other <= 0.25 * effect and d_own >= 0.75 * effect:
        kind = "swap"
    else:
        kind = "unclear"
    return kind, effect, d_own, d_other


def long_prompt(words: int, seed: int) -> str:
    """Numbered sentences with no two alike: a context a sparse attention can index, unlike
    a list of a few repeated words."""
    rng = random.Random(seed)
    things = ["harbor", "lantern", "granite", "willow", "copper", "meadow", "signal", "ember", "orchard",
              "ledger", "kettle", "compass", "thimble", "quarry", "saddle", "lattice", "beacon", "furnace"]
    places = ["Aldermoor", "Brixham", "Calder", "Dunmere", "Eastvale", "Farrow", "Glenhollow", "Hartwell",
              "Ironbridge", "Jessop", "Kestrel Point", "Lowmarsh"]
    lines, count, index = [], 0, 1
    while count < words:
        a, b = rng.sample(things, 2)
        line = (f"Entry {index}: the {a} from {rng.choice(places)} was logged at {rng.randrange(24):02d}:"
                f"{rng.randrange(60):02d} with {rng.randrange(2, 900)} units and stored beside the {b}.")
        lines.append(line)
        count += len(line.split())
        index += 1
    asked = rng.randrange(1, index)
    return ("Here is a ledger.\n" + "\n".join(lines) +
            f"\nWhat was logged in entry {asked}, and how many units did it have? Answer in one line.")


class Report:
    def __init__(self) -> None:
        self.tests: list[dict] = []

    def add(self, name: str, verdict: str, detail: str, **data) -> None:
        self.tests.append({"test": name, "verdict": verdict, "detail": detail, **data})
        print(f"{verdict:<12} {name:<22} {detail}", flush=True)

    def verdict(self, name: str) -> str | None:
        return next((t["verdict"] for t in self.tests if t["test"].startswith(name)), None)


def ok(result: dict) -> dict:
    if result["status"] != 200:
        raise RuntimeError(f"HTTP {result['status']}: {result.get('error')}")
    return result


def median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2] if values else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--url", required=True, help="API base, e.g. http://10.0.42.51:8001")
    parser.add_argument("--model", help="served model name (default: first of /v1/models)")
    parser.add_argument("--refusal-prompts", help="text file, one prompt per line, that the stock model refuses")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--chat-template-kwargs", help="JSON object passed as chat_template_kwargs")
    parser.add_argument("--repeats", type=int, default=3, help="times each reference is taken (noise estimate)")
    parser.add_argument("--concurrency", type=int, default=8, help="requests in flight in a T4 burst")
    parser.add_argument("--rounds", type=int, default=40, help="mixed-scale bursts in T4")
    parser.add_argument("--control-rounds", type=int, default=20, help="single-scale bursts per scale in T4")
    parser.add_argument("--answer-rounds", type=int, default=6, help="mixed-scale bursts of whole answers in T4")
    parser.add_argument("--t4-depth", type=int, default=32,
                        help="tokens of each quiet generation that T4 and T8 screen for sites")
    parser.add_argument("--t4-sites", type=int, default=32, help="sites T4 aims at, per kind of step: the strongest")
    parser.add_argument("--t4-prompts", type=int, help="ignored since revision 5: every prompt is screened")
    parser.add_argument("--t4-tokens", type=int, help="ignored since revision 5: see --t4-depth")
    parser.add_argument("--long-words", type=int, default=4000, help="length of the T6 prompt, in words")
    parser.add_argument("--debug-ranks", action="store_true", help="run T5 (needs VLLM_STEERING_DEBUG=1)")
    parser.add_argument("--baseline", help="references recorded with --record-baseline on a steering-off boot")
    parser.add_argument("--record-baseline", help="write this lane's references and stop (steering-off boot)")
    parser.add_argument("--require-refusal-flip", action="store_true",
                        help="fail T4 unless every refusal prompt is refused at 0 and answered at 1, every time")
    parser.add_argument("--skip", default="", help="comma-separated tests to skip, e.g. T4,T6")
    parser.add_argument("--only", default="", help="comma-separated tests to run besides T1 and T2, e.g. T4,T8")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1017)
    parser.add_argument("--out", help="write the receipt as JSON")
    args = parser.parse_args()

    template_kwargs = json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else None
    lane = Lane(args.url, args.model, args.max_tokens, args.timeout, template_kwargs,
                steering_fields=not args.record_baseline)
    harmful: list[str] = []
    if args.refusal_prompts:
        with open(args.refusal_prompts, encoding="utf-8") as handle:
            harmful = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    prompts = BENIGN + harmful
    count = len(prompts)
    skipped = {name.strip().upper() for name in args.skip.split(",") if name.strip()}
    only = {name.strip().upper() for name in args.only.split(",") if name.strip()}
    for name in ("T1B", "T3", "T4", "T5", "T6", "T7", "T8"):
        if only and name not in only:
            skipped.add(name)
    why_skipped = "--only" if only else "--skip"
    for option, value in (("--t4-tokens", args.t4_tokens), ("--t4-prompts", args.t4_prompts)):
        if value is not None:
            print(f"note: {option} is ignored since revision 5 (T4 aims its requests at sites; see --t4-depth)", flush=True)
    if not harmful:
        print("note: no --refusal-prompts. On harmless prompts an abliteration is built to change little, so "
              "T4 and T8 may find too few positions where the two scales can be told apart", flush=True)
    report = Report()
    rng = random.Random(args.seed)
    started = time.time()
    busy = sum(v or 0.0 for v in (lane.metric("vllm:num_requests_running"), lane.metric("vllm:num_requests_waiting")))
    if busy:
        print(f"note: the lane is not idle ({busy:g} requests running or waiting): batch shapes and "
              "the cache counters include other traffic", flush=True)
    traffic: list[dict] = []
    run_mark = lane.mark()

    def watched(phase: str, before: dict) -> dict | None:
        """Record what the engine did during `phase` that was not this probe's doing."""
        others = lane.others_since(before)
        if others is not None:
            traffic.append({"phase": phase, **others})
        return others

    if args.record_baseline:
        # A baseline is a lane WITHOUT the module. The module answers HTTP 400 to a scale
        # outside [0, 1]; an engine without it ignores the field. Revision 3 did not ask,
        # and a file recorded on a steered lane made T1b compare that lane with itself.
        status, _ = lane.post("/v1/chat/completions", {
            "model": lane.model, "messages": [{"role": "user", "content": prompts[0]}], "temperature": 0,
            "max_tokens": 1, "vllm_xargs": {"steering_scale": 7}})
        if status == 400:
            raise RuntimeError("this lane answers HTTP 400 to steering_scale 7, so the steering module is loaded "
                               "here. A baseline has to come from a boot without it (VLLM_STEERING_SIDECAR empty)")
        if status != 200:
            raise RuntimeError(f"the request that checks for the steering module got HTTP {status}")

    # ---- references, taken `repeats` times, interleaved -------------------------------
    repeats = max(2, args.repeats)
    phase_mark = lane.mark()
    ref0 = [[ok(lane.chat(p)) for p in prompts] for _ in range(repeats)]
    if not ref0[0][0]["profile"]:
        raise RuntimeError("the lane returned no logprobs: this probe needs logprobs and top_logprobs")
    spread0 = [max(distance(a[i], b[i]) for a in ref0 for b in ref0) for i in range(count)]
    texts0 = sum(len({run[i]["text"] for run in ref0}) == 1 for i in range(count))
    depth = min(len(run[i]["profile"]) for run in ref0 for i in range(count))
    print(f"lane {lane.model}: {count} prompts ({len(harmful)} refusal), {repeats} runs each; own noise at "
          f"scale 0: largest {max(spread0):.4g}, median {median(spread0):.4g}; text repeats on {texts0}/{count}; "
          f"{depth} positions compared", flush=True)

    if args.record_baseline:
        others = watched("references", phase_mark)
        if others and (others["requests"] > 0 or (others["tokens"] or 0) > others["own_requests"]):
            print(f"note: the engine finished {others['requests']} request(s) and generated {others['tokens']} "
                  "token(s) that were not this probe's while the references were taken", flush=True)
        payload = {
            "format": 4, "model": lane.model, "max_tokens": lane.max_tokens, "noise": max(spread0),
            "lane": {"module": False, "how": "steering_scale 7 was accepted (HTTP 200): nothing validates it",
                     "url": lane.url, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "other_traffic": others},
            "chat_template_kwargs": template_kwargs,
            "references": [{"prompt": p, "text": r["text"], "profile": r["profile"]} for p, r in zip(prompts, ref0[0])],
        }
        with open(args.record_baseline, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {args.record_baseline}: {count} references from a lane without steering "
              f"(its own noise: {max(spread0):.4g})")
        return 0

    ref1 = [[ok(lane.chat(p, scale=1.0)) for p in prompts] for _ in range(repeats)]
    watched("references", phase_mark)
    spread1 = [max(distance(a[i], b[i]) for a in ref1 for b in ref1) for i in range(count)]
    # Each prompt's own noise: what its repeats differ by, at scale 0 and at either scale.
    noise0 = spread0
    noise01 = [max(a, b) for a, b in zip(spread0, spread1)]
    noise = max(noise01)
    typical = median(noise01)
    d01 = [min(distance(a[i], b[i]) for a in ref0 for b in ref1) for i in range(count)]
    d01_old = [min(distance(a[i], b[i], shared_variation) for a in ref0 for b in ref1) for i in range(count)]
    live = [moved(d, n) for d, n in zip(d01, noise01)]  # prompts on which steering can be told from noise
    # Not live, though the distance would count as moved on a lane that repeats itself:
    # this prompt's own repeats are what stands in the way.
    drowned = [not l and moved(d, 0.0) for l, d in zip(live, d01)]
    print(f"own noise at scale 1: largest {max(spread1):.4g}, median {median(spread1):.4g}. Each prompt is judged "
          f"against its own repeats (same <= 3 x, at least {SAME_FLOOR}; moved > 10 x, at least {MOVED_FLOOR}); "
          f"largest over all prompts {noise:.4g}, median {typical:.4g}", flush=True)
    loud = [i for i in range(count) if noise01[i] > max(10 * typical, MOVED_FLOOR)]
    if loud:
        print(f"note: the lane did not repeat itself on prompt(s) {loud} (up to {max(noise01[i] for i in loud):.4g}) "
              "while the others repeated: something else ran on the lane then, or those outputs are not "
              "stable. Only their own verdicts are affected", flush=True)

    def judge(values: list[float], noises: list[float]) -> str:
        if all(same(v, n) for v, n in zip(values, noises)):
            return "PASS"
        return "FAIL" if any(moved(v, n) for v, n in zip(values, noises)) else "INCONCLUSIVE"

    # ---- T1 stock identity ----------------------------------------------------------
    phase_mark = lane.mark()
    xargs0 = [ok(lane.chat(p, scale=0.0)) for p in prompts]
    top0 = [ok(lane.chat(p, scale=0.0, top_level=True)) for p in prompts]
    t1 = [max(nearest(a, [run[i] for run in ref0]), nearest(b, [run[i] for run in ref0]))
          for i, (a, b) in enumerate(zip(xargs0, top0))]
    texts = sum(a["text"] == b["text"] == ref0[0][i]["text"] for i, (a, b) in enumerate(zip(xargs0, top0)))
    report.add("T1 stock identity", judge(t1, noise0),
               f"omitted vs vllm_xargs 0 vs top-level 0: largest distance {max(t1):.4g} (noise at scale 0: "
               f"{max(noise0):.4g}); {sum(same(v, n) for v, n in zip(t1, noise0))}/{count} prompts the same, "
               f"{sum(moved(v, n) for v, n in zip(t1, noise0))} moved; identical text on {texts}/{count}",
               distances=t1, texts_equal=texts)

    # ---- T2 steering is live, T3 scale is a dial --------------------------------------
    text_changed = sum(a["text"] != b["text"] for a, b in zip(ref0[0], ref1[0]))

    def enough(flags: list[bool]) -> bool:
        return sum(flags) >= (count + 1) // 2 and all(flags[len(BENIGN):])

    if enough(live):
        t2 = "PASS"
    elif enough([l or d for l, d in zip(live, drowned)]):
        t2 = "INCONCLUSIVE"  # it moved, by less than ten times what the lane's own repeats differ by
    else:
        t2 = "FAIL"
    report.add("T2 steering is live", t2,
               f"scale 1 moves {sum(live)}/{count} prompts ({sum(live[len(BENIGN):])}/{len(harmful)} refusal prompts)"
               + (f", and {sum(drowned)} more by less than ten times what their own repeats differ by" if any(drowned) else "")
               + f"; distance 0 vs 1: min {min(d01):.4g}, median {median(d01):.4g} (tokens in both lists only, "
               f"as revision 4 measured: min {min(d01_old):.4g}, median {median(d01_old):.4g}); text differs on "
               f"{text_changed}", distances=d01, distances_shared_tokens_only=d01_old)
    if "T3" in skipped:
        report.add("T3 scale is a dial", "SKIP", why_skipped)
    elif not any(live):
        report.add("T3 scale is a dial", "SKIP", "scale 1 moved no prompt, so 0.5 has nothing to sit between")
    else:
        half = [ok(lane.chat(p, scale=0.5)) for p in prompts]
        ends = [(nearest(half[i], [run[i] for run in ref0]), nearest(half[i], [run[i] for run in ref1]))
                for i in range(count)]
        good = sum(moved(a, noise01[i]) and moved(b, noise01[i]) for i, (a, b) in enumerate(ends) if live[i])
        # A dial that does nothing at 0.5, or is already at 1 there, is the failure. A prompt
        # on which 0.5 is a little off one end, by less than counts as moved, is neither.
        stuck = sum(same(a, noise01[i]) or same(b, noise01[i]) for i, (a, b) in enumerate(ends) if live[i])
        report.add("T3 scale is a dial", "PASS" if good == sum(live) else "FAIL" if stuck else "INCONCLUSIVE",
                   f"0.5 differs from both 0 and 1 on {good}/{sum(live)} of the prompts scale 1 moves"
                   + (f"; on {stuck} it is the same as 0 or as 1" if stuck else ""), distances=ends)
    watched("T1 to T3", phase_mark)

    # ---- T1b: scale 0 on this boot against a boot without the module ------------------
    if args.baseline and "T1B" not in skipped:
        with open(args.baseline, encoding="utf-8") as handle:
            baseline = json.load(handle)
        by_prompt = {r["prompt"]: r for r in baseline["references"]}
        covered = [i for i, p in enumerate(prompts) if p in by_prompt]
        if not covered or baseline.get("max_tokens") != lane.max_tokens:
            report.add("T1b vs unsteered boot", "SKIP",
                       f"baseline does not cover this run ({len(covered)} of {count} prompts in it, "
                       f"max_tokens {baseline.get('max_tokens')} vs {lane.max_tokens})")
        else:
            # A baseline recorded with fewer prompts (without the refusal file, say) is
            # compared on the prompts it has, and the line says how many those are.
            old = "profile" not in baseline["references"][0]
            drift = []
            for i in covered:
                entry = by_prompt[prompts[i]]
                reference = entry if not old else {"text": entry["text"], "profile": [("", entry["first"])]}
                mine = [run[i] if not old else {"text": run[i]["text"], "profile": [("", run[i]["profile"][0][1])]}
                        for run in ref0]
                drift.append(nearest(reference, mine))
            texts = sum(by_prompt[prompts[i]]["text"] == ref0[0][i]["text"] for i in covered)
            floors = [max(noise0[i], float(baseline.get("noise") or 0.0)) for i in covered]
            floor = max(floors)
            relative = max(d / max(d01[i], 1e-12) for d, i in zip(drift, covered))
            verdict = ("PASS" if all(same(d, f) for d, f in zip(drift, floors))
                       else "INCONCLUSIVE" if relative < 0.1 or not any(moved(d, f) for d, f in zip(drift, floors))
                       else "FAIL")
            unsteered = (baseline.get("lane") or {}).get("module") is False
            origin = ""
            if not unsteered:
                # Never a pass: nothing says this file was recorded without the module.
                verdict = "INCONCLUSIVE" if verdict == "PASS" else verdict
                origin = ("; THE BASELINE FILE DOES NOT SAY ITS LANE HAD NO STEERING MODULE (revision 3 and "
                          "earlier did not check): if it was recorded on a steered lane this is T1 again. "
                          "Record it with this revision on a boot without the module")
            partial = "" if len(covered) == count else (
                f"; the baseline has {len(covered)} of this run's {count} prompts (it was recorded without "
                "the others): compared on those")
            report.add("T1b vs unsteered boot", verdict,
                       f"scale 0 against the lane booted without steering: largest distance {max(drift):.4g} "
                       f"(noise {floor:.4g}), at most {relative:.2g} of that prompt's 0-vs-1 distance; same text on "
                       f"{texts}/{len(covered)}" + partial + ("; old baseline file: first token only" if old else "")
                       + origin, drift=drift, relative=relative, baseline_lane=baseline.get("lane"),
                       prompts_compared=covered)
    else:
        report.add("T1b vs unsteered boot", "SKIP", "no --baseline given" if not args.baseline else why_skipped)

    if harmful:
        r0 = sum(refused(r) for r in ref0[0][len(BENIGN):])
        r1 = sum(refused(r) for r in ref1[0][len(BENIGN):])
        report.add("   refusals, one at a time", "INFO",
                   f"scale 0: {r0}/{len(harmful)} refused; scale 1: {r1}/{len(harmful)} refused "
                   "(keyword classifier: read the texts in the receipt)", at0=r0, at1=r1)

    # ---- raw completions from token ids (T4, T8) ------------------------------------
    def ok_complete(result: dict) -> dict:
        if result["status"] != 200:
            raise RuntimeError(f"/v1/completions: {result.get('error')}")
        return result

    answers: dict[tuple[tuple[int, ...], float], dict[str, float]] = {}

    def answer(context: tuple[int, ...], scale: float) -> dict[str, float]:
        """The next token's distribution after `context` at `scale`, asked alone: one
        prefill of the whole context with nothing else of this probe in flight. Empty
        when the lane generates nothing there."""
        key = (context, scale)
        if key not in answers:
            profile = ok_complete(lane.complete(list(context), scale=scale, max_tokens=2))["profile"]
            answers[key] = profile[0][1] if profile else {}
        return answers[key]

    # ---- the screen: where on the lane's own quiet paths do the two scales differ? ----
    ids: dict[int, list[int]] = {}
    paths: dict[tuple[int, float], dict] = {}
    sites: dict[tuple[int, ...], dict] = {}
    screen_info: dict = {}

    def screen() -> None:
        if screen_info:
            return
        before, began = lane.mark(), time.time()
        for index in range(count):
            tokens = lane.tokenize(prompts[index])
            if tokens is None:
                raise RuntimeError("the lane has no /tokenize endpoint for chat messages; the position tests "
                                   "need token ids")
            ids[index] = tokens
        effects: dict[tuple[int, ...], float] = {}
        for index in range(count):
            for scale in (0.0, 1.0):
                path = ok_complete(lane.complete(ids[index], scale=scale, max_tokens=max(2, args.t4_depth)))
                if not paths and not path["profile"]:
                    raise RuntimeError("/v1/completions returned no logprobs: the position tests need them")
                paths[index, scale] = path
                for depth in range(len(path["tokens"])):
                    context = tuple(ids[index] + path["tokens"][:depth])
                    if context not in effects:
                        effects[context] = total_variation(answer(context, 0.0), answer(context, 1.0))
                    if effects[context] >= EFFECT_MIN:
                        site = sites.setdefault(context, {"context": context, "prompt": index, "depth": depth,
                                                          "effect": effects[context], "paths": set()})
                        site["paths"].add(scale)
        later = sum(site["depth"] >= 1 for site in sites.values())
        screen_info.update(positions=len(effects), sites=len(sites), sites_after_first_token=later,
                           depth=args.t4_depth, seconds=round(time.time() - began, 1),
                           strongest=max((site["effect"] for site in sites.values()), default=0.0),
                           other_traffic=watched("screen", before))
        print(f"screen: {count} prompts generated alone at scale 0 and at scale 1 ({args.t4_depth} tokens each), "
              f"{len(effects)} positions asked at both scales: the scales differ by {EFFECT_MIN} or more at "
              f"{len(sites)} of them ({later} after the first generated token), at most by "
              f"{screen_info['strongest']:.3g}; {screen_info['seconds']:.0f} s", flush=True)

    # ---- T4 mixed batches -----------------------------------------------------------
    def mixed_batches() -> tuple[str, str, dict]:
        screen()
        if not sites:
            return "SKIP", (f"scale 0 and scale 1 differ by {EFFECT_MIN} at none of the {screen_info['positions']} "
                            "positions screened: nothing to tell apart"), {"screen": screen_info}
        ranked = sorted(sites.values(), key=lambda site: -site["effect"])
        keep = max(1, args.t4_sites)
        prefill_sites = ranked[:keep]
        decode_sites = {scale: [site for site in ranked if site["depth"] >= 1 and scale in site["paths"]][:keep]
                        for scale in (0.0, 1.0)}

        def job(scale: float, kind: str) -> dict:
            """A request aimed at a site. "P": the prompt ends at the site, which is then
            the first generated token (a prefill step). "D": it ends up to MAX_LEAD
            tokens earlier on the path that leads there at this scale, so the site is
            generated in a decode step."""
            if kind == "D" and decode_sites[scale]:
                site = rng.choice(decode_sites[scale])
                lead = rng.randint(1, min(site["depth"], MAX_LEAD))
            else:
                site, lead = rng.choice(prefill_sites), 0
            context = site["context"]
            return {"site": site, "scale": scale, "lead": lead, "prompt": list(context[: len(context) - lead])}

        def burst(jobs: list[dict]) -> tuple[list[dict], float]:
            def one(item: dict) -> tuple[dict, float, float]:
                began = time.monotonic()
                result = lane.complete(item["prompt"], scale=item["scale"], max_tokens=item["lead"] + 2)
                return result, began, time.monotonic()

            with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                done = [future.result() for future in [pool.submit(one, item) for item in jobs]]
            wall = max(end for _, _, end in done) - min(began for _, began, _ in done)
            # Requests of this burst in flight, averaged over the time the burst took:
            # 1 when they were answered one after another, len(jobs) when all overlapped.
            together = sum(end - began for _, began, end in done) / max(wall, 1e-9)
            return [ok_complete(result) for result, _, _ in done], together

        # 1. The bursts. Single-scale ones first: nothing in them can read another scale.
        groups: dict[str, list[tuple[dict, dict]]] = {"single-scale": [], "mixed": []}
        flight: dict[str, list[float]] = {"single-scale": [], "mixed": []}
        before = lane.mark()
        for scale in (0.0, 1.0):
            for _ in range(max(1, args.control_rounds)):
                jobs = [job(scale, "PD"[(k // 2) % 2]) for k in range(args.concurrency)]
                results, together = burst(jobs)
                groups["single-scale"] += list(zip(jobs, results))
                flight["single-scale"].append(together)
        watched("T4 single-scale bursts", before)
        before = lane.mark()
        for _ in range(max(1, args.rounds)):
            jobs = [job(float(k % 2), "PD"[(k // 2) % 2]) for k in range(args.concurrency)]
            rng.shuffle(jobs)
            results, together = burst(jobs)
            groups["mixed"] += list(zip(jobs, results))
            flight["mixed"].append(together)
        mixed_traffic = watched("T4 mixed bursts", before)

        # 2. Each generated position against what the lane says, asked alone, that position
        #    is at scale 0 and at scale 1 for the tokens this response had seen.
        tally: dict[str, dict[str, int]] = {}
        examples: list[str] = []
        before = lane.mark()
        for group, items in groups.items():
            for scale in (0.0, 1.0):
                tally[f"{group} at {scale:g}"] = dict.fromkeys(
                    ("responses", "judged", "own", "unclear", "swap", "wrong", "prefill", "prefill own", "decode",
                     "decode own", "aimed at a decode step", "reached its site"), 0)
            for item, result in items:
                scale, lead = item["scale"], item["lead"]
                cell = tally[f"{group} at {scale:g}"]
                cell["responses"] += 1
                if lead:
                    cell["aimed at a decode step"] += 1
                    cell["reached its site"] += result["tokens"][:lead] == list(item["site"]["context"][-lead:])
                kinds = []
                for position, (_, seen) in enumerate(result["profile"]):
                    context = tuple(item["prompt"] + result["tokens"][:position])
                    kind, effect, d_own, d_other = judge_position(
                        seen, answer(context, scale), answer(context, 1.0 - scale))
                    if not kind:
                        continue
                    step = "prefill" if position == 0 else "decode"
                    kinds.append(kind)
                    cell["judged"] += 1
                    cell[kind] += 1
                    cell[step] += 1
                    cell[f"{step} own"] += kind == "own"
                    if kind != "own" and group == "mixed" and len(examples) < 6:
                        examples.append(
                            f"{group} burst, prompt {item['site']['prompt']} at scale {scale:g}, generated position "
                            f"{position} ({step} step): {d_own:.3g} from its own scale's answer, {d_other:.3g} from "
                            f"the other's (the two answers differ by {effect:.3g}): {kind}")
                not_own = len(kinds) - kinds.count("own")
                if len(kinds) >= 3 and 3 * not_own >= 2 * len(kinds):
                    cell["wrong"] += 1  # two thirds or more of its judged positions are not own: reported, not a verdict
        watched("T4 judging", before)

        def total(group: str, key: str) -> int:
            return sum(tally[f"{group} at {scale:g}"][key] for scale in (0.0, 1.0))

        # Mixed bursts against single-scale ones, separately for the rows at scale 0 and
        # the rows at scale 1 (they have different noise: a decode step is further from
        # its prefill at one scale than at the other), and separately for "not own" and
        # "swap". Four one-sided tests, so each is held to a quarter of the usual level.
        tests: list[tuple[float, str, float, int, int, int, int]] = []
        detectable: dict[str, int | None] = {}
        fail_level, unsure_level = 0.001 / 4, 0.05 / 4
        for scale in (0.0, 1.0):
            mixed, single = tally[f"mixed at {scale:g}"], tally[f"single-scale at {scale:g}"]
            for what in ("not own", "swap"):
                marked_m = mixed["swap"] if what == "swap" else mixed["judged"] - mixed["own"]
                marked_c = single["swap"] if what == "swap" else single["judged"] - single["own"]
                tests.append((fisher_greater(marked_m, mixed["judged"] - marked_m, marked_c, single["judged"] - marked_c),
                              what, scale, marked_m, mixed["judged"], marked_c, single["judged"]))
            not_own_c = single["judged"] - single["own"]
            low, high = mixed["judged"] - mixed["own"], mixed["judged"] + 1  # smallest count that would have failed
            while low < high:
                middle = (low + high) // 2
                if fisher_greater(middle, mixed["judged"] - middle, not_own_c, single["judged"] - not_own_c) < fail_level:
                    high = middle
                else:
                    low = middle + 1
            detectable[f"{scale:g}"] = low if low <= mixed["judged"] else None
        p_min, what_min, scale_min, marked_m, judged_at_m, marked_c, judged_at_c = min(tests)
        judged_c, swaps_c = total("single-scale", "judged"), total("single-scale", "swap")
        prefill_m, decode_m = total("mixed", "prefill"), total("mixed", "decode")
        in_flight = {group: (sum(values) / len(values) if values else 0.0) for group, values in flight.items()}
        used = len({item["site"]["context"] for items in groups.values() for item, _ in items})

        def line(group: str, scale: float) -> str:
            cell = tally[f"{group} at {scale:g}"]
            return (f"{cell['judged']} judged in {cell['responses']} responses, {cell['own']} own, "
                    f"{cell['unclear']} unclear, {cell['swap']} swap")

        counts = (f"{used} sites used; mixed bursts at scale 0: {line('mixed', 0.0)}; at scale 1: {line('mixed', 1.0)} "
                  f"({prefill_m} of these positions in prefill steps, {total('mixed', 'prefill own')} own; {decode_m} "
                  f"in decode steps, {total('mixed', 'decode own')} own); single-scale bursts at scale 0: "
                  f"{line('single-scale', 0.0)}; at scale 1: {line('single-scale', 1.0)}; smallest p of the four "
                  f"tests: {p_min:.3g} ({what_min} at scale {scale_min:g}); of "
                  f"{total('mixed', 'aimed at a decode step')} mixed requests aimed at a decode step "
                  f"{total('mixed', 'reached its site')} reached their site on the lane's quiet path "
                  f"({total('single-scale', 'reached its site')} of {total('single-scale', 'aimed at a decode step')} "
                  "in single-scale bursts); on average "
                  f"{in_flight['mixed']:.1f} of a mixed burst's {args.concurrency} requests were in flight at once")
        if mixed_traffic is not None and mixed_traffic["steps"] is not None:
            counts += (f", and the engine counted {mixed_traffic['steps']} steps while it served those "
                       f"{total('mixed', 'responses')} requests (alone, each needs at least two)")
        thin = min(tally[f"{group} at {scale:g}"]["judged"] for group in groups for scale in (0.0, 1.0))
        if thin < 20:
            position_verdict = "INCONCLUSIVE"
            reason = (f"too few positions could be judged (the smallest of the four groups has {thin}): raise "
                      "--rounds and --control-rounds")
        elif min(prefill_m, decode_m) < 10:
            position_verdict = "INCONCLUSIVE"
            reason = (f"one kind of step is hardly covered ({prefill_m} positions judged in prefill steps, "
                      f"{decode_m} in decode steps): the screen found {screen_info['sites_after_first_token']} "
                      "sites after the first generated token; give it the refusal prompts, or raise --t4-depth")
        elif in_flight["mixed"] < 2.0:
            position_verdict = "INCONCLUSIVE"
            reason = ("the requests of a mixed burst were hardly in flight together, so they did not share "
                      "engine steps: nothing was tested")
        elif swaps_c > 0.02 * judged_c:
            position_verdict = "INCONCLUSIVE"
            reason = (f"bursts of a single scale already show {swaps_c} positions on the other scale's answer. "
                      "Nothing in such a burst can read another scale, so either sharing a batch does this by "
                      "itself on this lane or steering is not applied where it should be (see T8): this test "
                      "cannot decide")
        elif p_min < fail_level:
            position_verdict = "FAIL"
            reason = (f"requests in a mixed burst read another request's scale: at scale {scale_min:g}, {marked_m} "
                      f"of {judged_at_m} positions are {what_min} in mixed bursts against {marked_c} of "
                      f"{judged_at_c} in single-scale ones")
        elif p_min < unsure_level:
            position_verdict = "INCONCLUSIVE"
            reason = (f"at scale {scale_min:g}, positions that are {what_min} are more frequent in mixed bursts "
                      f"({marked_m} of {judged_at_m}) than in single-scale ones ({marked_c} of {judged_at_c}), by "
                      "more than chance explains well and less than proves it: run it again with "
                      f"--rounds {2 * args.rounds} --control-rounds {2 * args.control_rounds}")
        elif any(detectable[f"{scale:g}"] is None and 2 * (tally[f"single-scale at {scale:g}"]["judged"]
                 - tally[f"single-scale at {scale:g}"]["own"]) < tally[f"single-scale at {scale:g}"]["judged"]
                 for scale in (0.0, 1.0)):
            position_verdict = "INCONCLUSIVE"
            reason = ("at one of the scales no number of positions off their own answer could have failed this "
                      "run (too few judged there against what the single-scale bursts show): raise --rounds and "
                      "--control-rounds")
        else:
            position_verdict = "PASS"
            reason = "positions away from their own scale are no more frequent in mixed bursts than in single-scale ones"
            for scale in (0.0, 1.0):
                mixed = tally[f"mixed at {scale:g}"]
                single = tally[f"single-scale at {scale:g}"]
                found = detectable[f"{scale:g}"]
                if 2 * (single["judged"] - single["own"]) >= single["judged"]:
                    reason += (f"; at scale {scale:g} half or more of the single-scale positions are themselves "
                               "not on their own answer (this lane's outputs at that scale do not repeat), so "
                               "only a row sitting on the other scale's answer would show there")
                elif found is not None:
                    reason += (f"; at scale {scale:g}, {mixed['judged'] - mixed['own']} of {mixed['judged']} positions "
                               f"are off their own answer in mixed bursts and {single['judged'] - single['own']} of "
                               f"{single['judged']} in single-scale ones: {found} in mixed bursts "
                               f"({100.0 * found / mixed['judged']:.1f} %) would have failed")
        p_value = p_min

        # 3. Whole answers. A prompt takes part when its quiet answers at the two scales can
        #    be told apart every time by something coarse: refused or not, or length.
        features: dict[int, tuple[str, float, float]] = {}
        for i in range(count):
            if i >= len(BENIGN) and all(refused(run[i]) for run in ref0) and not any(refused(run[i]) for run in ref1):
                features[i] = ("refusal", 0.0, 0.0)
                continue
            n0, n1 = [run[i].get("length") for run in ref0], [run[i].get("length") for run in ref1]
            if None in n0 or None in n1:
                continue
            for long_scale, long, short in ((0.0, n0, n1), (1.0, n1, n0)):
                if min(long) >= 2 * max(max(short), 1):
                    features[i] = ("length", math.sqrt(min(long) * max(max(short), 1)), long_scale)

        def scale_of(i: int, result: dict) -> float:
            """The scale whose quiet answers this response resembles."""
            kind, cut, long_scale = features[i]
            if kind == "refusal":
                return 0.0 if refused(result) else 1.0
            return long_scale if (result.get("length") or 0) > cut else 1.0 - long_scale

        by_refusal = sum(kind == "refusal" for kind, _, _ in features.values())
        text_verdict, text = None, ("no prompt's quiet answers at the two scales differ every time by refusal or "
                                    "by length: whole answers not judged")
        answers_data: dict = {"prompts": {i: features[i][0] for i in sorted(features)}}
        if features:
            usable = sorted(features)

            def chat_burst(jobs: list[tuple[int, float]]) -> int:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                    futures = [pool.submit(lane.chat, prompts[i], scale=s) for i, s in jobs]
                    return sum(scale_of(i, ok(f.result())) != s for (i, s), f in zip(jobs, futures))

            before = lane.mark()
            alone = alone_jobs = mixed_jobs = crossed = 0
            for scale in (0.0, 1.0):  # as many single-scale responses as mixed ones
                for _ in range(max(1, (args.answer_rounds + 1) // 2)):
                    alone += chat_burst([(rng.choice(usable), scale) for _ in range(args.concurrency)])
                    alone_jobs += args.concurrency
            for _ in range(max(1, args.answer_rounds)):
                jobs = [(rng.choice(usable), float(k % 2)) for k in range(args.concurrency)]
                rng.shuffle(jobs)
                crossed += chat_burst(jobs)
                mixed_jobs += len(jobs)
            watched("T4 whole answers", before)
            # The same comparison as for positions: a whole answer can come out on the other
            # side in a burst of one scale too (the lane's own batch effect), so what counts
            # is whether mixed bursts do it more often.
            p_answers = fisher_greater(crossed, mixed_jobs - crossed, alone, alone_jobs - alone)
            answers_data.update(mixed=mixed_jobs, crossed=crossed, single_scale=alone_jobs, single_scale_crossed=alone,
                                p=p_answers)
            text = (f"whole answers on the {len(features)} prompts whose two scales differ every time ({by_refusal} "
                    f"by refusal, {len(features) - by_refusal} by length): {crossed} of {mixed_jobs} responses in "
                    f"mixed bursts did what the other scale does, {alone} of {alone_jobs} in single-scale bursts "
                    f"(p = {p_answers:.3g})")
            text_verdict = "FAIL" if p_answers < 0.001 else "INCONCLUSIVE" if p_answers < 0.05 else "PASS"
        if args.require_refusal_flip and by_refusal < len(harmful):
            text_verdict = "FAIL"
            text += f"; --require-refusal-flip: only {by_refusal} of {len(harmful)} refusal prompts flip every time"

        if "FAIL" in (position_verdict, text_verdict):
            verdict = "FAIL"
        elif position_verdict == "PASS" and text_verdict in ("PASS", None):
            verdict = "PASS"
        else:
            verdict = "INCONCLUSIVE"
        detail = f"{reason}; {counts}; {text}"
        if examples:
            detail += "; e.g. " + "; ".join(examples[:3])
        return verdict, detail, {"positions": tally, "p_value": p_value, "detectable": detectable,
                                 "tests": [{"p": t[0], "what": t[1], "scale": t[2], "mixed": [t[3], t[4]],
                                            "single_scale": [t[5], t[6]]} for t in tests],
                                 "position_verdict": position_verdict, "answers_verdict": text_verdict,
                                 "answers": answers_data, "examples": examples, "screen": screen_info,
                                 "sites_used": used, "in_flight": in_flight, "effect_min": EFFECT_MIN,
                                 "engine_steps_mixed": None if mixed_traffic is None else mixed_traffic["steps"]}

    if "T4" in skipped:
        report.add("T4 mixed batches", "SKIP", why_skipped)
    else:
        try:
            verdict, detail, data = mixed_batches()
        except RuntimeError as exc:
            verdict, detail, data = "INCONCLUSIVE", f"could not run: {exc}", {}
        report.add("T4 mixed batches", verdict, detail, **data)

    # ---- T5 both stages -------------------------------------------------------------
    if not args.debug_ranks or "T5" in skipped:
        report.add("T5 both stages", "SKIP", "not asked for (--debug-ranks, on a VLLM_STEERING_DEBUG=1 boot)")
    else:
        first = lane.chat(prompts[0], scale=1.0, ranks=3)
        if first["status"] != 200:
            report.add("T5 both stages", "SKIP",
                       f"the lane rejects steering_ranks (HTTP {first['status']}): boot with VLLM_STEERING_DEBUG=1")
        else:
            failures, unclear, table = [], [], []
            for index, prompt in enumerate(prompts):
                masks = {m: ok(lane.chat(prompt, scale=1.0, ranks=m)) for m in (3, 1, 2, 0)}
                zero, one = [run[index] for run in ref0], [run[index] for run in ref1]
                values = {
                    "both = scale 1": (nearest(masks[3], one), "same"),
                    "neither = stock": (nearest(masks[0], zero), "same"),
                }
                if live[index]:
                    values.update({
                        "stage 0 alone differs from stock": (nearest(masks[1], zero), "moved"),
                        "stage 1 alone differs from stock": (nearest(masks[2], zero), "moved"),
                        "stage 0 alone differs from both": (nearest(masks[1], one), "moved"),
                        "stage 1 alone differs from both": (nearest(masks[2], one), "moved"),
                        "the two stages differ": (distance(masks[1], masks[2]), "moved"),
                    })
                table.append({name: value for name, (value, _) in values.items()})
                own_noise = noise01[index]
                for name, (value, want) in values.items():
                    if (want == "same" and same(value, own_noise)) or (want == "moved" and moved(value, own_noise)):
                        continue
                    clear = moved(value, own_noise) if want == "same" else same(value, own_noise)
                    (failures if clear else unclear).append(f"prompt {index}: {name} ({value:.3g})")
            verdict = "FAIL" if failures else "INCONCLUSIVE" if unclear else "PASS"
            report.add("T5 both stages", verdict,
                       f"{count} prompts x 4 stage masks; " + (
                           "each stage steers on its own and together they equal scale 1" if verdict == "PASS"
                           else "; ".join((failures or unclear)[:4]) + f" ({len(failures)} clear, {len(unclear)} "
                           f"between same and moved; noise {noise:.4g})"),
                       failures=failures, unclear=unclear, distances=table)

    # ---- T6 prefix cache ------------------------------------------------------------
    def cache_test(words: int) -> tuple[list[str], list[str], dict]:
        prompt = long_prompt(words, args.seed)
        steps: dict[str, dict] = {}

        def step(name: str, scale: float, salt: str) -> dict:
            steps[name] = ok(lane.chat(prompt, scale=scale, salt=salt, count_cache=True))
            return steps[name]

        a, b, c, d = ("probe-" + uuid.uuid4().hex for _ in range(4))
        cold0, cold0_again = step("cold 0", 0.0, a), step("cold 0, other salt", 0.0, b)
        warm0 = step("hit at 0", 0.0, a)
        cross1 = step("1 after 0, same salt", 1.0, a)
        cold1 = step("cold 1", 1.0, c)
        warm1 = step("hit at 1", 1.0, a)
        back0 = step("0 again", 0.0, a)
        step("cold 1, fourth salt", 1.0, d)
        cross0 = step("0 after 1, same salt", 0.0, d)
        cold_noise = max(typical, distance(cold0, cold0_again))
        hit_effect = distance(warm0, cold0)
        gap = distance(cold0, cold1)
        tokens = {name: (s["cached"] or {}).get("tokens") for name, s in steps.items()}
        source = next(((s["cached"] or {}).get("source") for s in steps.values() if s["cached"]), None)
        problems, notes = [], []
        if source is None:
            notes.append("the lane reports no cached-token count (no usage details, no counter): reuse unmeasured")
        else:
            for name in ("1 after 0, same salt", "0 after 1, same salt"):
                if tokens[name]:
                    problems.append(f"{tokens[name]} prompt tokens came from the cache on '{name}': blocks are "
                                    "shared across scales")
            for name in ("cold 0", "cold 0, other salt", "cold 1", "cold 1, fourth salt"):
                if tokens[name]:
                    notes.append(f"{tokens[name]} cached tokens on a request with a fresh salt ('{name}'): other "
                                 "traffic moved the counter, or salts do not isolate")
            if not (tokens["hit at 0"] and tokens["hit at 1"]):
                problems.append(f"no reuse within one scale (cached tokens: {tokens['hit at 0']} at 0, "
                                f"{tokens['hit at 1']} at 1)")
        # Outputs. A response is judged against its own scale's cold response, with what
        # this prompt does on its own as the allowance: cold twice, and a hit at scale 0.
        allowance = max(3 * cold_noise, SAME_FLOOR)
        hit_allowance = max(allowance, 3 * hit_effect)
        if not moved(gap, cold_noise):
            notes.append(f"scale 1 does not move this prompt beyond its noise (0 vs 1: {gap:.3g}, cold twice: "
                         f"{cold_noise:.3g}), so its outputs cannot show a leak; the token counts still can")
        else:
            for name, result, own, other, limit in (
                ("1 after 0, same salt", cross1, cold1, cold0, allowance),
                ("0 after 1, same salt", cross0, cold0, cold1, allowance),
                ("hit at 1", warm1, cold1, cold0, hit_allowance),
                ("0 again", back0, cold0, cold1, hit_allowance),
            ):
                d_own, d_other = distance(result, own), distance(result, other)
                if d_own >= d_other:
                    problems.append(f"'{name}' is nearer the other scale's cold output ({d_own:.3g} vs {d_other:.3g})")
                elif d_own > max(limit, 0.25 * gap):
                    problems.append(f"'{name}' is {d_own:.3g} from its own cold output: more than this prompt's "
                                    f"own variation allows ({limit:.3g}) and more than a quarter of 0 vs 1 ({gap:.3g})")
                elif d_own > limit:
                    notes.append(f"'{name}' is {d_own:.3g} from its own cold output (allowance {limit:.3g}, "
                                 f"0 vs 1 {gap:.3g})")
        data = {"prompt_tokens": cold0["usage"].get("prompt_tokens"), "cached_tokens": tokens, "source": source,
                "cold_twice": cold_noise, "hit_at_scale_0": hit_effect, "zero_vs_one": gap,
                "hit_at_scale_1": distance(warm1, cross1), "words": words}
        return problems, notes, data

    if "T6" in skipped:
        report.add("T6 prefix cache", "SKIP", why_skipped)
    else:
        phase_mark = lane.mark()
        problems, notes, data = cache_test(args.long_words)
        if any("no reuse within one scale" in p for p in problems):
            # A hit shorter than the engine's replay window is recomputed in full: try a longer prompt once.
            first_try = data
            problems, notes, data = cache_test(3 * args.long_words)
            data["first_try"] = first_try
        summary = (f"prompt of {data['prompt_tokens']} tokens; cached tokens ({data['source'] or 'unmeasured'}): "
                   + ", ".join(f"{k}: {v}" for k, v in data["cached_tokens"].items())
                   + f"; this prompt's own variation: cold twice {data['cold_twice']:.3g}, a hit at scale 0 "
                   f"{data['hit_at_scale_0']:.3g}, a hit at scale 1 {data['hit_at_scale_1']:.3g}; 0 vs 1 {data['zero_vs_one']:.3g}")
        verdict = "FAIL" if problems else "INCONCLUSIVE" if data["source"] is None else "PASS"
        report.add("T6 prefix cache", verdict, "; ".join(problems + notes + [summary]), **data)
        watched("T6", phase_mark)

    # ---- T7 validation --------------------------------------------------------------
    if "T7" in skipped:
        report.add("T7 validation", "SKIP", why_skipped)
    else:
        bad = [("1.5", {"scale": 1.5}), ("-0.1", {"scale": -0.1}), ("'abc'", {"scale": "abc"}),
               ("top-level 2", {"scale": 2, "top_level": True})]
        statuses = {name: lane.chat(prompts[0], max_tokens=1, **kw)["status"] for name, kw in bad}
        edge = {name: lane.chat(prompts[0], max_tokens=1, scale=value)["status"] for name, value in (("0", 0), ("1", 1))}
        good = all(s == 400 for s in statuses.values()) and all(s == 200 for s in edge.values())
        report.add("T7 validation", "PASS" if good else "FAIL",
                   "; ".join(f"{k} -> {v}" for k, v in {**statuses, **edge}.items()), statuses=statuses)

    # ---- T8 decode steps steer like prefill -----------------------------------------
    def decode_test() -> tuple[str, str, dict]:
        """The lane's own quiet generations, at every site after the first generated
        token: the distribution a decode step produced there against what the same
        context gets when prefilled alone at the same scale and at the other one."""
        screen()
        rows = []
        for (index, scale), path in paths.items():
            for depth in range(1, len(path["tokens"])):
                context = tuple(ids[index] + path["tokens"][:depth])
                if context not in sites:
                    continue
                kind, effect, d_own, d_other = judge_position(
                    path["profile"][depth][1], answer(context, scale), answer(context, 1.0 - scale))
                rows.append({"prompt": index, "scale": scale, "position": depth, "kind": kind, "effect": effect,
                             "own": d_own, "other": d_other})
        steered = [r for r in rows if r["scale"] == 1.0]
        stock = [r for r in rows if r["scale"] == 0.0]
        data = {"rows": rows, "effect_min": EFFECT_MIN, "screen": screen_info}
        if not steered:
            return ("INCONCLUSIVE", f"within {args.t4_depth} tokens no position after the first generated token "
                    f"of a scale-1 generation has the two scales {EFFECT_MIN} apart ({len(stock)} on the scale-0 "
                    "paths): give it the refusal prompts, or raise --t4-depth", data)
        not_own = [r for r in rows if r["own"] >= r["other"]]  # nearer the other scale's prefill, or as near
        worst = max(rows, key=lambda r: r["own"] - r["other"])
        text = (f"{len(steered)} decode-step positions of the scale-1 generations and {len(stock)} of the scale-0 "
                f"ones, each against its context prefilled alone at both scales: {len(rows) - len(not_own)} nearer "
                f"their own scale; together they are {sum(r['own'] for r in rows):.3g} from their own scale's "
                f"prefill and {sum(r['other'] for r in rows):.3g} from the other's; the least clear one (prompt "
                f"{worst['prompt']}, scale {worst['scale']:g}, generated position {worst['position']}) is "
                f"{worst['own']:.3g} from its own and {worst['other']:.3g} from the other's, which differ by "
                f"{worst['effect']:.3g}")
        if stock:
            text += f"; at scale 0, decode against prefill: largest {max(r['own'] for r in stock):.3g}"
        if 3 * len(not_own) >= len(rows):
            return "FAIL", ("a third or more of the tokens generated in decode steps are not nearer what their "
                            "context gets at their own scale: decode steps are not steered like prefill; ") + text, data
        if not_own:
            return "INCONCLUSIVE", f"{len(not_own)} decode-step tokens are not nearer their own scale; " + text, data
        # Every judged position is nearer its own scale. Together they must be clearly so:
        # a position at which steering is worth little counts for little.
        if sum(r["own"] for r in rows) <= 0.25 * sum(r["other"] for r in rows):
            return "PASS", text, data
        return "INCONCLUSIVE", ("the decode-step tokens are nearer their own scale, but taken together by less "
                                "than a factor of four; ") + text, data

    if "T8" in skipped:
        report.add("T8 decode = prefill", "SKIP", why_skipped)
    else:
        try:
            verdict, detail, data = decode_test()
        except RuntimeError as exc:
            verdict, detail, data = "SKIP", str(exc), {}
        report.add("T8 decode = prefill", verdict, detail, **data)

    # ---- other traffic --------------------------------------------------------------
    whole = lane.others_since(run_mark)
    if whole is None:
        report.add("   other traffic", "INFO", "the lane exports no vllm:request_success counter: whether it was "
                   "quiet while the probe ran is unmeasured")
    else:
        def foreign(entry: dict) -> bool:
            # A token of slack per request: the engine and the response may count a
            # request's last token differently.
            return entry["requests"] > 0 or (entry["tokens"] or 0) > entry["own_requests"]

        phases = [t for t in traffic if foreign(t)]
        if not foreign(whole) and not phases:
            detail = (f"none seen: the engine finished {whole['own_requests']} requests while the probe ran and the "
                      f"probe made {whole['own_requests']} of them")
        else:
            detail = (f"the engine finished {max(whole['requests'], 0)} request(s) that were not this probe's and "
                      f"generated {max(whole['tokens'] or 0, 0)} token(s) more than the probe was answered with; "
                      "during: " + (", ".join(f"{t['phase']} ({t['requests']} requests, {t['tokens']} tokens)"
                                              for t in phases) or "between the tests")
                      + ". Requests that share engine steps move each other's output on this lane: verdicts "
                      "taken during those phases are from a lane that was not quiet")
        report.add("   other traffic", "INFO", detail, whole=whole, phases=traffic)

    # ---- verdict --------------------------------------------------------------------
    needed = ("T1 stock", "T2", "T4", "T6", "T7")
    failed = [t["test"] for t in report.tests if t["verdict"] == "FAIL"]
    unmet = [n for n in needed if report.verdict(n) != "PASS"]
    verdict = "PASS" if not failed and not unmet else "FAIL"
    print(f"STEERING-PROBE {verdict}: {len(failed)} failed, {len(unmet)} required test(s) not passed"
          f"{' (' + ', '.join(unmet) + ')' if unmet else ''}; {lane.requests} requests in "
          f"{time.time() - started:.0f} s")
    if args.out:
        receipt = {
            "revision": 5, "verdict": verdict, "url": lane.url, "model": lane.model, "max_tokens": lane.max_tokens,
            "noise": noise, "noise_scale0": spread0, "noise_scale1": spread1, "positions": depth,
            "busy_at_start": busy, "tests": report.tests,
            "references": [
                {"prompt": p, "scale0": a["text"], "scale1": b["text"], "distance": d, "moved": l,
                 "tokens_scale0": a.get("length"), "tokens_scale1": b.get("length")}
                for p, a, b, d, l in zip(prompts, ref0[0], ref1[0], d01, live)
            ],
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, default=lambda value: sorted(value) if isinstance(value, set) else str(value))
        print(f"wrote {args.out}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"STEERING-PROBE ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
