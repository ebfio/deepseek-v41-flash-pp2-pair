# DeepSeek V4.1 Flash — PP2 + DSpark on a GB300 two-station pair

A complete, running recipe for serving **DeepSeek V4.1 Flash** as **TP1 × PP2
(pipeline parallelism) with DSpark speculative decoding** across two
NVIDIA DGX Station GB300 systems over 400G CX8 RoCE — plus the four small
vLLM source fixes that make it work on builds that ship the DeepSeek V4.1
NVIDIA tree (`vllm/models/deepseek_v41/nvidia/`).

## Why PP2

Under TP2, every decode step pays 84 all-reduces (~32% of the 12.3 ms step on
this fabric). PP2 splits the 40 layers 20/20 across the two Grace-Blackwell
systems and replaces the collectives with two point-to-point hops per step.
Under TP2 every decode step pays 84 all-reduces, ~32% of the 12.3 ms step on
this fabric. PP2 splits the 40 layers 20/20 across the two Grace-Blackwell
systems and replaces the collectives with two point-to-point hops per step.

Measured on this pair. The PP2 column is the shipped config (boot
B-double-prime, 2026-11-05); the TP2 column is the 2026-10-25 baseline and has
not been re-measured on the current build.

| Metric | TP2 (2026-10-25) | PP2 (this recipe, current) |
| --- | --- | --- |
| All-reduces per decode step | 84 | **0** |
| Single-stream tok/s (identical temp-0 prompt) | 290.5 | **410.4** (+41%) |
| Step time | 12.3 ms | **9.25 ms** |
| KV cache pool | — | **70,529,814 tokens** (was 45.5M before the serving-config work) |
| GSM8K | — | **0.965** (386/400, 95% CI 0.942-0.979) |

Every PP2 figure above is reproducible from this repo: `verify/full_ab.py` for
tok/s and step time (steps from the engine's own draft counters, never derived
from tok/s), `verify/gsm8k_width.py` for the rate with an interval. The
intermediate configs and what they cost are in "The serving config that won".

**Where PP2 wins and where it may not:** PP2 is the single-stream shape. The
pipeline needs scheduler admission depth to fill both stages — measured
aggregate tok/s on this build (2026-10-25, pre-autotune): C1 323 · C2 406 ·
C4 595 · C8 974 (3.01 × C1).
TP2's step is batch-insensitive, so TP2 vs PP2 at C4–C16 is **not yet
measured on this build** and TP2 may be the faster shape at higher
concurrency; run the identical soak on both before committing a fan-out
design. For interactive per-stream latency, cap concurrency around 4–6.

Host memory held flat through the C8 soak: zero swap, negligible compaction.

Vision works end-to-end (image tokens route through the MoE `bias_vl` path on
both stages — see fix 3), and generation is deterministic within PP2.

## The four fixes (five files)

Stock builds reject DSpark + pipeline parallelism for this model, and three
more walls sit behind the first. The fixes are small and mainline-shaped:
four source overlays against `vllm/vllm-openai:v0.31.0` (shipped both as
whole files and as `.diff`s alongside them) plus one startup-time attribute
flip. **The whole-file overlays are copies of v0.31.0 sources with these
edits — on any other tag, use the `.diff` files against that tag's pristine
sources and re-verify; do not mount the whole files.** The overlaid files
are modified vLLM sources and remain under Apache-2.0.

1. **`sitecustomize.py`** — set `supports_aux_hidden_states_over_pp = True`
   on the *inner decoder* classes (`DeepseekV4Model`,
   `DSparkDeepseekV4Model`). The gate in `eagle3_utils.py` reads
   `model.get_language_model() → .model`, not the `ForCausalLM` wrapper;
   flipping the wrapper does nothing. **Partition-dependent:** the
   flag-only fix is correct when every draft target layer (37–39) sits on
   the last stage — true for the default 20/20 split. A partition whose
   last stage starts above 37 would silently degrade the drafts
   (acceptance collapse, no error), so `serve_vllm_node.sh` refuses it.
2. **`dspark.py`** — `self.has_own_embed_tokens = get_pp_group().world_size
   > 1` in the draft's `__init__`, plus a `name == "embed.weight"` catch in
   `load_weights` that loads the draft's own embedding (~1.3 GB on rank 1),
   with a post-load assert that it actually loaded — a renamed checkpoint
   key fails loudly instead of producing garbage drafts. On PP stage 1 the
   target's `embed_tokens` is a `PPMissingLayer`, so the stock share path
   raises; with the flag, mainline's `maybe_share_target_embed` skips the
   share cleanly.
3. **`model_runner.py` + `cudagraph_utils.py`** — gate the non-first-PP-rank
   nulling of `model_inputs["input_ids"]` on
   `not requires_raw_input_tokens(self.model)` at both the runner and the
   capture site. Vision MoE routing needs the raw token ids on every rank;
   the stock PP contract nulls them past rank 0 and the model raises
   `DeepSeek V4 vision MoE routing requires input_ids.` (Upstream vLLM main
   has since merged the same gate — this overlay is for the v0.31.0 tag.)
4. **`kv_cache_utils.py`** — filter a group's `kv_cache_specs` to the layer
   names the worker actually owns. Under PP, a rank holding none of a
   group's layers still sees the group's global spec, and
   `allocate_kv_cache` dies with `StopIteration` (`worker/utils.py:432`).
   This filter originated in an earlier private two-station port of the
   same idea and was re-based onto v0.31.0 here; the crash site exists
   verbatim in vLLM main.

Fixes 3 and 4 are generic vLLM v1 pipeline-parallel bugs, not DeepSeek
specifics — they apply to any model with `requires_raw_input_tokens = True`
or hybrid KV groups under PP.

## What this costs

Both Engram tables are pinned in host memory on rank 0 (~189 GiB) — rank 0
is also the vision-encoder rank and the API rank, so it carries the most
host memory pressure and should be the bigger-memory box if your pair is
uneven. The draft's own embedding costs ~1.3 GB device memory on rank 1.
Each decode step pays two point-to-point hops (hidden states down, drafts
back), ~0.1–0.2 ms of the ~9.25 ms step.

## Layout

```
recipe/               # the running lane (as deployed on the pair; IPs and
                      # ssh targets are placeholders — fill your own)
  config.env          # rendered production config (PP2 + DSpark, spec n=5)
  serve_vllm_node.sh  # per-rank runner: mounts, gates, docker invocation
  launch_cluster.sh   # two-node launcher (health-waits, streams both ranks)
  preflight.sh        # pre-launch checks
  stage_host_rdma.sh  # host rdma-core staging for the container
patches/
  vllm-pp2-flag/      # the four overlays + sitecustomize + .diff files,
                      # plus pp_autotune.py (PP-aware autotune, opt-in via
                      # VLLM_PATCH_PP_AUTOTUNE=1)
verify/               # the instruments used to validate the lane
  texts_prod.py       # train-prompt determinism receipt (md5 ×3)
  pp2_receipts.py     # 24-prompt parity sweep + counter-based acceptance
  pp2_soak.py         # C1/C2/C4/C8 aggregate + host memory gates
  prefix_probe_agentscale.py  # prefix-cache hit rate on growing sessions
  gsm8k_gate.py       # 100-question GSM8K slice (task-level gate)
```

## Running it

Hardware: two DGX Station GB300, 2 400G CX8 rails between them, DeepSeek V4.1
Flash checkpoint on local NVMe, image `vllm/vllm-openai:v0.31.0`.

1. Fill the placeholder IPs/ssh targets in `recipe/config.env` for your pair.
2. Stage the recipe directory on both nodes (same path on both).
3. `bash -n recipe/config.env` — always syntax-check the rendered config
   before any launch.
4. From rank 0: `bash recipe/launch_cluster.sh` — it launches both ranks,
   health-waits, and streams both logs.
5. Verify: `python3 verify/texts_prod.py` (expect three identical md5s),
   then `python3 verify/pp2_soak.py`.

The `serve_vllm_node.sh` gate refuses any `VLLM_PARALLEL=pp` + DSpark launch
without the overlay mounted, and refuses layer partitions whose last stage
excludes the draft target layers — stock builds fail ~16 seconds in with a
`does not support dspark with pipeline parallelism` error that wastes four
minutes of weight loading.

## Verification notes

**GSM8K slice definition** (for reproducing 97/100): the first 100 questions
of the `openai/gsm8k` test set (config `main`), 0-shot, temp 0,
`max_tokens` 2048, prompt suffixed with "Reason step by step, then give the
final numeric answer on a line starting with '#### '." Scored by exact
integer match: the target's `#### N` value vs the last integer in the
model's content field. Three misses, all near-miss arithmetic.

**Prefix-cache method** (for the 0.95–0.99 row): measured on a growing
8-turn session with a ~10K-token context, real assistant content fed back
each turn. `vllm:prefix_cache_queries`/`hits` count tokens, block size is
128, and bounded SWA replay discards hits ≤ its window — so a conversation
under ~300 tokens *cannot* register a hit by design; don't file a bug when
a toy probe reads zero. The healthy signature at scale: every turn hits
except the most recent assistant exchange (re-tokenization).

**Parity note.** PP2 changes temp-0 outputs relative to TP2: 0/24 md5s
match. This is documented divergence, not a bug — TP2 splits every GEMM
across two GPUs and sums bf16 partials; PP2 computes each GEMM whole, which
is numerically what single-GPU execution would be. Generation is
deterministic within a parallel shape. If you need cross-mode
reproducibility for audits, scope that contract to a single mode.

## FlashInfer autotune under PP2 — the world-barrier bug

Turning autotune on (`--enable-flashinfer-autotune`) with stock v0.31.0 makes
this pair unbootable. `flashinfer_autotune()` broadcasts and barriers over the
**world** group and only the **leader** saves, but under PP each stage tunes a
different op set (rank 1 additionally tunes the DSpark drafter). Rank 0
finishes its tune, hits `world.barrier()` (`kernel_warmup.py:480`), and waits
for rank 1's much longer tune; the gloo group times out (~1800 s default,
`cpu_distributed_timeout_seconds`) and the engine dies with
`EngineDeadError`. Measured three times, all at ~44 min:

```
kernel_warmup.py:480   world.barrier()
parallel_state.py:1362 torch.distributed.barrier(group=self.cpu_group)
RuntimeError: [gloo/transport/tcp/pair.cc:553] Connection closed by peer ...
```

Worse, rank 1's drafter tactics are never persisted (leader-only save), so
every boot re-tunes them.

`patches/vllm-pp2-flag/pp_autotune.py` fixes both, gated by
`VLLM_PATCH_PP_AUTOTUNE=1`:

- cache file per PP stage (`autotune_configs.pp0.json`, `.pp1.json`)
- broadcast/barrier scoped to the TP group (size 1 per stage here, so no
  cross-rank collectives at all)
- no world barrier; each rank saves its own tactics and proceeds

`pp_autotune.py` v3 also skips profiling entirely when this config's cache
loads (override with `VLLM_FLASHINFER_AUTOTUNE_FORCE_RETUNE=1`). Receipts: zero
`Tuning` lines, `init engine` 111 s vs 591 s cold, boot ~8 min, md5 unchanged
across the restart.

Measured after adoption (2026-11-05, same prompts, temp 0):

| metric | autotune off | autotune + overlay |
|---|---|---|
| C1 tok/s | 391 | **407.9** (+4.3%, 4 runs x 3 reps, spread 0.1%) *superseded by B-double-prime, 410.4* |
| md5 (train prompt) | 616400039175 | cb6c80a8fb12 (x12 identical) |
| completion tokens | 293 | 301 |
| GSM8K | 97/100 | **99/100** |
| acceptance length | ~3.7 | 3.73 |
| KV pool | 45,521,213 | 45,521,213 (unchanged) |

Warm-boot prediction, not yet measured: with both `pp*.json` files present the
next boot should skip tuning entirely (`tune_mode` only profiles missing keys).

**Two traps found on the way.**

1. `--kv-cache-dtype` is resolved by the attention backend. Under
   `FLASHMLA_MEGA_ATTN_DSV41` an unspecific value becomes `nvfp4_ds_mla`
   (4-bit compressed cache), not `fp8`. Spell `fp8_ds_mla` explicitly if you
   want the production KV format.
2. `FLASHMLA_MEGA_ATTN_DSV41` is a **regression at C1** on this pair:
   364.4 tok/s vs 391 (-6.8%, 9 samples, spread <0.5%), KV pool -5.2%, and
   +1.51 GiB persistent on rank 1 (155.34 vs 153.83 GiB across three
   production boots). The fused kernel's fixed tile cost does not amortize at
   six tokens. Kept here as a negative result so it is not re-run.

## The serving config that won (boot B-double-prime, 2026-11-05)

Three settings, one at a time, measured on this pair. All C1 numbers are the
train prompt at temp 0, 3 runs x 3 reps, steps counted from the engine's own
draft counters (never derived from tok/s).

| change | effect | verdict |
|---|---|---|
| `FLASHMLA_MEGA_ATTN_DSV41` + `fp8_ds_mla` | C1 364.4 vs 391 (-6.8%), KV -5.2%, +1.51 GiB persistent on rank 1 | rejected |
| `num_speculative_tokens_per_batch_size` + pinned capture list | C1 384.0 vs 407.9 (-5.9%), acceptance 3.562 vs 3.73 | rejected |
| `--max-num-batched-tokens 8192` + `--long-prefill-token-threshold 6144` | **KV 70,529,814 vs 45,521,213 (+55%)**, C1 flat, acceptance flat | **adopted** |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | **KV +15.8% on top** (60.9M -> 70.5M), same acceptance | **adopted** |

Production now runs `max-num-batched-tokens 8192`, `long-prefill-token-threshold
6144`, default capture list, no k-schedule, and the profiler estimate off.

**Why the profiler flag matters and what it costs.** `gpu_worker.py:707` logs
that `--gpu-memory-utilization=0.9000 is equivalent to 0.8811` when the profiler
is on: it measures a real activation peak and withholds it. Turning it off makes
that a *guess*. Measured on this model, the guess is 9.5 GiB low on rank 1
(11.83 GiB measured peak vs 2.34 GiB estimate), and that 9.5 GiB becomes KV
cache. So the pool gain is real and the reserve it came from is not. Gate any
further utilization increase behind a 1M-context prefill soak with
`nvidia-smi` peak watched; do not stack `--gpu-memory-utilization 0.95` on top.

**Warm boot.** `pp_autotune.py` now loads this config's cache and returns
without profiling. Receipts from the drill: zero `Tuning` lines, `init engine`
111 s vs 591 s cold, boot ~8 min, and the md5 is unchanged across the restart -
which is the proof the loaded tactics are the ones serving.

**Identity, corrected.** The cache file is keyed by the full `vllm_config` hash.
Same hash reproduces the same md5 across boots (3/3 measured). A different hash
is *expected* to produce a different md5; cross-config quality is GSM8K's job,
not md5's. A "seed" env that loaded another config's tactics was tried and
removed: it produced `0 new, N from previous` but the tune still re-benchmarked
and re-selected, so the numerics did not follow the seed.

**GSM8K on the adopted config.** 386/400 = 0.965 (95% CI 0.942-0.979) on
q101-500, against 97/100 on the q1-100 gate - overlapping intervals, no
regression. Five of the fourteen misses were `max_tokens` truncation at 4096;
at 8192 they pass (q87 verified). Budget 8192, not 4096, for this reasoning
model, or you will manufacture a ~1.25% phantom error rate.

**Traps found the hard way.**
1. `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` is spelled with an A. Misspell it
   and vLLM logs `Unknown vLLM environment variable`, then profiles as usual -
   a silent no-op.
2. `--kv-cache-dtype` is resolved by the attention backend: under
   `FLASHMLA_MEGA_ATTN_DSV41` an unspecific value becomes `nvfp4_ds_mla`, not
   `fp8`. Spell `fp8_ds_mla` if you want the production KV format.
3. The engine container has no `/models` mount. Any host path you pass in an
   env var for use inside the container must be the container path
   (`/root/.cache/...`), not the host one.

## Upstream

Three changes belong in vLLM proper, in this order (links to follow):

1. The `input_ids` PP-contract gate (fix 3) — merged in main before we
   could file it; the same gate we wrote independently. [link]
2. The KV-group filter for PP ranks owning none of a group's layers
   (fix 4) — the `StopIteration` site exists in main; issue filed. [link]
3. DeepSeek V4.1 PP + DSpark enablement (fixes 1–2, plus a
   partition-independent pack/collect for the aux states so any split
   works) — vendor-tree dependent; tracked here. [link]
4. `flashinfer_autotune()` under pipeline parallelism: scope the
   broadcast/barrier to the TP group and save per PP stage instead of
   leader-only over the world group (see the autotune section above) — the
   overlay here is the working shape. [link]

## Credits

This port was performed by **Iris** — a GLM-5.3 model running on 4× RTX Pro
6000s — with help from **Eduardo**, and from **Kestrel**, Eduardo's personal
agent running on DeepSeek-v4.1-flash on the machines in question. The lane
Kestrel runs on is the one this recipe builds.
