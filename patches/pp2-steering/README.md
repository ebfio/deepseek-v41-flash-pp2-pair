# pp2-steering: per-request activation steering on the PP2 lane

One stock checkpoint, two behaviours. Each request carries `steering_scale` in [0, 1]:
0 is the stock model bit for bit, 1 applies the steering sidecar in full. Two outputs of
every steered layer are edited, the attention module's and the shared expert's, each by
its own operator, without replacing a single vLLM file:

    x' = (x + s * write * (read . x)) * (1 + s * (row_gain - 1))        s = the request's scale

"Sidecar" here always means the steering sidecar, a small `.safetensors` of per-layer
vectors. It has nothing to do with an RTX expert sidecar.

## Status, read this first

| What | State |
|---|---|
| The defaults on the pair (replay off, torch kernel, published sidecar) | **Booted and steered on 2026-11-20**, vLLM v0.34.0, PP2, stock weights on both stages, first with revision 2 of the module and in the evening with revision 4. Hooks reported on both ranks: 224 and 238 extra kernels. As reported, revision 4's probe passed T1, T2, T3, T1b, T6 and T7 on the evening's boot; T5 passed on the morning's (revision 3's probe). T4 (mixed batches) is undecided. See "What the pair showed". |
| Scale 0 against a boot without the module (T1b) | **Passed, as reported** (revision 4's probe): the baseline was recorded on a boot without the module, which came up with the replay-boundary guard, and scale 0 on the steered boot matched it. Revision 5 reads the same baseline file and compares by its own measure, which is the stricter one: run it once more. |
| Requests of different scales in flight together (T4) | **Not established.** Revision 4's T4 could not decide it on the pair, for reasons that were the probe's: see "What the pair showed". Revision 5's T4 has run on toy lanes only. Until it has passed on the pair, treat mixed batches as unproven. |
| Math, per-token gather, stage ownership, replay batch, cache salt, API validation, both sidecar formats, fail-closed paths, the stage check | 28 CPU tests against stand-ins: `python3 test_pp2_steering.py` (the fused kernel in Triton's interpreter) |
| A module and a sidecar that do not go together are refused before a boot | `stage_check.py`, run by `recipe/serve_vllm_node.sh` on each node: dry runs with the stub docker, and a test that holds it against the loader. **Not run on the boxes.** |
| The patches on the real v0.34.0 classes (request classes, `Request`, block hasher, model-state resolver, `InputBatch`, `DecoderReplayLayers`) | `verify/steering_image_smoke.py`: passes against the v0.34.0 source tree on a CPU host; revision 2 of it passed in the image except for one check that was the script's own defect |
| The seams exist where the module expects them | `verify/steering_seams.py`: PASS on v0.34.0, v0.33.0 and v0.31.0 trees, FAIL on a broken one; passed in the image |
| Which float edit the abliterated checkpoint was built with | `verify/sidecar_vs_weights.py`: validated on synthetic checkpoints only. **Not yet run on the real pair of checkpoints.** See "Which sidecar". |
| Launcher: steering off changes nothing, steering on mounts and refuses correctly | stub-docker dry run, both lanes. Revision 2 handed a relative sidecar path to docker as a volume name; fixed, and the stub now refuses that. |
| `VLLM_STEERING_REPLAY=keep` (upstream's decoder replay stays on) | CPU tests, and the real `DecoderReplayLayers` on a CPU. **Never run in an engine, and of no use on this pair's 20/20 split**: vLLM v0.34.0 cannot replay on a stage that begins at the last KV source layer. See "Decoder bounded replay". |
| `VLLM_STEERING_KERNEL=fused` (one kernel per steered output) | compiled for sm_90, sm_100, sm_103 and sm_120 with Triton 3.8.0 and run in Triton's CPU interpreter. **Never run on a GPU**: `verify/steering_kernel_check.py` is that run. |
| Speed | Measured on the pair by `verify/decode_rate.py --train-prompt --tokens 512` (Iris, 2026-11-20), same prompt on three boots: **the hooks cost 1.19 ms per speculative round at scale 0, 12 % of C1 tok/s** (429.1 without the module, 376.7 with it). Tokens per round are the same with and without the module (3.69). See "Cost". |

## Request API

    {"model": "...", "messages": [...], "vllm_xargs": {"steering_scale": 1.0}}
    {"model": "...", "messages": [...], "steering_scale": 1.0}          # same thing

Omitted means `VLLM_STEERING_DEFAULT_SCALE` (0 unless set). Out of range or not a number
is HTTP 400. The engine honours whoever sends the field: if "unlimited" is meant to be an
access-controlled model name, the control has to sit in the router in front (set the field
per alias, strip it from client requests).

## Which sidecar

Two formats load.

| | Published (v1) | Derived (v2) |
|---|---|---|
| File | the sidecar published as `jcartu/WARMIND-1017-Steering-Sidecar` | written by `verify/sidecar_vs_weights.py --derive` |
| Tensors | `layers.{L}.{alpha, direction, attn_row_gain}`, layers 4..36 | `layers.{L}.{attn,shared}.{read, write[, row_gain]}`, FP32 |
| Operator | both sites: read = `direction`, write = `-alpha * direction`; row gain on the attention output only (the arithmetic of hyperflare-inference PR #404) | whatever rebuilds the abliterated checkpoint; read and write may differ |
| Refused when | inventory, dtypes or norms differ from the reference loader's rules | its metadata does not say `REBUILT` (checked at interpreter start) |

**The published sidecar has not been shown to be the abliterated checkpoint's edit, and
the first comparison says it is not.** Measured on the Stations on 2026-11-20 with
revision 2 of `sidecar_vs_weights.py`: the 132 rewritten tensors are exactly the two
writers in 33 layers, the change lies along the sidecar's direction (cosine 0.964 and
0.960), and its strength along that direction reads 2.60 and 2.55 where the sidecar says
3.19. Two things are mixed in those numbers:

1. Rounding. The abliterated checkpoint is the stock one dequantised, edited and quantised
   to MXFP8 again. A value changes only when its edit exceeds half its rounding step, so
   the difference between the two checkpoints is a thresholded copy of the float edit:
   weaker, and bent. Revision 2 compared the dequantised weights and so could not have
   read 3.19 even from an exact rebuild (on a synthetic checkpoint of the real size: 3.01
   of 3.19, cosine 0.995).
2. A different operator. A build that normalises each row, removes the direction from the
   unit rows, renormalises and restores the row norms, at strength 3.0, has a write vector
   of `row_norm * d` and a read vector of `d / row_norm`. Its strength measured along `d`
   alone is 3.0 x mean(row_norm) x mean(1 / row_norm), the means weighted by the squared
   entries of `d`, which is above 3.0. A synthetic
   checkpoint of the real size built that way, with row norms spread so that this product
   is 3.19, reads cosine 0.965 and 2.64 of 3.19 in revision 2's test. The Stations read
   0.964 and 2.60 of 3.19. That is a match on numbers I did not fit, and it is still an
   inference from a simulation.

Revision 3 of the script settles it from the bytes: it quantises each candidate's float
result onto the abliterated checkpoint's own grid and counts the values that agree (J).
An exact candidate scores 0.97 or better, 3 % off in strength about 0.89, a direction 15
degrees off 0.2 to 0.4. Its verdict decides what to load:

| Verdict | Load | Scale 1 then is |
|---|---|---|
| `REBUILT-AS-PUBLISHED` | the published sidecar | the float edit the checkpoint was built with |
| `REBUILT` | the file `--derive` wrote, not the published one | the same; the published operator would have been a different edit (the script says by how much) |
| `OPEN` | nothing | not established: do not spend a window on it |

Even on `REBUILT` the lane and the abliterated checkpoint are not bit-equal, and cannot
be. The checkpoint is the float edit after MXFP8 rounding; the lane applies the float
edit itself. The script reports how much of the float edit the checkpoint carries
("carries", 0.93 to 0.95 on the synthetic checkpoints of the real size). Scale 1 is the
edit as designed; a scale equal to the carried share has the checkpoint's strength along
the edit. Which of the two "unlimited" should mean is a product decision, and the probe's
refusal counts at both are the evidence for it.

One build the lane cannot reproduce: a gain per input column (columns normalised instead
of rows). The attention writer's input is an MXFP8-quantised activation on the decode
path (`vllm/models/deepseek_v41/nvidia/ops/o_proj.py:100-113`), so there is nothing to
multiply. The script identifies that build, says how much of the edit the gain is, and
answers `OPEN`.

## How it attaches (v0.34.0 file:line, all read in the tag's source)

Worker, every PP rank.
- The runner builds its model state through `resolve_model_state_cls`
  (`vllm/v1/worker/gpu/model_states/__init__.py:28`, called at
  `vllm/v1/worker/gpu/model_runner.py:447`, before any forward pass). The module wraps that
  function and returns a subclass of `DeepseekV41ModelState`
  (`vllm/models/deepseek_v41/nvidia/model_state.py:173`).
- `add_request` runs on every rank for every new request
  (`vllm/v1/worker/gpu/model_runner.py:1320`), preempted requests included, because they
  are removed and come back as new (`vllm/v1/worker/gpu/model_runner.py:1262`). The subclass
  reads `sampling_params.extra_args["steering_scale"]` there.
- `prepare_inputs` runs every step (`vllm/v1/worker/gpu/model_runner.py:2086`). The subclass
  fills a persistent `[max_num_batched_tokens, 1]` buffer with each row's scale, on the GPU,
  no host sync. Same pattern as upstream's own `lookback_token_ids`
  (`vllm/models/deepseek_v41/nvidia/model_state.py:208`), so captured CUDA graphs read it
  on replay. Graph capture zeroes it through `prepare_dummy_inputs`
  (`vllm/v1/worker/gpu/cudagraph_utils.py:753`).
- Forward hooks on `layer.attn` and `layer.ffn.shared_experts` of the layers this stage owns
  and the sidecar names. The attention module returns its writer's output unchanged
  (`vllm/models/deepseek_v41/attention.py:769`) and both are reached through
  `Module.__call__`: `vllm/models/deepseek_v41/nvidia/model.py:644` and
  `vllm/model_executor/layers/fused_moe/runner/shared_experts.py:159` (auxiliary stream) and
  `vllm/model_executor/layers/fused_moe/runner/shared_experts.py:191` (inline). The shared
  expert's output is steered before it is added to the routed output
  (`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:804`), as in the reference.
- Rows whose scale is 0 go through `torch.where` and keep their bits.

Engine core.
- `Request.from_engine_core_request` (`vllm/v1/request.py:257`) is wrapped: a non-zero scale
  is appended to `cache_salt`, which enters the first block's hash
  (`vllm/v1/core/kv_cache_utils.py:656`) and through the chain every later one. Scale 0 is
  left alone, so stock requests keep sharing prefix blocks as before. Hashes precomputed by
  a frontend are dropped for a steered request.

API server (optional seams: without them steering still works through `vllm_xargs`).
- `to_sampling_params` of the chat and completion request classes
  (`vllm/entrypoints/openai/chat_completion/protocol.py:750`,
  `vllm/entrypoints/openai/completion/protocol.py:359`) is wrapped to validate and to accept
  the top-level spelling. The error is a `VLLMValidationError`, which the server maps to 400
  (`vllm/entrypoints/serve/exception_handling/error_response.py:43`).

Loading. `patches/vllm-pp2-flag-v034/sitecustomize.py` imports this module when
`PP2_STEERING_SIDECAR` is set, in every Python process of the container. Nothing heavy is
imported at interpreter start; each seam is patched when vLLM imports its module.

## What it refuses, and why

It ends the process (status 78) when the sidecar is missing, its sha256 differs, a
derived sidecar does not say `REBUILT`, or a required seam cannot be patched: Python
swallows exceptions from `sitecustomize`, and an engine that boots without the hooks
serves every request stock. At model-state creation it raises on anything that runs the
layers on something other than "all rows of the step, in batch order":

| Refused | Because |
|---|---|
| tensor parallel > 1 | not validated; `PP2_STEERING_ALLOW_TP=1` lifts it for a test boot, see below |
| data parallel > 1, prefill context parallel, sequence parallel | not validated; rows are sharded |
| microbatching (`--enable-dbo`, `--ubatch-size`; `vllm/config/parallel.py:720`) | the model runs on sub-batches |
| `decode_backend=latency_specialized` (`vllm/models/deepseek_v41/nvidia/model.py:1711`, `vllm/models/deepseek_v41/nvidia/specialized_decode/__init__.py:141`) | another model state and decode path |
| a fused shared expert (MegaMoE, needs `--enable-expert-parallel`: `vllm/models/deepseek_v4/nvidia/model.py:232`, `vllm/models/deepseek_v4/nvidia/model.py:399`) | no module left to hook |
| decoder replay still on with `VLLM_STEERING_REPLAY=off`; replay graphs with `keep`; KV-sharing fast prefill (`vllm/config/cache.py:370`) | each runs hooked layers on a row subset the buffer in use does not describe; see "Decoder bounded replay" |

The launcher (`recipe/serve_vllm_node.sh`) refuses before that: any lane but v0.34.0 PP with
the flag overlay, the graph-attribution overlay, the Rust frontend, a bad sha256, and
`MODEL_DIR_RANK0 != MODEL_DIR_RANK1`. `launch_cluster.sh` fails a READY cluster on which
either stage did not log `[pp2_steering] rank N: steering layers ...`, and
`pair_supervisor.sh` then removes both ranks instead of leaving them serving.

## A test boot on a box that cannot run a pipeline stage

A single box with several smaller GPUs can host a v0.34.0 engine under tensor parallel,
not a PP2 stage. `PP2_STEERING_ALLOW_TP=1` in the container's environment lets the
module start there. The argument that it is correct: both edits are linear in the hooked
tensor, so they commute with the sum over TP ranks whether a hook sees the reduced tensor
or one rank's partial. The attention writer reduces its own output unless sequence
parallel or the fused all-reduce took that over
(`vllm/models/deepseek_v41/nvidia/model.py:372-373`,
`vllm/models/deepseek_v41/attention.py:835-847`), and the shared expert's output is a
partial that is reduced later with the routed one
(`vllm/models/deepseek_v4/nvidia/model.py:333`). That is an argument from the source, not a
measurement. What such a boot can show before a window on the Stations: the hooks under
CUDA graphs and the auxiliary stream, probe tests T1 to T4, T6 and T7, and the cost per
step on that hardware. What it cannot: the hand-off between two stages (T5) and any
number for the pair.

## Decoder bounded replay: off or kept (`VLLM_STEERING_REPLAY`)

Since 0.33 the stage that holds the last KV source layer through the end runs the layers
from that one on, in eager prefill steps, on each request's last window of rows only
(`vllm/models/deepseek_v41/nvidia/model.py:884`,
`vllm/models/deepseek_v41/nvidia/model.py:891`,
`vllm/models/deepseek_v41/nvidia/model.py:1295`). The first of them still writes its KV
for every row (`vllm/models/deepseek_v41/nvidia/model.py:712`, a method call, so no hook
fires there, and none should: KV comes from the layer's input).

Those calls see a sub-batch, and a hook that read the first rows of the whole batch's
buffer would give them other rows' scales whenever a step mixes scales.

**On this pair there is nothing to keep.** Rank 1 holds layers 20 to 39. If the
checkpoint's last KV source layer is 20, rank 1 begins exactly there, and vLLM v0.34.0
cannot replay on such a stage: it turns replay on and then fails the stage's first
forward with an `AssertionError` that has no message
(`vllm/models/deepseek_v41/nvidia/model.py:1332`,
`vllm/models/deepseek_v41/nvidia/model.py:1120`,
`vllm/models/deepseek_v41/nvidia/model.py:1247`; the whole account is in
`patches/vllm-pp2-flag-v034/README.md`). The boot of 2026-11-20 without this module died
that way, as far as its surviving log shows. So the reason the lane boots with the module
is that the module turns replay off. Revision 3 of this README described "keep" as the
arrangement for rank 1 of the pair and planned a boot with it; that boot would have died
the same way. I read the support check and did not see that equality passes it.

**off** (default, and what booted on 2026-11-20). `_decoder_replay_supported` answers no,
which puts the stage in the state upstream itself uses when replay is unsupported
(`decoder_replay_layers` stays `None`,
`vllm/models/deepseek_v41/nvidia/model_state.py:223`). Rank 1 logged
`decoder SWA bounded replay would be on for this stage and steering turns it off`. Rank 1
then runs its 20 layers on every prompt row. Since revision 4 the overlay's
`sitecustomize.py` does the same for a stage that begins at the last KV source layer
whether or not this module is loaded, so on the pair a boot without the module computes
what a boot with it computes at scale 0.

**keep.** Upstream's gate is left alone and the module follows the sub-batch instead. The
model state builds it in `_prepare_replay_batch`
(`vllm/models/deepseek_v41/nvidia/model_state.py:493`), called from `prepare_attn`, which
the runner calls before `prepare_inputs`
(`vllm/v1/worker/gpu/model_runner.py:2035`, `vllm/v1/worker/gpu/model_runner.py:2086`).
The sub-batch holds the same requests in the same order, each with its last rows, and
its row ends are `self._kept_query_start_loc`
(`vllm/models/deepseek_v41/nvidia/model_state.py:743`). The subclass fills a second
buffer from those ends right after upstream built the batch, and a hook of a layer from
`decoder_replay_start` on reads that buffer while the replay layers run. It tells by the
forward context: the replay layers run under the sub-batch's own
(`vllm/models/deepseek_v41/decoder_replay_layers.py:123`).

Where keep can apply: a pipeline stage that begins before the last KV source layer and
holds everything after it, for example rank 1 of a 19/21 split. Refused under keep:
- a stage that begins at the last KV source layer with upstream's replay on (the guard
  removed): the module stops with the reason instead of letting the engine die without one;
- a stage that runs its replay batch in CUDA graphs of its own. Those exist on the first
  rank only (`vllm/models/deepseek_v41/nvidia/model_state.py:268`), so on a single-stage
  lane. Their capture goes through a reference to the layer runner taken before the
  module could see it, under a copy of the context
  (`vllm/models/deepseek_v41/nvidia/decoder_replay_cudagraph.py:57`,
  `vllm/models/deepseek_v41/nvidia/decoder_replay_cudagraph.py:98`).

keep has not run in an engine.

## Cost

**What is known.** By op count the torch path adds 6 kernels per steered output and 8
where it has a row gain: 14 per layer with the published sidecar, 224 on rank 0 and 238
on rank 1, in every forward pass, for every request, scale 0 included, because a captured
graph cannot branch on data. In an eager prefill step each of those kernels also walks a
`[rows, 5120]` tensor, twice as FP32 temporaries.

**What the pair measured** (Iris, 2026-11-20, `verify/decode_rate.py --train-prompt --tokens
512`, a quiet lane, cold prefill, every request at scale 0, C1; the last row is the
repository README's number for v0.31.0, taken by another script):

| Boot | tok/s | tokens per round | ms per round |
|---|---|---|---|
| v0.34.0 without the module | 429.1 | 3.69 | 8.58 |
| v0.34.0, published sidecar (462 kernels per round) | 376.7 (377.0 on a second run) | 3.69 | 9.77 |
| v0.34.0, tracer sidecar (528 kernels per round) | 369.9 | 3.69 | 9.95 |
| v0.31.0, the README's train-prompt figure | 410.4 | 3.80 | 9.25 |

So at scale 0 the hooks cost **1.19 ms per round**, 12.2 % of C1 tok/s, and all of it is
time per round: the drafter accepts as many tokens with the module as without. The stock
alias pays this for as long as the module is loaded.

The two steered boots give the same price per kernel: 1.19 ms over 462 kernels and 1.37 ms
over 528 are both 2.6 microseconds per kernel per round. That is the op-count model above,
measured. An inference from it, not a measurement: `fused` leaves 66 kernels per round, and
at the same price that is 0.17 ms, about 2 % of C1 instead of 12 %.
`verify/steering_kernel_check.py` on one GPU and then a boot with
`VLLM_STEERING_KERNEL=fused` are what would measure it.

Earlier the same day the hooks were bounded at 0.7 ms by comparing a steered v0.34.0 round
with a v0.31.0 step. That bound assumed nothing else had changed between the tags, and
v0.34.0 turned out 0.67 ms per round faster underneath. The first table of that day
(262.5 tok/s, 2.62 tokens per round) was on the script's essay prompt: tokens per round
follow the text, and those rows are not comparable with these.

**`VLLM_STEERING_KERNEL=fused`** replaces the 6 to 8 kernels of an output by one
(`pp2_steering_kernel.py`): 32 and 34 per pass instead of 224 and 238. A row at scale 0
returns before it reads its activations and is never written, so a prefill step of stock
requests costs one scale load per row. Steered values can differ from the torch path's by
one unit in the last place (the dot product is summed in another order); stock rows
cannot differ. It edits the hooked output in place.

**How to measure.**
- `verify/steering_kernel_check.py` on any idle GPU: the kernel against the torch path,
  then microseconds per graph replay of each stage's hooks at 6, 24 and 96 rows and
  milliseconds per eager pass at 8192 and 16384 rows, torch and fused. No model needed.
- On the lane, before any restart: `verify/decode_rate.py`. TTFT, tokens per second after
  the first token, and tokens and milliseconds per speculative round from the engine's
  own counters (`vllm:spec_decode_num_drafts`, `vllm/v1/spec_decode/metrics.py:229`;
  `vllm:generation_tokens`, `vllm/v1/metrics/loggers.py:799`). That separates "each
  round is slower" from "each round yields fewer tokens". `--train-prompt` sends the
  prompt the README's numbers are on. It says whether each prefill was cold.
- The gate stays C1 and C4 decode tok/s and TTFT at 32k with the module on at scale 0
  against a boot without it, measured by the same script on the same prompt.

The drafter is not steered. At scale 1 it drafts for a model it was not trained against:
acceptance and tok/s at scale 1 are their own measurement.

## Files

| File | |
|---|---|
| `pp2_steering.py` | the module; `python3 pp2_steering.py inspect <sidecar>` prints either format's inventory |
| `pp2_steering_kernel.py` | the fused kernel (Triton); imported only with `VLLM_STEERING_KERNEL=fused` |
| `test_pp2_steering.py` | CPU tests, stand-ins only: `python3 test_pp2_steering.py` or pytest |
| `stage_check.py` | before a boot, from the two files alone: will the module staged here load the sidecar staged here? One line, exit 0 or 1. The node script runs it; `--allow-unverified` as `PP2_STEERING_ALLOW_UNVERIFIED=1` |
| `SEAMS-v0.34.0.sha256` | the 23 v0.34.0 files the seams were read on |
| `verify/steering_seams.py` | static: are the seams in this image? |
| `verify/steering_image_smoke.py` | CPU: do the patches work on this image's real classes? |
| `verify/steering_kernel_check.py` | one idle GPU: is the fused kernel right, and what do the hooks cost? |
| `verify/sidecar_vs_weights.py` | CPU + disk: which float edit built the abliterated checkpoint, and the sidecar that applies it |
| `verify/steering_probe.py` | a booted lane: T1 stock identity, T2 live, T3 dial, T4 mixed batches, T5 both stages, T6 prefix cache, T7 validation, T8 decode steps, and whether anybody else was on the lane meanwhile. Its header says how each decides. |
| `verify/steering_probe_selftest.py` | the measure on hand-written lists, then the probe against toy servers: correct ones (with the pair's batch effect, with a sidecar that matters at few positions, with a tracer, with another client on the lane) and fourteen that are each broken in one way |
| `verify/make_tracer_sidecar.py` | a sidecar for a test boot whose effect nobody can miss (every steered output times 1/64); the module refuses it without `PP2_STEERING_ALLOW_UNVERIFIED=1`. T4's second instrument |
| `verify/decode_rate.py` | a booted lane: TTFT, tokens per second after the first token, tokens and milliseconds per speculative round |
| `verify/decode_rate_selftest.py` | that script against a toy lane whose three numbers are known |

## Knobs (`recipe/config.env`)

| Knob | |
|---|---|
| `VLLM_STEERING_SIDECAR` | the sidecar: an absolute host path, or one relative to the recipe directory. The same on both nodes. Empty = off, and the launch is byte for byte what it was. |
| `VLLM_STEERING_SHA256` | required; checked by the launcher on each node and by every engine process |
| `VLLM_STEERING_DEFAULT_SCALE` | scale of a request that sends none. Keep 0 on a shared lane. |
| `VLLM_STEERING_DEBUG` | 1 accepts `steering_ranks` (bitmask of stages to steer) and logs each steered request per rank. Test boots only. |
| `VLLM_STEERING_REPLAY` | `off` (default) or `keep`: see "Decoder bounded replay". On the pair's 20/20 split `keep` changes nothing. |
| `VLLM_STEERING_KERNEL` | `torch` (default) or `fused`: see "Cost". A boot with `fused` that cannot load or compile the kernel stops. |

`launch_cluster.sh` forwards all six to the remote rank. Two more exist only as container
environment (`EXTRA_ENV`), for experiments: `PP2_STEERING_ALLOW_TP=1` and
`PP2_STEERING_ALLOW_UNVERIFIED=1` (loads a derived sidecar that does not say `REBUILT`).

## What the pair showed on 2026-11-20

All of it is Iris's. The morning's runs were on the lane booted with revision 2 of the
module, the published sidecar, replay off, the torch kernel; the evening's on revision 4.

**Revision 3's probe** (254 requests, 77 s, `VLLM_STEERING_DEBUG=1`):

| | Result | What it establishes |
|---|---|---|
| T1 | omitted, `vllm_xargs` 0 and top-level 0: distance 0, identical text on 12/12 | the three spellings are one request, and asked alone the lane repeats itself exactly |
| T2, T3 | scale 1 moves 12/12 prompts (distance min 0.093, median 0.54); 0.5 differs from both | steering is live and graded |
| T5 | 12 prompts x 4 stage masks: each stage steers alone, both together equal scale 1 | both ranks apply their own layers |
| T6 | prompt of 5,955 tokens: 0 tokens from the cache across scales, 5,888 within a scale, both ways | the prefix cache does not cross scales, and it does reuse within one |
| T7 | 1.5, -0.1, 'abc', top-level 2: HTTP 400; 0 and 1: 200 | validation |
| refusals | 5/8 at scale 0, 0/8 at scale 1 (revision 2's run) | the published operator abliterates in the engine |
| T8 | decode against prefill: 0.063 at scale 0, 0.009 at scale 1, steering worth 0.47 there: INCONCLUSIVE | the numbers say decode steps are steered (0.009 against 0.47); revision 3's rule asked for an effect ten times the scale-0 figure and got 7.5 times. Revision 4 judges by which scale's prefill the token sits on. |
| T4 | INCONCLUSIVE: an all-stock burst alone moved a response by 0.38 | nothing about cross-talk. Requests sharing a batch change each other's output on this lane with no steering involved, by as much as steering moves many prompts. Revision 4's T4 is built around that. |
| T1b | "largest distance 0" against the baseline file | **nothing yet**: see below |

**T1b proves nothing until its baseline is known to come from a boot without the module.**
Revision 3 recorded a baseline on whatever lane it was pointed at. As reported, the only
boot of v0.34.0 without the module died before serving, so the file may have been
recorded on the steered lane. T1b then compares the lane with itself, and a distance of
exactly 0 is what that gives.
Revision 4 asks the lane for `steering_scale` 7 before recording and refuses when the
answer is HTTP 400; a file that does not carry that check is never a PASS. The evening's
baseline was recorded that way, on the boot without the module.

**Revision 2's probe**, the morning's run, failed T1, T4, T5 and T6. Those verdicts were
its own defects: a noise floor of exactly 0 from one repeat, a distance made of the far
tail of the first token's logprobs, a cache test with no control (a hit is not a cold run
in this engine: `vllm/config/cache.py:379`, `vllm/v1/kv_cache_interface.py:1045`,
`vllm/v1/core/sched/scheduler.py:3613`) and a prompt of eight repeated words. Its own
receipt already had the answer to T6 in its counters: 0 tokens across scales.

What the source says about the cache, which T6 has now confirmed on the lane: block
hashes are computed in one place, the engine core (`vllm/v1/engine/core.py:242`,
`vllm/v1/engine/core.py:1288`). No worker and no pipeline stage hashes. The salt is an
extra key of the first block (`vllm/v1/core/kv_cache_utils.py:657`) and chains to the
rest.

**The boot without the module** died: `Worker failed with error ''` on rank 0, in the same
second as rank 0's `Available KV cache memory` line, no traceback in rank 0's log. That is
what an assert without a message in rank 1's first forward looks like, and v0.34.0 has one
on a stage that begins at the last KV source layer. See "Decoder bounded replay" and
`patches/vllm-pp2-flag-v034/README.md`. Rank 1's log of that boot has not been read.

**The evening's boots** (Iris; revision 4 of the module and of the probe).

| Boot | What was reported |
|---|---|
| Without the module, with the replay-boundary guard | came up and served; the baseline was recorded on it; 429.1 tok/s at C1 on the train prompt |
| Published sidecar | T1: largest distance 0, identical text on 4/4. T2, T3, T1b, T6 and T7 pass. T5 not run (no debug boot). 376.7 tok/s. T4: INCONCLUSIVE twice, 16 judged positions and then 27 with twice the bursts |
| Tracer sidecar | a steered answer is 15 generated tokens with no content and no reasoning text; the text differed on 4/4. The probe read 0.008 between the scales, failed T2 and skipped T4. 369.9 tok/s at scale 0 |

**Why revision 4 could not decide T4**, all three of them the probe's doing:

1. *Its distance could not see a token that one list lacks.* It compared the tokens both
   top-20 lists name plus one bucket for the rest. Two lists with nothing in common are
   then 0 apart: on the tracer boot it read 0.008 where every steered answer was empty.
   With the published sidecar the same blind spot sits on the positions that matter
   most, those where steering replaces the leading token. `steering_probe_selftest.py`
   has both cases written out by hand, and revision 4's probe run against the new toy
   lanes gives the pair's two outcomes: "too few positions" and "T2 failed, T4 skipped".
2. *It did not aim.* It generated 16 tokens per burst request and judged whichever
   positions happened to matter: 16 of 768. Those runs also had the four harmless prompts
   only (identical text on 4/4), on which an abliteration is built to change little.
3. *One noise figure for all prompts.* The largest distance between any two repeats of any
   prompt set every prompt's bar at ten times itself. One disturbed repeat switches T2
   off and T4 with it, with the message "scale 1 moved no prompt": that is what the run
   with `--t4-tokens 24` printed, 20 seconds after T2 had passed. The option is read in one
   place, after T2 has been reported, and cannot have done it; a toy lane with a single
   disturbed request reproduces the message. What disturbed that run is not known here:
   its first lines ("noise used below") say.

Revision 5 measures what the two lists prove, screens the lane's own quiet generations
for the positions where the scales differ and aims every burst request at one, judges
each prompt against its own repeats, and reports from the engine's counters whether
anybody else was on the lane. A tracer boot is the second instrument if the serving
sidecar still leaves too few sites.

**One reading of the tracer boot is not explained.** T1 there read 0.1599, with a bar of
0.02 for T2 in the same run, so the noise was at most 0.002. T1 compares requests at scale
0 with requests that send no scale: no steered row is involved, and on the published
sidecar the same test read exactly 0. Either something else was on the lane during T1,
or a stock request that follows tracer requests is not the request it was before them.
The receipt of that run, or a rerun of T1 on a tracer boot with revision 5 (which says
whether the lane was quiet), decides between the two.

Still open: whether requests of different scales in flight together stay apart (T4).
Until T4 has passed on the pair, treat mixed batches as unproven.

## Not covered

- KV connectors (LMCache): the salted `cache_salt` is longer than the frontend's 128
  characters and has not been seen by a connector
  (`vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py:221`).
- Endpoints other than chat and completions get no validation; a malformed scale reaching
  the engine from elsewhere is served stock and logged.
- v0.31.0: the seams are there (`steering_seams.py` passes on that tree) and the module has
  no tag-specific code, but nothing loads it on that lane and the launcher refuses.
- If mixed batches misbehave, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=1`
  (`vllm/model_executor/layers/fused_moe/runner/shared_experts.py:74`) takes the auxiliary
  stream out of the picture for one diagnostic boot.
- The requests the drafter proposes for are verified by the steered target, so a wrong
  draft costs speed, not correctness. Nothing here measures that speed.
- Why requests that share a batch change each other's output on this lane. Asked alone
  it repeats itself exactly (T1); an all-stock burst moved a response by 0.38. The probe
  measures that and judges against it. Where it comes from was not established.
