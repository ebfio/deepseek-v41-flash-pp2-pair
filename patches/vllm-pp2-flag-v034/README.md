# vllm-pp2-flag-v034: the PP2 + DSpark overlay set for vLLM v0.34.0

Status: cut and checked against pristine `v0.34.0` (`62d4bc3c`) on 2026-11-19. **Booted on
the pair on 2026-11-20 together with the steering module**, which turns decoder replay
off. A boot of this set without that module died in rank 1's first forward the same day
("Worker failed with error ''"). The guard described under "Decoder replay on a stage that
begins at the last KV source layer" is the answer to that, and **a boot with the guard and
without the module came up and served the same evening** (as reported: 429.1 tok/s, 3.69
tokens and 8.58 ms per speculative round at C1 on the train prompt, `verify/decode_rate.py`).

Select it with `VLLM_OVERLAY_TAG=v0.34.0` and a v0.34.0 `VLLM_IMAGE`. The launcher
refuses to mount it on an image whose `dspark.py` is not the file it was cut from.

## What is in this directory

| File | What it is |
|---|---|
| `sitecustomize.py` | Fix 1, the `supports_aux_hidden_states_over_pp` flip. Same as v0.31.0, minus the autotune hook. Since revision 4 also the replay boundary guard. |
| `dspark.py` | Fix 2, the draft owns its embedding under PP>1. Pristine v0.34.0 file plus 19 lines. |
| `dspark.py.diff` | The same change as a diff against pristine v0.34.0. `git apply --check` clean, no offset, no fuzz. |
| `PRISTINE.sha256` | sha256 of the v0.34.0 files this set was cut from and read against. |
| `rebase_dspark.py` | Rebuilds `dspark.py` from a pristine file with exact anchors. For the next tag. |

## What happened to the four v0.31.0 overlays

| v0.31.0 overlay | On v0.34.0 | Evidence in v0.34.0 |
|---|---|---|
| `model_runner.py` (fix 3) | Upstream since 0.33.0. Do not mount. | `vllm/v1/worker/gpu/model_runner.py:2088-2092` |
| `cudagraph_utils.py` (fix 3) | Upstream since 0.33.0. Do not mount. | `vllm/v1/worker/gpu/cudagraph_utils.py:755-759` |
| `kv_cache_utils.py` (fix 4) | Upstream in 0.34.0 only, and moved. Do not mount. | `vllm/v1/core/kv_cache_planning.py:741-743`. On 0.33.0 the old loop is still at `kv_cache_planning.py:701`. |
| `dspark.py` (fix 2) | Still needed. Rebased: the import hunk is rewritten, three hunks re-anchor. | Share helper unchanged: `vllm/v1/worker/gpu/spec_decode/eagle/utils.py:83-115` |

And the two files that are not vLLM copies:

| File | On v0.34.0 | Evidence in v0.34.0 |
|---|---|---|
| `sitecustomize.py` (fix 1) | Still needed. | DSV4.1's `DeepseekV4Model` has no flag: `vllm/models/deepseek_v41/nvidia/model.py:715`. Gate: `vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:47-53`. |
| `pp_autotune.py` | Dead, and partly upstream. Never enable on this tag. | Per-stage tune and per-rank files: `vllm/model_executor/warmup/kernel_warmup.py:569-572, 600, 637-638`. The overlay calls `kw.resolve_flashinfer_autotune_file` and `kw._run_flashinfer_bf16_autotune_dummy_run`; neither is in that module now. |

### What the flag turns on in v0.34.0

Setting `supports_aux_hidden_states_over_pp` does two things on this tag. It passes the
gate, as on v0.31.0. It also makes `reserve_aux_intermediate_tensor_slots` read the
decoder's aux-slot layout on every rank after the first
(`eagle3_utils.py:69-100`, layout from `vllm/model_executor/models/interfaces.py:1635-1651`).
The slot count is the number of draft target layers at or below the stage's first
layer. For the 20/20 split with target layers 37-39 it is 0, and the function returns
at `eagle3_utils.py:85-86`.

The DSV4.1 decoder still never relays aux states between stages. Only the V4 decoder
calls `pack_local_aux_hidden_states` and `collect_remote_aux_hidden_states`
(`vllm/models/deepseek_v4/nvidia/model.py:999, 1039`). So the flag-only fix is exactly
as partition-dependent as the top-level README says, and the launcher's partition
refusal still applies.

As of vLLM main `5cd63b65` (2026-11-20) neither remaining fix is upstream: the DSV4.1
decoder has no flag and the draft has no PP branch for its embedding.

### What upstream did not take from pp_autotune.py

The overlay did three things: scope the tune's collectives to the TP group, save one
cache file per stage, and drop the world barrier. v0.34.0 does the first two
(`kernel_warmup.py:571, 611-612` and `:600, 637-638`). It keeps one `world.barrier()`
after the tune (`:630-631`), before the save.

Read from the code, not measured: that barrier is a wait, not the v0.31.0 hang. Both
ranks reach it once, so the stage that finishes first waits for the other. It can
still kill a boot when the wait exceeds the gloo timeout, 1800 s by default
(`vllm/config/parallel.py:408-410`). The likely way to get there is one node tuning
cold while the other loads its cache, because the cache files are per rank and live
on each node's own disk. The recipe passes `--cpu-distributed-timeout-seconds 7200` on
this tag for that case.

That value is part of the config hash that names the autotune cache directory:
`ParallelConfig.compile_factors` does not ignore it (`vllm/config/parallel.py:1011-1053`).
Changing it later forces a cold tune, so set it once before the first baseline.

## Decoder replay on a stage that begins at the last KV source layer

v0.33.0 added decoder-side SWA bounded replay: in prefill, the layers from the last KV
source layer on run on each request's last window only. v0.31.0 has none of it, so the
v0.31.0 lane never met this.

**What v0.34.0 does on such a stage.** `cut` is the checkpoint's last KV source layer
(`vllm/models/deepseek_v41/nvidia/model.py:882`). Replay is turned on when the stage
holds `cut` and everything after it; the test refuses `self.start_layer > cut`
(`vllm/models/deepseek_v41/nvidia/model.py:1332`), so a stage that begins exactly at
`cut` passes. `forward` then runs the layers before the replay layers,
`range(self.start_layer, self.decoder_replay_start)`
(`vllm/models/deepseek_v41/nvidia/model.py:1120`), which is empty on that stage, with
`residual`, `post_mix` and `res_mix` still `None`
(`vllm/models/deepseek_v41/nvidia/model.py:1111`). `_run_layers` ends in
`assert residual is not None and post_mix is not None`
(`vllm/models/deepseek_v41/nvidia/model.py:1247`). An ordinary first layer of a stage
handles `residual is None` itself (`vllm/models/deepseek_v41/nvidia/model.py:540`); the
replay path's `write_kv` has no such branch
(`vllm/models/deepseek_v41/nvidia/model.py:680`). So it is not one stray assert: the
replay path was not written for a stage that begins at `cut`.

**How that looks from outside.** The first forward of a rank is the memory profile
(`vllm/v1/worker/gpu_worker.py:732`, `vllm/v1/worker/gpu/model_runner.py:1086`). The
worker logs the traceback in its own process
(`vllm/v1/executor/multiproc_executor.py:1223`) and sends the engine core only
`str(exception)` (`vllm/v1/executor/multiproc_executor.py:1155`), which for a bare
assert is the empty string. The engine core prints
`Worker failed with error ''` (`vllm/v1/executor/multiproc_executor.py:550`) on rank 0's
host, right after rank 0's own `Available KV cache memory` line
(`vllm/v1/worker/gpu_worker.py:796`), and the traceback is in rank 1's log on rank 1's
host.

**On this pair.** The split is 20/20, so rank 1 holds layers 20-39. It is such a stage
if the largest of the checkpoint's `kv_source_layer_ids` is 20. That value was not read
from the checkpoint by the author of this set; one line on the box settles it, and did
(`[2, 8, 14, 20]`, see "Confirmed on the pair" below). It looks inside nested configs too:

    python3 -c "import json,sys; f=lambda d,p='': [print(p+k, v) if k in ('kv_source_layer_ids','sliding_window','num_hidden_layers') else f(v, p+k+'.') if isinstance(v, dict) else None for k, v in d.items()]; f(json.load(open(sys.argv[1])))" <model dir>/config.json

**The guard.** `sitecustomize.py` wraps `DeepseekV4Model._decoder_replay_supported`: when
upstream answers "supported" and `self.start_layer == cut`, the answer becomes "not
supported", with one `[pp2_flag]` line on stderr. Every other stage gets upstream's own
answer. The rank then runs all its layers on every row, which is what the steering module
has been making it do. `EXTRA_ENV='VLLM_PATCH_PP_REPLAY_BOUNDARY=0'` removes the guard.

**Checked without a GPU**, on the real v0.34.0 class: `verify/step0_import_smoke.py` asks
the method about four made-up stages (guarded: 20-39 off, 16-39 on, 0-39 on, 24-39 off;
upstream alone: 20-39 on) and runs upstream's own `_run_layers` on the empty range, which
raises `AssertionError` with the message `''`. `verify/step0_overlays.py` says whether a
tree still has the three pieces.

**Confirmed on the pair** (Iris, 2026-11-20). The checkpoint's
`text_config.kv_source_layer_ids` is `[2, 8, 14, 20]`, so the cut is 20 and rank 1 holds
layers 20 to 39. Rank 1's log of the boot that died
(`/opt/gb300-pair/logs/live-rank1-20261120T131629.log` on hades) has the traceback:
`(Worker_PP1 pid=341) ERROR [multiproc_executor.py:1223]`, `assert residual is not None and
post_mix is not None`, `AssertionError`; rank 0's has `Available KV cache memory: 80.39
GiB` and then `RuntimeError: Worker failed with error ''`. A boot with the guard and
without the steering module came up and served.

**What it means for replay on this pair.** At 20/20 there is no decoder replay to have:
upstream cannot run it there. A split that puts layer 19 on rank 1
(`VLLM_PP_LAYER_PARTITION=19,21`) would let upstream replay, on a lane that is then not
the measured one. Reasoning, not a measurement: in a two-stage pipeline the slower stage
sets the pace of a prefill, and replay only shortens the second stage.

Upstream should hear of it: either `_decoder_replay_supported` refuses
`self.start_layer >= cut`, or `forward` and `write_kv` handle a stage that begins there.

## The shipped `dspark.py.diff` for v0.31.0 is stale

`patches/vllm-pp2-flag/dspark.py.diff` has three hunks. The whole file next to it has
four: the diff is missing the post-load assert. This set was rebased from the whole
file, so the assert is here.

## Verify

Inside the image, on each node:

    docker run --rm --entrypoint python3 \
        -v "$PWD/patches/vllm-pp2-flag-v034:/overlay:ro" -v "$PWD/verify:/verify:ro" \
        vllm/vllm-openai:v0.34.0 /verify/step0_overlays.py --overlay-dir /overlay --json -

`STEP0 PASS` means the image's files are the ones this set was cut for, the diff
reproduces the whole file byte for byte, and the three upstream fixes are present.
It says nothing about whether the lane boots.

Then check that `sitecustomize.py` really sets the flag in that image, and that it
costs nothing at interpreter start. No GPU is needed, so this can run while the box
serves:

    docker run --rm --entrypoint python3 \
        -e PYTHONPATH=/opt/pp2-flag \
        -v "$PWD/patches/vllm-pp2-flag-v034:/opt/pp2-flag:ro" -v "$PWD/verify:/verify:ro" \
        vllm/vllm-openai:v0.34.0 /verify/step0_import_smoke.py

`STEP0-IMPORT PASS` means a fresh interpreter starts without importing torch or vllm,
both inner decoder classes carry the flag once their modules are imported, and the
replay boundary guard answers as described above. It loads no weights.

### Why the flag is set from an import hook

`sitecustomize.py` runs in every Python process of the container. The v0.31.0 file
imports the two model modules right there, so every process pays a full vllm + torch
import before its first line: measured at 8.9 s per process in the 0.34.0 image
(0.55 s to 9.48 s, on the boxes, 2026-11-20) and at 15.5 s against the v0.34.0 source
tree on a CPU-only build host. This set registers a `sys.meta_path` hook instead and
sets the flag when vLLM imports the module: start-up on that build host went back to
0.02 s, and the flag is `True` on both inner classes. The v0.31.0 set
(`patches/vllm-pp2-flag/`) is untouched and still pays the tax.

## Not verified

- The guard's own log line (`[pp2_flag] layers 20-39: decoder SWA bounded replay is off
  for this stage`) and rank 0's KV cache size on the boot without the module: that boot
  served and was timed, and neither line was in the report.
- GSM8K, the soak and the parity sweep on v0.34.0: not run.
- By the author: anything in the image or on a GPU. `step0_overlays.py` and
  `step0_import_smoke.py` were run in the image on the boxes (2026-11-20, revision 3);
  the guard's checks in them are new in revision 4.
