#!/usr/bin/env bash
# Start one vLLM rank of the two-node DeepSeek-V4.1-Flash server on this host (pipeline parallel by default).
# Usage: NODE_RANK=0|1 ./serve_vllm_node.sh   (normally invoked by launch_cluster.sh with ENGINE=vllm)
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/config.env"
# Knobs newer than the config.env a box may still carry (it is rendered by the role, not
# copied from this repo). Under `set -u` a missing one aborts the launch, and a missing
# default changes the lane without a word, so the launcher owns them. An older config.env
# therefore launches the v0.31.0 lane exactly as before.
: "${VLLM_OVERLAY_TAG:=v0.31.0}" "${VLLM_OVERLAY_GUARD:=enforce}" "${VLLM_PATCH_PP_AUTOTUNE:=0}"
[[ "$VLLM_OVERLAY_TAG" == v0.31.0 ]] || : "${VLLM_CPU_DISTRIBUTED_TIMEOUT_S=7200}"
: "${VLLM_CPU_DISTRIBUTED_TIMEOUT_S:=}" "${VLLM_SPEC_EXTRA:=}"
: "${VLLM_STEERING_SIDECAR:=}" "${VLLM_STEERING_SHA256:=}" "${VLLM_STEERING_DEFAULT_SCALE:=0}" "${VLLM_STEERING_DEBUG:=0}"
: "${VLLM_STEERING_REPLAY:=off}" "${VLLM_STEERING_KERNEL:=torch}"

node_rank="${NODE_RANK:?Set NODE_RANK to 0 or 1}"
case "$node_rank" in
  0) model_dir="$MODEL_DIR_RANK0"; node_ip="$RANK0_IP" ;;
  1) model_dir="$MODEL_DIR_RANK1"; node_ip="$RANK1_IP" ;;
  *) echo "NODE_RANK must be 0 or 1" >&2; exit 2 ;;
esac
[[ -s "$model_dir/model.safetensors.index.json" ]] || { echo "Checkpoint index missing at $model_dir" >&2; exit 1; }
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Container $CONTAINER_NAME already exists on $(hostname); run stop_cluster.sh first." >&2; exit 1
fi
for dev in /dev/infiniband/uverbs0 /dev/infiniband/uverbs1; do [[ -c "$dev" ]] || { echo "Missing RDMA device $dev" >&2; exit 1; }; done
gpu_uuid="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', *' '$2 ~ /GB300/ {print $1}')"
[[ "$(wc -l <<<"$gpu_uuid")" -eq 1 && -n "$gpu_uuid" ]] || { echo "Expected exactly one GB300 on $(hostname)" >&2; exit 1; }

case "$VLLM_PARALLEL" in
  pp) tp_size=1; pp_size=2 ;;
  tp) tp_size=2; pp_size=1 ;;
  *) echo "VLLM_PARALLEL must be pp or tp" >&2; exit 2 ;;
esac

case "$VLLM_OVERLAY_TAG" in
  v0.31.0) cache_dir="$CACHE_ROOT/vllm-dsv41" ;;
  # Every other tag gets its own caches: root-cache is seeded once from the image's
  # /root/.cache (prebuilt FlashInfer JIT artifacts), so sharing it across images
  # serves one image the other's seed, and it keeps the v0.31.0 caches intact for rollback.
  v0.34.0) cache_dir="$CACHE_ROOT/vllm-dsv41-$VLLM_OVERLAY_TAG" ;;
  *) echo "VLLM_OVERLAY_TAG must be v0.31.0 or v0.34.0" >&2; exit 2 ;;
esac
mkdir -p "$cache_dir"/{root-cache,triton,tilelang,nv} "$script_dir/logs"
# Seed the persistent cache from the image's own /root/.cache (prebuilt FlashInfer JIT artifacts) on first use.
if [[ ! -e "$cache_dir/root-cache/.seeded" ]]; then
  docker run --rm --volume "$cache_dir/root-cache:/mnt" --entrypoint bash "$VLLM_IMAGE" -c 'cp -a /root/.cache/. /mnt/ && touch /mnt/.seeded'
fi

docker_env=(
  --env "VLLM_HOST_IP=$node_ip"
  # 7200: a cold FlashInfer autotune boot can exceed an hour; 3600 kills it
  # mid-tune. Paired with VLLM_PATCH_PP_AUTOTUNE=1 (see README).
  --env "VLLM_ENGINE_READY_TIMEOUT_S=7200"
  --env "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"
)
# v0.31.0 only. Upstream tunes per PP stage since 0.33, and pp_autotune.py calls
# kernel_warmup names that 0.34.0 no longer has, so the hook is not shipped there.
[[ "$VLLM_OVERLAY_TAG" == v0.31.0 ]] && docker_env+=(--env "VLLM_PATCH_PP_AUTOTUNE=$VLLM_PATCH_PP_AUTOTUNE")
docker_env+=(
  --env "VLLM_USE_RUST_FRONTEND=$VLLM_RUST_FRONTEND"
  --env "VLLM_ALLREDUCE_USE_SYMM_MEM=0"
  --env "VLLM_ALLREDUCE_USE_FLASHINFER=0"
  --env "VLLM_USE_NCCL_SYMM_MEM=0"
  --env "VLLM_DEEP_GEMM_WARMUP=$VLLM_DEEP_GEMM_WARMUP"
  --env "GLOO_SOCKET_IFNAME=$FABRIC_IFACE"
  --env "NCCL_SOCKET_IFNAME=$FABRIC_IFACE"
  --env "NCCL_IB_HCA=$NCCL_HCAS"
  --env "NCCL_IB_DISABLE=0"
  --env "NCCL_NET_GDR_LEVEL=SYS"
  --env "NCCL_DMABUF_ENABLE=1"
  --env "NCCL_MNNVL_ENABLE=0"
  --env "NCCL_CUMEM_ENABLE=0"
  --env "NCCL_DEBUG=$NCCL_DEBUG_LEVEL"
  --env "NCCL_DEBUG_SUBSYS=$NCCL_DEBUG_SUBSYS"
)
if [[ "$NCCL_TUNING" == tuned ]]; then
  # PROTO=SIMPLE forced: auto-selection (LL/LL128) crashes with
  # "unhandled cuda error" on this RoCE+QPS fabric (verified 2026-10-23);
  # re-test protocols only via the nccl-tests bracket first
  docker_env+=(--env "NCCL_ALGO=RING" --env "NCCL_PROTO=SIMPLE" --env "NCCL_MIN_NCHANNELS=8" --env "NCCL_MAX_NCHANNELS=8"
               --env "NCCL_IB_QPS_PER_CONNECTION=4" --env "NCCL_IB_SPLIT_DATA_ON_QPS=1")
fi
[[ -n "$VLLM_PP_LAYER_PARTITION" ]] && docker_env+=(--env "VLLM_PP_LAYER_PARTITION=$VLLM_PP_LAYER_PARTITION")
while IFS='=' read -r k v; do [[ -n "$k" ]] && docker_env+=(--env "$k=$v"); done <<<"${EXTRA_ENV:-}"

extra_mounts=()
if [[ "$HOST_RDMA" == 1 ]]; then
  ibv_real="$(docker run --rm --entrypoint readlink "$VLLM_IMAGE" -f /usr/lib/aarch64-linux-gnu/libibverbs.so.1)"
  mlx5_real="$(docker run --rm --entrypoint readlink "$VLLM_IMAGE" -f /usr/lib/aarch64-linux-gnu/libmlx5.so.1)"
  [[ -n "$ibv_real" && -n "$mlx5_real" ]] || { echo "Could not resolve rdma-core library names in $VLLM_IMAGE" >&2; exit 1; }
  extra_mounts+=(
    --volume "$HOST_RDMA_DIR/libibverbs.so.1:$ibv_real:ro"
    --volume "$HOST_RDMA_DIR/libmlx5.so.1:$mlx5_real:ro"
    --volume /usr/lib/aarch64-linux-gnu/libibverbs:/usr/lib/aarch64-linux-gnu/libibverbs:ro
    --volume /etc/libibverbs.d:/etc/libibverbs.d:ro
  )
fi

# Optional source patches bind-mounted over the image (documented in patches/vllm/README.md).
if [[ "$VLLM_PATCH_PP" == 1 ]]; then
  [[ -s "$script_dir/patches/dsv41_pp2/deepseek_v4_nvidia_model.py" ]] || { echo "missing patches/dsv41_pp2/deepseek_v4_nvidia_model.py" >&2; exit 1; }
  extra_mounts+=(--volume "$script_dir/patches/dsv41_pp2/deepseek_v4_nvidia_model.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/model.py:ro")
  [[ -s "$script_dir/patches/dsv41_pp2/kv_cache_utils.py" ]] || { echo "missing patches/dsv41_pp2/kv_cache_utils.py" >&2; exit 1; }
  extra_mounts+=(--volume "$script_dir/patches/dsv41_pp2/kv_cache_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py:ro")
fi
# PP2 one-liner flag overlay (patches/vllm-pp2-flag/README.md): the ClassVar
# flip only; the 5-file port (dsv41_pp2_dspark) is the fallback, never both.
if [[ "${VLLM_PATCH_PP2_FLAG:-0}" == 1 ]]; then
  site_pkgs=/usr/local/lib/python3.12/dist-packages
  # The whole-file overlays are copies of one vLLM tag's sources plus small edits,
  # so each set names the files it replaces and may only be mounted on that tag.
  case "$VLLM_OVERLAY_TAG" in
    v0.31.0)
      # Mainline-aligned PP2+DSpark increments (2026-10-25): the ClassVar
      # flip (sitecustomize) + draft-own-embed (dspark.py) + keep input_ids on
      # non-first PP ranks for vision MoE routing (model_runner.py,
      # cudagraph_utils.py) + KV-group filtering under PP (kv_cache_utils.py:
      # without it, allocate_kv_cache raises StopIteration on a PP rank that
      # owns none of a group's layers; receipt 2026-10-25 attempt 4).
      overlay_dir="$script_dir/patches/vllm-pp2-flag"
      overlay_files=(
        "dspark.py:vllm/models/deepseek_v41/nvidia/dspark.py"
        "model_runner.py:vllm/v1/worker/gpu/model_runner.py"
        "cudagraph_utils.py:vllm/v1/worker/gpu/cudagraph_utils.py"
        "kv_cache_utils.py:vllm/v1/core/kv_cache_utils.py"
      ) ;;
    v0.34.0)
      # The input_ids gate (since 0.33) and the KV-group filter (0.34, now in
      # kv_cache_planning.py) are upstream. Only the flag flip and the
      # draft-own-embed file are left; see patches/vllm-pp2-flag-v034/README.md.
      overlay_dir="$script_dir/patches/vllm-pp2-flag-v034"
      overlay_files=("dspark.py:vllm/models/deepseek_v41/nvidia/dspark.py")
      if [[ "${VLLM_PATCH_PP_AUTOTUNE:-0}" == 1 ]]; then
        echo "VLLM_PATCH_PP_AUTOTUNE=1 is v0.31.0-only: 0.34.0 tunes per PP stage itself and the overlay would crash warmup. Unset it." >&2
        exit 2
      fi ;;
  esac
  [[ -s "$overlay_dir/sitecustomize.py" ]] || { echo "missing $overlay_dir/sitecustomize.py" >&2; exit 1; }
  extra_mounts+=(--volume "$overlay_dir:/opt/pp2-flag:ro")
  docker_env+=(--env "PYTHONPATH=/opt/pp2-flag")
  for spec in "${overlay_files[@]}"; do
    src="$overlay_dir/${spec%%:*}"; rel="${spec#*:}"
    [[ -s "$src" ]] || { echo "missing $src" >&2; exit 1; }
    # A set that ships PRISTINE.sha256 only replaces the exact file it was cut from.
    if [[ -s "$overlay_dir/PRISTINE.sha256" ]]; then
      want="$(awk -v f="$rel" '$2 == f {print $1}' "$overlay_dir/PRISTINE.sha256")"
      # A probe that fails (file absent in the image, docker error) counts as a mismatch.
      got="$(docker run --rm --entrypoint sha256sum "$VLLM_IMAGE" "$site_pkgs/$rel" 2>/dev/null | awk '{print $1}' || true)"
      if [[ -z "$want" || "$want" != "$got" ]]; then
        echo "$VLLM_IMAGE: $rel is not the $VLLM_OVERLAY_TAG file this overlay was cut from (want ${want:-<unlisted>}, got ${got:-<none>})." >&2
        [[ "${VLLM_OVERLAY_GUARD:-enforce}" == warn ]] || { echo "Refusing to mount it (VLLM_OVERLAY_GUARD=warn overrides)." >&2; exit 1; }
      fi
    fi
    extra_mounts+=(--volume "$src:$site_pkgs/$rel:ro")
  done
fi
# Per-request activation steering (patches/pp2-steering/README.md). The module is loaded by
# the v0.34.0 overlay's sitecustomize, in every Python process of the container, and ends the
# process if it cannot install. Nothing below runs, and nothing is mounted, while the sidecar
# knob is empty.
if [[ -n "${VLLM_STEERING_SIDECAR:-}" ]]; then
  steer_dir="$script_dir/patches/pp2-steering"
  steer_refuse() { echo "steering: $*. Refusing to launch." >&2; exit 2; }
  [[ "$VLLM_OVERLAY_TAG" == v0.34.0 ]] || steer_refuse "wired into the v0.34.0 lane only (VLLM_OVERLAY_TAG=$VLLM_OVERLAY_TAG)"
  [[ "$VLLM_PARALLEL" == pp && "${VLLM_PATCH_PP2_FLAG:-0}" == 1 ]] || steer_refuse "needs VLLM_PARALLEL=pp and VLLM_PATCH_PP2_FLAG=1 (that overlay's sitecustomize loads the module)"
  [[ "${VLLM_PATCH_GRAPH_ATTR:-0}" != 1 ]] || steer_refuse "VLLM_PATCH_GRAPH_ATTR=1 replaces PYTHONPATH, so the module would never load"
  [[ "$VLLM_RUST_FRONTEND" == 0 ]] || steer_refuse "not validated with the Rust frontend (VLLM_RUST_FRONTEND=$VLLM_RUST_FRONTEND)"
  # Steering edits activations of the stock model. A stage that loads an already abliterated
  # checkpoint is edited twice at scale 1 and is not stock at scale 0, and nothing would say so.
  [[ "$MODEL_DIR_RANK0" == "$MODEL_DIR_RANK1" ]] || steer_refuse "both stages must load the same stock checkpoint (MODEL_DIR_RANK0=$MODEL_DIR_RANK0, MODEL_DIR_RANK1=$MODEL_DIR_RANK1)"
  [[ -s "$steer_dir/pp2_steering.py" ]] || steer_refuse "missing $steer_dir/pp2_steering.py"
  # docker reads a --volume source that does not start with a slash as the NAME of a
  # volume: it creates one and mounts an empty directory, and the checks below, which run
  # here on the host, pass all the same (2026-11-20: the engine died on IsADirectoryError).
  # So the path is made absolute once, before anything reads it. A relative path is taken
  # from this script's directory, where config.env lives: the same file on both nodes
  # whatever the caller's working directory.
  [[ "$VLLM_STEERING_SIDECAR" == /* ]] || VLLM_STEERING_SIDECAR="$script_dir/$VLLM_STEERING_SIDECAR"
  [[ -f "$VLLM_STEERING_SIDECAR" && -s "$VLLM_STEERING_SIDECAR" ]] || steer_refuse "no sidecar file at $VLLM_STEERING_SIDECAR on $(hostname)"
  VLLM_STEERING_SIDECAR="$(realpath -e -- "$VLLM_STEERING_SIDECAR")"
  [[ "${VLLM_STEERING_SHA256:-}" =~ ^[0-9a-f]{64}$ ]] || steer_refuse "VLLM_STEERING_SHA256 must be the sidecar's sha256 (64 lowercase hex)"
  steer_got="$(sha256sum "$VLLM_STEERING_SIDECAR" | awk '{print $1}')"
  [[ "$steer_got" == "$VLLM_STEERING_SHA256" ]] || steer_refuse "sidecar on $(hostname) has sha256 $steer_got, expected $VLLM_STEERING_SHA256"
  [[ "${VLLM_STEERING_DEFAULT_SCALE:-0}" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || steer_refuse "VLLM_STEERING_DEFAULT_SCALE must be a number in [0, 1]"
  [[ "${VLLM_STEERING_DEBUG:-0}" =~ ^[01]$ ]] || steer_refuse "VLLM_STEERING_DEBUG must be 0 or 1"
  [[ "$VLLM_STEERING_REPLAY" =~ ^(off|keep)$ ]] || steer_refuse "VLLM_STEERING_REPLAY must be off or keep"
  [[ "$VLLM_STEERING_KERNEL" =~ ^(torch|fused)$ ]] || steer_refuse "VLLM_STEERING_KERNEL must be torch or fused"
  [[ "$VLLM_STEERING_KERNEL" == torch || -s "$steer_dir/pp2_steering_kernel.py" ]] || steer_refuse "VLLM_STEERING_KERNEL=fused needs $steer_dir/pp2_steering_kernel.py"
  # Does the module staged here read the sidecar staged here? The module answers that
  # when the model is built, minutes into a boot (2026-11-20: a tracer sidecar next to
  # revision 2's module, which reads only the published format, cost a nine-minute boot
  # and a rollback). stage_check.py answers it now from the two files, and refuses only
  # what the module would refuse. Skipped on a host without python3.
  if [[ -s "$steer_dir/stage_check.py" ]] && command -v python3 >/dev/null 2>&1; then
    steer_allow=0
    while IFS='=' read -r k v; do
      if [[ "$k" == PP2_STEERING_ALLOW_UNVERIFIED && "$v" == 1 ]]; then steer_allow=1; fi
    done <<<"${EXTRA_ENV:-}"
    steer_said="$(PP2_STEERING_ALLOW_UNVERIFIED="$steer_allow" python3 -B "$steer_dir/stage_check.py" --module "$steer_dir/pp2_steering.py" --sidecar "$VLLM_STEERING_SIDECAR" 2>&1)" \
      || steer_refuse "on $(hostname): ${steer_said:-stage_check.py did not run}"
    echo "steering: $steer_said" >&2
  fi
  extra_mounts+=(
    --volume "$steer_dir:/opt/pp2-steering:ro"
    --volume "$VLLM_STEERING_SIDECAR:/opt/pp2-steering-sidecar/steering.safetensors:ro"
  )
  docker_env+=(
    --env "PP2_STEERING_SIDECAR=/opt/pp2-steering-sidecar/steering.safetensors"
    --env "PP2_STEERING_SHA256=$VLLM_STEERING_SHA256"
    --env "PP2_STEERING_DEFAULT_SCALE=${VLLM_STEERING_DEFAULT_SCALE:-0}"
    --env "PP2_STEERING_DEBUG=${VLLM_STEERING_DEBUG:-0}"
  )
  # Passed only when they leave the default, so a launch that sets neither is the
  # command line it was before these two knobs existed.
  [[ "$VLLM_STEERING_REPLAY" == off ]] || docker_env+=(--env "PP2_STEERING_REPLAY=$VLLM_STEERING_REPLAY")
  [[ "$VLLM_STEERING_KERNEL" == torch ]] || docker_env+=(--env "PP2_STEERING_KERNEL=$VLLM_STEERING_KERNEL")
fi
# One-shot 2-rank all-reduce overlay (patches/vllm-oneshot-ar/README.md).
# Flip only after the nccl-tests allgather-vs-allreduce bracket passes on this fabric.
if [[ "${VLLM_PATCH_ONESHOT_AR:-0}" == 1 ]]; then
  [[ -s "$script_dir/patches/vllm-oneshot-ar/cuda_communicator.py" ]] || { echo "missing patches/vllm-oneshot-ar/cuda_communicator.py" >&2; exit 1; }
  extra_mounts+=(--volume "$script_dir/patches/vllm-oneshot-ar/cuda_communicator.py:/usr/local/lib/python3.12/dist-packages/vllm/distributed/device_communicators/cuda_communicator.py:ro")
  docker_env+=(--env "VLLM_ONESHOT_AR_2RANK=1")
fi
# Graph attribution overlay (one-boot diagnostic; patches/vllm-graph-attribution/README.md).
# PYTHONPATH makes Python import sitecustomize.py at startup; it wraps CUDAGraph
# replay/capture_end to log call sites. Never leave enabled on the production lane.
if [[ "${VLLM_PATCH_GRAPH_ATTR:-0}" == 1 ]]; then
  [[ -s "$script_dir/patches/vllm-graph-attribution/sitecustomize.py" ]] || { echo "missing patches/vllm-graph-attribution/sitecustomize.py" >&2; exit 1; }
  extra_mounts+=(--volume "$script_dir/patches/vllm-graph-attribution:/opt/graph-attr:ro")
  docker_env+=(--env "PYTHONPATH=/opt/graph-attr")
fi
args=(
  serve /model
  --served-model-name "$SERVED_MODEL_NAME"
  --trust-remote-code
  --tokenizer-mode deepseek_v41
  --tensor-parallel-size "$tp_size" --pipeline-parallel-size "$pp_size"
  --distributed-executor-backend mp --nnodes 2 --node-rank "$node_rank"
  --master-addr "$RANK0_IP" --master-port "$VLLM_MASTER_PORT"
  --disable-custom-all-reduce
  --max-model-len "$CONTEXT_LENGTH"
  --max-num-seqs "$MAX_RUNNING_REQUESTS"
  --max-num-batched-tokens "$CHUNKED_PREFILL_SIZE"
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --engram-config "{\"cpu_offload\": $( [[ "$ENGRAM_HOST_TABLE" == 1 ]] && echo true || echo false ), \"use_thp\": $( [[ "${ENGRAM_THP:-1}" == 1 ]] && echo true || echo false )}"
  --reasoning-parser deepseek_v41
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice
)
[[ "$VLLM_LANGUAGE_MODEL_ONLY" == 1 ]] && args+=(--language-model-only)
[[ "$VLLM_PREFIX_CACHING" == 1 ]] && args+=(--enable-prefix-caching) || args+=(--no-enable-prefix-caching)
[[ "$VLLM_FLASHINFER_AUTOTUNE" == 0 ]] && args+=(--no-enable-flashinfer-autotune)
[[ "$VLLM_ASYNC_SCHEDULING" == 0 ]] && args+=(--no-async-scheduling)
[[ -n "${VLLM_KV_CACHE_DTYPE:-}" ]] && args+=(--kv-cache-dtype "$VLLM_KV_CACHE_DTYPE")
[[ -n "${VLLM_ATTENTION_CONFIG:-}" ]] && args+=(--attention-config "$VLLM_ATTENTION_CONFIG")
# prefill/TBT knob: when set, long prefills are yielded at this threshold
[[ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" && "${LONG_PREFILL_TOKEN_THRESHOLD}" != "0" ]] && args+=(--long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD")
[[ -n "${VLLM_COMPILATION_CONFIG:-}" ]] && args+=(--compilation-config "$VLLM_COMPILATION_CONFIG")
[[ "$VLLM_ENFORCE_EAGER" == 1 ]] && args+=(--enforce-eager)
[[ -n "${VLLM_OVERRIDE_GENERATION_CONFIG:-}" ]] && args+=(--override-generation-config "$VLLM_OVERRIDE_GENERATION_CONFIG")
[[ -n "${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]] && args+=(--default-chat-template-kwargs "$VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS")
# gloo group timeout (default 1800 s). A cold autotune ends in a world barrier, and
# the rank that finishes first waits there for the rest of the other stage's tune.
# The value feeds the config hash that names the autotune cache directory.
[[ -n "${VLLM_CPU_DISTRIBUTED_TIMEOUT_S:-}" ]] && args+=(--cpu-distributed-timeout-seconds "$VLLM_CPU_DISTRIBUTED_TIMEOUT_S")
case "$MODE" in
  throughput) ;;
  low-latency)
    # adaptive can't help at 1-2 streams (verify cost
    # doesn't scale with tokens at C1) and it launders warmup noise into
    # steady-state draft budgets (the 69-tok/s sick-boot receipt). Off on
    # the production lane; flip per-batch-lane if one ever exists.
    adaptive=$( [[ "${VLLM_ADAPTIVE_VERIFICATION:-0}" == 1 ]] && echo true || echo false )
    if [[ "$VLLM_PARALLEL" == pp && "${VLLM_PATCH_PP2_FLAG:-0}" != 1 ]]; then
      # Stock 0.31.0 rejects DSpark+PP for DeepseekV41 at load_model
      # (the eagle3_utils verify). The overlay must gate the launch.
      echo "DSpark under PP needs VLLM_PATCH_PP2_FLAG=1 (the overlay); refusing stock launch" >&2
      exit 2
    fi
    if [[ "$VLLM_PARALLEL" == pp && -n "${VLLM_PP_LAYER_PARTITION:-}" ]]; then
      # The flag-only fix requires every draft target layer (37-39) on the
      # last stage. Refuse partitions whose last stage starts above 37:
      # the drafts would silently degrade (acceptance collapse, no error).
      first_layers=$(echo "$VLLM_PP_LAYER_PARTITION" | awk -F, '{s=0; for(i=1;i<NF;i++) s+=$i; print s}')
      if [[ "$first_layers" -gt 37 ]]; then
        echo "partition $VLLM_PP_LAYER_PARTITION starts the last stage at layer $((first_layers+1)); the draft needs layers 37-39 there. Refusing." >&2
        exit 2
      fi
    fi
    # VLLM_SPEC_EXTRA: extra speculative-config keys as a JSON fragment without braces, e.g.
    #   "draft_lm_head_quantization":"fp8"   or   "num_speculative_tokens_per_batch_size":[[1,4,5],[5,7,4],[8,128,3]]
    # so a draft-side experiment is a config change on both ranks, not a script edit on one.
    args+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${VLLM_DSPEC_TOKENS:-5},\"draft_sample_method\":\"${VLLM_DRAFT_SAMPLE_METHOD:-probabilistic}\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":$adaptive${VLLM_SPEC_EXTRA:+,$VLLM_SPEC_EXTRA}}") ;;
  *) echo "MODE must be throughput or low-latency" >&2; exit 2 ;;
esac
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)
if [[ "$node_rank" == 0 ]]; then args+=(--host "$API_HOST" --port "$API_PORT"); else args+=(--headless); fi

# A --volume source without a leading slash is a volume name to docker, not a path.
for ((i = 0; i + 1 < ${#extra_mounts[@]}; i++)); do
  [[ "${extra_mounts[i]}" == --volume ]] || continue
  mount_src="${extra_mounts[i + 1]%%:*}"
  [[ "$mount_src" == /* ]] || { echo "Refusing to launch: --volume source '$mount_src' is not an absolute path; docker would create a volume of that name and mount an empty directory." >&2; exit 2; }
done
for mount_src in "$model_dir" "$cache_dir"; do
  [[ "$mount_src" == /* ]] || { echo "Refusing to launch: '$mount_src' is not an absolute path; docker would create a volume of that name." >&2; exit 2; }
done

docker run --detach \
  --name "$CONTAINER_NAME" \
  --device "nvidia.com/gpu=$gpu_uuid" \
  --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 \
  --ipc host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --cap-add SYS_NICE \
  --volume "$model_dir:/model:ro" \
  --volume "$cache_dir/root-cache:/root/.cache" \
  --volume "$cache_dir/triton:/root/.triton" \
  --volume "$cache_dir/tilelang:/root/.tilelang" \
  --volume "$cache_dir/nv:/root/.nv" \
  "${extra_mounts[@]}" \
  "${docker_env[@]}" \
  --entrypoint vllm \
  "$VLLM_IMAGE" "${args[@]}" >/dev/null

{ echo "host=$(hostname) rank=$node_rank image=$VLLM_IMAGE image_id=$(docker image inspect --format '{{.Id}}' "$VLLM_IMAGE")"
  echo "env: ${docker_env[*]}"; echo "args: ${args[*]}"; } > "$script_dir/logs/launch-vllm-rank$node_rank.txt"
echo "Started $CONTAINER_NAME rank $node_rank on $(hostname) (vllm $VLLM_PARALLEL mode=$MODE partition=${VLLM_PP_LAYER_PARTITION:-auto})"
