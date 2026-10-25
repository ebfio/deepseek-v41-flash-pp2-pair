#!/usr/bin/env bash
# Start one vLLM rank of the two-node DeepSeek-V4.1-Flash server on this host (pipeline parallel by default).
# Usage: NODE_RANK=0|1 ./serve_vllm_node.sh   (normally invoked by launch_cluster.sh with ENGINE=vllm)
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/config.env"

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

cache_dir="$CACHE_ROOT/vllm-dsv41"
mkdir -p "$cache_dir"/{root-cache,triton,tilelang,nv} "$script_dir/logs"
# Seed the persistent cache from the image's own /root/.cache (prebuilt FlashInfer JIT artifacts) on first use.
if [[ ! -e "$cache_dir/root-cache/.seeded" ]]; then
  docker run --rm --volume "$cache_dir/root-cache:/mnt" --entrypoint bash "$VLLM_IMAGE" -c 'cp -a /root/.cache/. /mnt/ && touch /mnt/.seeded'
fi

docker_env=(
  --env "VLLM_HOST_IP=$node_ip"
  --env "VLLM_ENGINE_READY_TIMEOUT_S=3600"
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
  [[ -s "$script_dir/patches/vllm-pp2-flag/sitecustomize.py" ]] || { echo "missing patches/vllm-pp2-flag/sitecustomize.py" >&2; exit 1; }
  extra_mounts+=(--volume "$script_dir/patches/vllm-pp2-flag:/opt/pp2-flag:ro")
  docker_env+=(--env "PYTHONPATH=/opt/pp2-flag")
  # Mainline-aligned PP2+DSpark increments (2026-10-25): the ClassVar
  # flip (sitecustomize) + draft-own-embed (dspark.py) + keep input_ids on
  # non-first PP ranks for vision MoE routing (model_runner.py,
  # cudagraph_utils.py). All three overlays are v0.31.0 sources + small edits;
  # the 5-file port is discarded (doubly stale for this image).
  [[ -s "$script_dir/patches/vllm-pp2-flag/dspark.py" ]] || { echo "missing patches/vllm-pp2-flag/dspark.py" >&2; exit 1; }
  extra_mounts+=(--volume "$script_dir/patches/vllm-pp2-flag/dspark.py:/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/nvidia/dspark.py:ro")
  [[ -s "$script_dir/patches/vllm-pp2-flag/model_runner.py" ]] || { echo "missing patches/vllm-pp2-flag/model_runner.py" >&2; exit 1; }
  extra_mounts+=(--volume "$script_dir/patches/vllm-pp2-flag/model_runner.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py:ro")
  [[ -s "$script_dir/patches/vllm-pp2-flag/cudagraph_utils.py" ]] || { echo "missing patches/vllm-pp2-flag/cudagraph_utils.py" >&2; exit 1; }
  extra_mounts+=(--volume "$script_dir/patches/vllm-pp2-flag/cudagraph_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/cudagraph_utils.py:ro")
  # KV-group filtering under PP (the fork's base patch, re-based to v0.31.0;
  # applies clean, offset 63): without it, allocate_kv_cache raises
  # StopIteration on a PP rank that owns none of a group's layers
  # (receipt 2026-10-25 attempt 4, worker/utils.py:432).
  [[ -s "$script_dir/patches/vllm-pp2-flag/kv_cache_utils.py" ]] || { echo "missing patches/vllm-pp2-flag/kv_cache_utils.py" >&2; exit 1; }
  extra_mounts+=(--volume "$script_dir/patches/vllm-pp2-flag/kv_cache_utils.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py:ro")
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
[[ -n "${VLLM_COMPILATION_CONFIG:-}" ]] && args+=(--compilation-config "$VLLM_COMPILATION_CONFIG")
[[ "$VLLM_ENFORCE_EAGER" == 1 ]] && args+=(--enforce-eager)
[[ -n "${VLLM_OVERRIDE_GENERATION_CONFIG:-}" ]] && args+=(--override-generation-config "$VLLM_OVERRIDE_GENERATION_CONFIG")
[[ -n "${VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS:-}" ]] && args+=(--default-chat-template-kwargs "$VLLM_DEFAULT_CHAT_TEMPLATE_KWARGS")
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
    args+=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${VLLM_DSPEC_TOKENS:-5},\"draft_sample_method\":\"${VLLM_DRAFT_SAMPLE_METHOD:-probabilistic}\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":$adaptive}") ;;
  *) echo "MODE must be throughput or low-latency" >&2; exit 2 ;;
esac
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && args+=($EXTRA_ARGS)
if [[ "$node_rank" == 0 ]]; then args+=(--host "$API_HOST" --port "$API_PORT"); else args+=(--headless); fi

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
