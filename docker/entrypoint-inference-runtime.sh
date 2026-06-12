#!/bin/sh
# Inference entrypoint for the validated ARM TRT-LLM/DataSystem runtime image.

set -eu

# Drop stale preload paths inherited from the base image or an interactive
# container. The runtime below rebuilds the exact preload chain from files that
# exist inside this image.
unset LD_PRELOAD || true

MODEL_PATH="${MODEL_PATH:-/app/checkpoints/decoder_qwen3/decoder_epoch_20.pt}"
QWEN3_MODEL_PATH="${QWEN3_MODEL_PATH:-/app/models/Qwen3-0.6B}"
TRT_ENGINE_DIR="${TRT_ENGINE_DIR:-/app/trt_engines/qwen3_rec_v4}"
PORT="${PORT:-18000}"
DEVICE="${DEVICE:-cuda}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
BACKBONE="${BACKBONE:-qwen3}"
USE_TRT_LLM="${USE_TRT_LLM:-true}"
DATASYSTEM_HOST="${DATASYSTEM_HOST:-}"
DATASYSTEM_PORT="${DATASYSTEM_PORT:-31501}"
PYTHON_DATASYSTEM_ENABLED="${PYTHON_DATASYSTEM_ENABLED:-1}"
ENABLE_DATASYSTEM_PRELOADS="${ENABLE_DATASYSTEM_PRELOADS:-1}"
TRT_MAX_KV_TOKENS="${TRT_MAX_KV_TOKENS:-1024}"
TRT_KV_CACHE_HOST_CACHE_SIZE="${TRT_KV_CACHE_HOST_CACHE_SIZE:-0}"
TRT_SCHEDULER_POLICY="${TRT_SCHEDULER_POLICY:-max_utilization}"
TRT_MAX_INPUT_LEN="${TRT_MAX_INPUT_LEN:-64}"
TRT_NUM_SAMPLES="${TRT_NUM_SAMPLES:-1}"
TRT_RESULT_CACHE_ENABLED="${TRT_RESULT_CACHE_ENABLED:-1}"

export PYTHONPATH="${PYTHONPATH:-/app:/home/TensorRT-LLM}"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}"
export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-14/root/usr/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib:/usr/lib64:/usr/local/lib:${LD_LIBRARY_PATH:-}"
export DATASYSTEM_HOST DATASYSTEM_PORT
export TRT_KV_CACHE_HOST_CACHE_SIZE

find_first_file() {
    find "$1" -name "$2" 2>/dev/null | head -1 || true
}

PRELOADS=""
append_preload() {
    candidate="$1"
    if [ -n "$candidate" ] && [ -s "$candidate" ]; then
        if [ -n "$PRELOADS" ]; then
            PRELOADS="$PRELOADS $candidate"
        else
            PRELOADS="$candidate"
        fi
    fi
}

NVML=""
if [ -s /usr/lib64/libnvidia-ml.so.570.124.06 ]; then
    NVML="/usr/lib64/libnvidia-ml.so.570.124.06"
elif [ -e /usr/lib64/libnvidia-ml.so.1 ]; then
    NVML="$(readlink -f /usr/lib64/libnvidia-ml.so.1 || true)"
    if [ ! -s "$NVML" ]; then
        NVML=""
    fi
fi

ABSEIL="$(find_first_file /usr/local libabseil_dll.so.2407.0.0)"

if [ "$ENABLE_DATASYSTEM_PRELOADS" != "0" ]; then
    append_preload /opt/pairec/lib/block_ds_consumer.so
    append_preload /opt/pairec/lib/stub_gpu.so
fi
append_preload "$NVML"
append_preload "$ABSEIL"

if [ -n "$PRELOADS" ]; then
    export LD_PRELOAD="$PRELOADS"
else
    unset LD_PRELOAD
fi

echo "========================================="
echo "PaiRec4TigerLLM Inference Runtime"
echo "========================================="
echo "Model path:      $MODEL_PATH"
echo "Qwen3 path:      $QWEN3_MODEL_PATH"
echo "TRT engine dir:  $TRT_ENGINE_DIR"
echo "Port:            $PORT"
echo "Device:          $DEVICE"
echo "Use TRT-LLM:     $USE_TRT_LLM"
echo "C++ DataSystem:  ${DATASYSTEM_HOST:-<unset>}:${DATASYSTEM_PORT}"
echo "Python DS:       $PYTHON_DATASYSTEM_ENABLED"
echo "DS preloads:     $ENABLE_DATASYSTEM_PRELOADS"
echo "TRT samples:     $TRT_NUM_SAMPLES"
echo "TRT host cache:  $TRT_KV_CACHE_HOST_CACHE_SIZE"
echo "Result cache:    $TRT_RESULT_CACHE_ENABLED"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "LD_PRELOAD:      ${LD_PRELOAD:-<unset>}"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model checkpoint not found: $MODEL_PATH" >&2
    exit 1
fi

if [ ! -d "$QWEN3_MODEL_PATH" ]; then
    echo "ERROR: Qwen3 model dir not found: $QWEN3_MODEL_PATH" >&2
    exit 1
fi

if [ "$USE_TRT_LLM" = "true" ] && [ ! -d "$TRT_ENGINE_DIR" ]; then
    echo "ERROR: TRT engine dir not found: $TRT_ENGINE_DIR" >&2
    exit 1
fi

SEMANTIC_MAP="/app/data/tenrec/processed/semantic_id_map.json"
if [ ! -f "$SEMANTIC_MAP" ]; then
    echo "WARNING: semantic map not found: $SEMANTIC_MAP" >&2
fi

ARGS="--model_path $MODEL_PATH --port $PORT --device $DEVICE"
ARGS="$ARGS --max_batch_size $MAX_BATCH_SIZE --max_seq_len $MAX_SEQ_LEN"

if [ "$BACKBONE" = "qwen3" ]; then
    ARGS="$ARGS --qwen3_model_path $QWEN3_MODEL_PATH"
fi

if [ "$USE_TRT_LLM" = "true" ]; then
    ARGS="$ARGS --use_trt_llm --trt_engine_dir $TRT_ENGINE_DIR"
fi

if [ "$PYTHON_DATASYSTEM_ENABLED" != "0" ] && [ -n "$DATASYSTEM_HOST" ]; then
    ARGS="$ARGS --datasystem_host $DATASYSTEM_HOST --datasystem_port $DATASYSTEM_PORT"
fi

ARGS="$ARGS --trt_max_kv_tokens $TRT_MAX_KV_TOKENS"
ARGS="$ARGS --trt_kv_cache_host_cache_size $TRT_KV_CACHE_HOST_CACHE_SIZE"
ARGS="$ARGS --trt_scheduler_policy $TRT_SCHEDULER_POLICY"
ARGS="$ARGS --trt_max_input_len $TRT_MAX_INPUT_LEN"
ARGS="$ARGS --trt_num_samples $TRT_NUM_SAMPLES"
ARGS="$ARGS --trt_result_cache_enabled $TRT_RESULT_CACHE_ENABLED"

echo "Starting inference server..."
exec python -m inference.trt_llm.server $ARGS
