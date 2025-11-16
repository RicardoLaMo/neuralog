#!/bin/bash
#
# Start vLLM server with H200 GPU for NeuraLog
#
# Usage:
#   ./start_vllm_server.sh [model_name] [num_gpus]
#
# Examples:
#   ./start_vllm_server.sh Qwen/QwQ-32B-Preview 1
#   ./start_vllm_server.sh deepseek-ai/DeepSeek-R1 4
#

set -e

# Default values
MODEL="${1:-Qwen/QwQ-32B-Preview}"
NUM_GPUS="${2:-1}"
PORT="${3:-8000}"
GPU_MEMORY_UTIL="${4:-0.90}"
MAX_MODEL_LEN="${5:-32768}"

echo "================================================="
echo "Starting vLLM Server for NeuraLog"
echo "================================================="
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS"
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "================================================="

# Set CUDA devices based on number of GPUs
if [ "$NUM_GPUS" -eq "1" ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$NUM_GPUS" -eq "2" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
elif [ "$NUM_GPUS" -eq "4" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$NUM_GPUS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --use-v2-block-manager \
    --max-num-seqs 256 \
    --disable-log-requests

echo ""
echo "vLLM server started at http://localhost:$PORT"
echo "OpenAI-compatible API available at http://localhost:$PORT/v1"
echo ""
echo "Test with:"
echo "curl http://localhost:$PORT/v1/models"
