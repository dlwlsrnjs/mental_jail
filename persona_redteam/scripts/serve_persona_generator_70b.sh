#!/usr/bin/env bash
set -euo pipefail

# Dedicated loopback-only persona generator. This must not replace the responder
# or evaluator endpoints used by the experiment.
MODEL_ID="huihui-ai/Llama-3.3-70B-Instruct-abliterated"
MODEL_REVISION="fa13334669544bab573e0e5313cad629a9c02e2c"
VLLM_BIN="${VLLM_BIN:-vllm}"
PERSONA_GPU_IDS="${PERSONA_GPU_IDS:-0,1}"
PERSONA_PORT="${PERSONA_PORT:-8020}"

export CUDA_VISIBLE_DEVICES="${PERSONA_GPU_IDS}"

exec "${VLLM_BIN}" serve "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --served-model-name "${MODEL_ID}" \
  --host 127.0.0.1 \
  --port "${PERSONA_PORT}" \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --max-num-seqs 4
