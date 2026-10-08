#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

set -a
# shellcheck disable=SC1091
source configs/llama33_70b_abliterated.env.example
set +a

python experiments/pcsa_attack_optimize.py \
  --per-label "${PER_LABEL:-5}" \
  --max-turns "${MAX_TURNS:-6}" \
  --workers "${WORKERS:-2}" \
  --surrogate-base-urls "${SURROGATE_BASE_URL:-http://127.0.0.1:8017}" \
  --out-dir "${OUT_DIR:-outputs/gen-llama33-70b-abliterated__sur-llama31-8b__eval-4omini__pilot30}"
