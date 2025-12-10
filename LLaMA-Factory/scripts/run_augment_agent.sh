#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
INPUT_FILE="${ROOT_DIR}/Alpha-SQL-master/mini_dev_mysql_alpaca_filtered.json"
OUTPUT_FILE="${ROOT_DIR}/data/augmented_decomposition_agent.json"
CORR_LOG="${ROOT_DIR}/data/corrections_log.jsonl"

python "${ROOT_DIR}/Alpha-SQL-master/augment_decomposition.py" \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --min_steps 3 --max_steps 5 \
  --variants_per_question 2 \
  --max_step_retries 2 \
  --limit 2 \
  --endpoint_type online \
  --llm_model qwen3-coder-480b-a35b-instruct \
  --openai_base_url "https://hk-api.gptbest.vip/v1" \
  --openai_api_key "${OPENAI_API_KEY:-sk-GMYNUCidV96DStXskUpPqgemoaDur0alDXZkeyiq5E3mXGZn}" \
  --corrections_file "${CORR_LOG}"