#离线（不依赖LLM，仅验证执行链路）
python Alpha-SQL-master/augment_decomposition.py \
  --input_file data/augment_one.json \
  --output_file data/augmented_decomposition_one.json \
  --min_steps 2 --max_steps 2 \
  --variants_per_question 1 \
  --max_step_retries 0 \
  --limit 1 \
  --offline

# 本地 vLLM（OpenAI 兼容服务）
python Alpha-SQL-master/augment_decomposition.py \
  --input_file Alpha-SQL-master/mini_dev_mysql_alpaca_filtered.json \
  --output_file data/augmented_decomposition_vllm.json \
  --min_steps 3 --max_steps 3 \
  --variants_per_question 1 \
  --max_step_retries 1 \
  --limit 1 \
  --endpoint_type vllm \
  --llm_model Qwen3-Coder-30B-A3B-Instruct \
  --openai_base_url http://127.0.0.1:8000/v1 \
  --openai_api_key sk-ignored-for-vllm \
  --corrections_file data/corrections_log.jsonl





python Alpha-SQL-master/augment_decomposition.py \
  --input_file data_updated.json \
  --output_file data/augmented_decomposition_all.json \
  --min_steps 4 --max_steps 10 \
  --variants_per_question 2 \
  --max_step_retries 3 \
  --endpoint_type online \
  --llm_model qwen3-coder-480b-a35b-instruct \
  --openai_base_url https://hk-api.gptbest.vip/v1 \
  --openai_api_key 'sk-GMYNUCidV96DStXskUpPqgemoaDur0alDXZkeyiq5E3mXGZn' \
  --corrections_file data/corrections_log.jsonl
