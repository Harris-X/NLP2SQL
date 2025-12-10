# 1. 安装依赖
cd /root/autodl-tmp/comp/LLaMA-Factory/Alpha-SQL-master
pip install -r requirements.txt

# 2. 另一终端里启动 vLLM + Qwen3-Coder-30B-A3B-Instruct（保持运行）
cd /root/autodl-tmp/comp/LLaMA-Factory
CUDA_VISIBLE_DEVICES=0 \
nohup vllm serve ./Qwen3-Coder-30B-A3B-Instruct \
  --served-model-name Qwen3-Coder-30B-A3B-Instruct \
  --port 9999 \
  -tp 1 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 262144 > vllm.log 2>&1 &

echo "Waiting for vLLM to start..."
while ! grep -q "Uvicorn running on" vllm.log; do
  sleep 5
  echo "Waiting..."
  if grep -q "EngineCore failed to start" vllm.log; then
    echo "vLLM failed to start. Check vllm.log."
    exit 1
  fi
done
echo "vLLM started successfully."

# 3. 当前终端中跑预处理 + MCTS + 流水线（推荐先手工跑预处理 & MCTS）
cd /root/autodl-tmp/comp/LLaMA-Factory/Alpha-SQL-master
python -m alphasql.runner.preprocessor \
  --data_file_path "mini_dev_mysql_alpaca_filtered.json" \
  --database_root_dir "mysql_db" \
  --save_root_dir "data/preprocessed/mysql_test" \
  --lsh_threshold 0.5 \
  --lsh_signature_size 128 \
  --lsh_n_gram 3 \
  --lsh_top_k 20 \
  --edit_similarity_threshold 0.5 \
  --embedding_similarity_threshold 0.5 \
  --n_parallel_processes 1 \
  --max_dataset_samples -1

python -m alphasql.runner.mcts_runner config/mysql_exp.yaml

bash script/pipeline_mysql.sh