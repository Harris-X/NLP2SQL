# Alpha-SQL + Qwen3-Coder-30B-A3B-Instruct 全流程运行说明（MySQL 版本）

本文档记录了在当前云环境下，使用本地部署的 **Qwen3-Coder-30B-A3B-Instruct** 替换原来的在线 API，运行 `Alpha-SQL-master/script/pipeline_mysql.sh` 的完整步骤。Embedding 配置保持不变，数据库连接为：

```sql
mysql -h 127.0.0.1 -P 9030 -uroot
USE final_algorithm_competition;
```

## 1. 前置假设

- 当前工作目录：`/root/autodl-tmp/comp/LLaMA-Factory`
- Alpha-SQL 代码路径：`Alpha-SQL-master/`
- Qwen3-Coder 模型路径：`Qwen3-Coder-30B-A3B-Instruct/`
- 已在本机安装 CUDA 与驱动，具备足够显存加载 30B 模型。

## 2. 安装 Alpha-SQL 运行环境

在当前（已有的）Python 环境中安装 Alpha-SQL 依赖：

```bash
cd /root/autodl-tmp/comp/LLaMA-Factory/Alpha-SQL-master
pip install -r requirements.txt
```

如果你打算单独建立 Conda 环境，可参考 README 中的建议（此处略）。

## 3. 使用 vLLM 部署本地 Qwen3-Coder-30B-A3B-Instruct

1. 安装 vLLM（若尚未安装）：

   ```bash
   pip install vllm
   ```

2. 启动 vLLM 服务（建议在单独一个终端 / 后台进程中运行）：

   ```bash
   cd /root/autodl-tmp/comp/LLaMA-Factory

   CUDA_VISIBLE_DEVICES=0,1,2,3 \
   vllm serve ./Qwen3-Coder-30B-A3B-Instruct \
     --served-model-name Qwen3-Coder-30B-A3B-Instruct \
     --port 9999 \
     -tp 4
   ```

   说明：
   - `./Qwen3-Coder-30B-A3B-Instruct` 为模型本地路径（已包含 config、tokenizer、safetensors 等文件）。
   - `--served-model-name` 必须与 Alpha-SQL 中使用的 model 名一致（我们已在配置中设置为 `Qwen3-Coder-30B-A3B-Instruct`）。
   - `--port 9999` 对应下文 .env 的 `OPENAI_BASE_URL=http://0.0.0.0:9999/v1`。

## 4. 配置 Alpha-SQL 使用本地 Qwen3-Coder

### 4.1 修改 `.env`

`Alpha-SQL-master/.env` 已经被更新为：

```dotenv
OPENAI_API_KEY=EMPTY
OPENAI_BASE_URL=http://0.0.0.0:9999/v1

# Embedding 模型保持不变
EMBEDDING_MODEL=text-embedding-3-large

# 数据库配置
SR_HOST=127.0.0.1
SR_USER=root
SR_DB=final_algorithm_competition
SR_PORT=9030
```

说明：
- Alpha-SQL 使用 `openai` 兼容协议访问 LLM，`OPENAI_BASE_URL` 指向 vLLM。
- `OPENAI_API_KEY` 对于本地 vLLM 通常不校验，因此设为 `EMPTY` 即可。
- Embedding 仍使用原先配置（可继续走远程服务或本地服务，根据需求在 `.env` 中单独设置 `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL`）。

### 4.2 修改 MCTS 配置使用 Qwen3-Coder

`Alpha-SQL-master/config/mysql_exp.yaml` 已被更新为：

```yaml
mcts_model_kwargs:
    model: "Qwen3-Coder-30B-A3B-Instruct"
    n: 1
    top_p: 0.8
    max_tokens: 65536
    temperature: 0.8
    n_strategy: "single"
```

这会让 `alphasql.runner.mcts_runner` 通过 OpenAI 接口调用本地的 Qwen3-Coder-30B-A3B-Instruct。

## 5. 确认数据集与数据库

1. 数据集：

   - `Alpha-SQL-master/mini_dev_mysql_alpaca_filtered.json` 已存在，对应 `script/pipeline_mysql.sh` 中的 `DATASET` 配置：

     ```bash
     DATASET="mini_dev_mysql_alpaca_filtered.json"
     ```

2. 数据库：

   - 已在云服务上暴露本地 MySQL（StarRocks/Fusion 兼容）端口：`127.0.0.1:9030`，数据库为 `final_algorithm_competition`。
   - 你可以用以下命令确认连接：

     ```bash
     mysql -h 127.0.0.1 -P 9030 -uroot
     USE final_algorithm_competition;
     ```

## 6. 流水线脚本配置调整

`Alpha-SQL-master/script/pipeline_mysql.sh` 关键修改如下：

```bash
EXTRACT_SCRIPT="alphasql/runner/sql_selection.py"           # 提取脚本路径（使用仓库内默认实现）

execute_sqls(){
  log "执行收集的 SQL -> ${FINAL_EXEC_OUTPUT}"
  python sql_exe.py --mode query \
    --input "${EXTRACT_OUTPUT}" \
    --output "${FINAL_EXEC_OUTPUT}" \
    --host "${HOST}" --port "${PORT}" --user "${USER}" --password "${PASSWORD}" --db "${DBNAME}" || true
  log "执行完成，结果位于 ${FINAL_EXEC_OUTPUT}" 
}
```

说明：
- 使用仓库内的 `alphasql/runner/sql_selection.py` 作为 SQL 提取脚本（原先的 `extract_mcts_sqls.py` 在该仓库中不存在）。
- SQL 执行改为调用当前目录下的 `sql_exe.py`，使用你给出的数据库连接配置。

## 7. 运行 Alpha-SQL 全流程

> 建议在 `Alpha-SQL-master` 目录下运行：

```bash
cd /root/autodl-tmp/comp/LLaMA-Factory/Alpha-SQL-master

# 步骤 1：预处理（如 tasks.pkl 已存在可跳过）
# bash script/pipeline_mysql.sh 中目前注释了 run_preprocess 和 run_mcts，
# 如需完整 MCTS 流程，可先手动运行：

# 1.1 数据预处理
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

# 1.2 运行 MCTS 搜索
python -m alphasql.runner.mcts_runner config/mysql_exp.yaml

# 步骤 2：一键提取 SQL + 执行
bash script/pipeline_mysql.sh
```

当前 `pipeline_mysql.sh` 中的主流程默认只执行：

```bash
extract_sqls
execute_sqls
summary
```

如果你希望让脚本本身也自动执行预处理和 MCTS，可以在 `pipeline_mysql.sh` 末尾解除注释：

```bash
# run_preprocess
# run_mcts
extract_sqls
execute_sqls
summary
```

改为：

```bash
run_preprocess
run_mcts
extract_sqls
execute_sqls
summary
```

## 8. 日志与结果位置

- 主日志：`Alpha-SQL-master/logs/pipeline_*.log`
- MCTS 结果目录：`Alpha-SQL-master/results/mysql_exp/`
- 提取的 SQL：`Alpha-SQL-master/data/mysql_mcts_pred_exec.json`
- SQL 执行结果：`Alpha-SQL-master/../result/mcts_sql_execution.json`（即仓库根目录下的 `result/` 文件夹）

## 9. 快速检查清单

在真正大规模运行前，可以检查以下几点：

1. vLLM 是否正常启动，端口 `9999` 是否监听：
   ```bash
   curl http://0.0.0.0:9999/v1/models
   ```

2. `.env` 中 `OPENAI_BASE_URL` 是否指向 `http://0.0.0.0:9999/v1`，`OPENAI_API_KEY` 是否为 `EMPTY`。
3. `config/mysql_exp.yaml` 中的 `model` 是否为 `Qwen3-Coder-30B-A3B-Instruct`。
4. `mysql -h 127.0.0.1 -P 9030 -uroot` 能否成功连接，并能 `USE final_algorithm_competition;`。
5. `python -m alphasql.runner.preprocessor ...` 是否能成功生成 `data/preprocessed/mysql_test/dev/tasks.pkl`。
6. `python -m alphasql.runner.mcts_runner config/mysql_exp.yaml` 是否能在 `results/mysql_exp/` 下产生 MCTS 结果文件。
7. `bash script/pipeline_mysql.sh` 是否能产出最终的 `mcts_sql_execution.json`。

确认以上步骤正常后，即完成了基于 **Qwen3-Coder-30B-A3B-Instruct + 本地 MySQL** 的 Alpha-SQL 端到端推理与 SQL 执行流程。
