#!/usr/bin/env bash
# 一键运行：预处理 -> MCTS -> 提取SQL -> 执行SQL
# 运行位置：建议在 Alpha-SQL-master 目录下执行: bash script/pipeline_mysql.sh
set -euo pipefail

# Ensure Python can import the local `alphasql` package when running scripts directly.
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# 日志目录可通过环境变量覆盖 ALPHASQL_LOG_DIR，若未设置则使用默认 logs
LOG_DIR=${ALPHASQL_LOG_DIR:-logs}
mkdir -p "$LOG_DIR"
RUN_TS=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/pipeline_${RUN_TS}.log"
export ALPHASQL_LOG_DIR="$LOG_DIR"  # 供 Python 日志模块使用
export ALPHASQL_LOG_VERBOSE=${ALPHASQL_LOG_VERBOSE:-0}
echo "[PIPELINE] 日志文件: $MAIN_LOG"
exec > >(tee -a "$MAIN_LOG") 2>&1

# -------- 配置区（可根据需要修改） --------
DATASET="mini_dev_mysql_alpaca_filtered.json"          # 数据集文件
DB_ROOT_DIR="mysql_db"                                 # 数据库 schema 根目录
PREPROCESS_SAVE_ROOT="data/preprocessed/mysql_test"    # 预处理保存根目录
MCTS_CONFIG="config/mysql_exp.yaml"                    # MCTS 配置文件
MCTS_RESULT_DIR="results/mysql_exp"                    # MCTS 输出目录
EXTRACT_SCRIPT="alphasql/runner/sql_selection.py"           # 提取脚本路径（使用仓库内默认实现）
EXTRACT_OUTPUT="data/mysql_mcts_pred_exec.json"        # 收集的可执行 SQL 输出文件
FINAL_EXEC_OUTPUT="../result/mcts_sql_execution.json"  # 查询执行结果文件（仓库根 result/ 目录）
LIMIT_PER_TASK=0                                         # 每任务最多保留 SQL 数 (0 不限)
DEDUPLICATE=1                                            # 是否去重 (1 是, 0 否)

# 数据库连接 (可从环境变量覆盖)
HOST=${SR_HOST:-127.0.0.1}
PORT=${SR_PORT:-9030}
USER=${SR_USER:-root}
PASSWORD=${SR_PASSWORD:-}
DBNAME=${SR_DB:-final_algorithm_competition}

# -------- 函数区 --------
log(){
  echo "[PIPELINE] $*"
}

run_preprocess(){
  if [[ -f "${PREPROCESS_SAVE_ROOT}/dev/tasks.pkl" ]];
  then
    log "检测到已有 tasks.pkl，跳过预处理。"
  else
    log "开始预处理数据集 ${DATASET}"
    python -m alphasql.runner.preprocessor \
      --data_file_path "${DATASET}" \
      --database_root_dir "${DB_ROOT_DIR}" \
      --save_root_dir "${PREPROCESS_SAVE_ROOT}" \
      --lsh_threshold 0.5 \
      --lsh_signature_size 128 \
      --lsh_n_gram 3 \
      --lsh_top_k 20 \
      --edit_similarity_threshold 0.5 \
      --embedding_similarity_threshold 0.5 \
      --n_parallel_processes 1 \
      --max_dataset_samples -1
  fi
}

run_mcts(){
  log "运行 MCTS 搜索 (配置: ${MCTS_CONFIG})"
  python -m alphasql.runner.mcts_runner "${MCTS_CONFIG}"
  log "MCTS 搜索结束"
}

extract_sqls(){
  log "提取最终 SQL -> ${EXTRACT_OUTPUT}"
  local dedup_flag=""
  if [[ "${DEDUPLICATE}" == "1" ]]; then
    dedup_flag="--deduplicate"
  fi
  # Use module invocation to ensure `alphasql` package is correctly resolved.
  python -m alphasql.runner.sql_selection \
    --results_dir "${MCTS_RESULT_DIR}" \
    --db_root_dir "${DB_ROOT_DIR}" \
    --output_path "${EXTRACT_OUTPUT}"
}

execute_sqls(){
  log "执行收集的 SQL -> ${FINAL_EXEC_OUTPUT}"
  python sql_exe.py --mode query \
    --input "${EXTRACT_OUTPUT}" \
    --output "${FINAL_EXEC_OUTPUT}" \
    --host "${HOST}" --port "${PORT}" --user "${USER}" --password "${PASSWORD}" --db "${DBNAME}" || true
  log "执行完成，结果位于 ${FINAL_EXEC_OUTPUT}" 
}

summary(){
  log "流水线完成。概览："
  log "  数据集: ${DATASET}"
  log "  任务文件: ${PREPROCESS_SAVE_ROOT}/dev/tasks.pkl"
  log "  MCTS结果目录: ${MCTS_RESULT_DIR}"
  log "  提取SQL: ${EXTRACT_OUTPUT}"
  log "  执行结果: ${FINAL_EXEC_OUTPUT}"
  log "  主日志: ${MAIN_LOG}"
}

# -------- 主流程 --------
run_preprocess
run_mcts
extract_sqls
execute_sqls
summary
