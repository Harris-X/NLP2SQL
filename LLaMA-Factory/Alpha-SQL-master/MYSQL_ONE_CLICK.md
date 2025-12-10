# Alpha-SQL MySQL 一键复现与 API 模型测试指南

本文件提供从数据准备、模型调用改造到一键流水线执行的完整过程说明，结合新增脚本与代码修改，帮助快速复现 `results/mysql_exp` 的 MCTS 搜索 + SQL 执行测试。

---
## 目录
1. 目标概述
2. 新增/修改文件速览
3. 环境准备
4. 模型调用改造 (API 方式)
5. 数据预处理 (自定义 MySQL 数据集)
6. MCTS 搜索运行
7. 提取最终 SQL
8. 执行 SQL 并生成结果
9. 一键流水线脚本使用
10. 常见问题 & 排查
11. 全量命令速查

---
## 1. 目标概述
- 使用数据集：`mini_dev_mysql_alpaca_filtered.json`（含 input/output）
- 使用配置：`config/mysql_exp.yaml`（模型已改为外部 API 对接模型）
- 通过 API 方式调用 LLM（参考 `augment_dataset.py` 的 HTTP 调用模式）
- 运行 MCTS，保存 reasoning 路径到 `results/mysql_exp`；提取最终 SQL；批量执行并得到结果 JSON。

---
## 2. 新增/修改文件速览
| 文件 | 类型 | 说明 |
|------|------|------|
| `alphasql/llm_call/openai_llm.py` | 修改 | 增加 GPTBEST HTTP 分支，环境变量触发，保留 OpenAI SDK 回退 |
| `script/extract_mcts_sqls.py` | 新增 | 从 MCTS 结果 .pkl 中提取终止节点 SQL，支持去重与限制条数 |
| `script/pipeline_mysql.sh` | 新增 | 串联预处理 → MCTS → SQL 提取 → 执行，全自动流程 |
| `code/sql_exe.py` | 已改 | 支持命令行参数，执行批量查询，输出结果 JSON |
| `MYSQL_ONE_CLICK.md` | 新增 | 本操作说明文档 |

---
## 3. 环境准备
```bash
conda create -n alphasql python=3.11 -y
conda activate alphasql
pip install -r Alpha-SQL-master/requirements.txt
```
若不使用本地 vLLM，可跳过 README 中 vLLM 部分。

---
## 4. 模型调用改造 (API 方式)
### 4.1 环境变量设置
```bash
export GPTBEST_API_HOST="hk-api.gptbest.vip"      # 或你的自定义域名
export GPTBEST_API_KEY="sk-xxxxxxxx"              # 替换为真实 key
# 指定 config/mysql_exp.yaml 中的模型名保持一致
export AUGMENT_MODEL="claude-opus-4-5-20251101"   # 可与 mcts_model_kwargs.model 对齐
```
- 若存在 `GPTBEST_API_HOST` + `GPTBEST_API_KEY`，`call_openai` 将自动使用自定义 HTTPS 请求。
- 未设置则退回 OpenAI SDK（可通过 `OPENAI_BASE_URL` / `OPENAI_API_KEY` 覆盖）。

### 4.2 核心改动说明 (`openai_llm.py`)
新增逻辑：
1. 检测环境变量 `GPTBEST_API_HOST` & `GPTBEST_API_KEY`。
2. 构造与 `augment_dataset.py` 一致的 `POST /v1/chat/completions` 请求体：
   ```json
   {
     "model": "claude-opus-4-5-20251101",
     "messages": [{"role": "user", "content": "..."}],
     "temperature": 0.2,
     "top_p": 0.8,
     "stream": false,
     "max_tokens": 4096
   }
   ```
3. 支持重试与多次采样 (`n`, `n_strategy`)。
4. 返回 `List[str]` 保持与原 MCTS 调用兼容。

---
## 5. 数据预处理 (自定义 MySQL 数据集)
脚本：`script/preprocess_mysql.sh` 已配置：
```bash
python -m alphasql.runner.preprocessor \
  --data_file_path "mini_dev_mysql_alpaca_filtered.json" \
  --database_root_dir "mysql_db" \
  --save_root_dir "data/preprocessed/mysql_test" \
  ...
```
说明：
- 数据集中每条记录包含 `input` 与 `output`。预处理阶段会解析出 question / evidence，并设置统一的 `db_id` 为 `mysql_db`。
- 需要在 `mysql_db` 目录下提供相应 schema（如无真实库，可提供简化 schema.json）。

手动运行：
```bash
cd Alpha-SQL-master
bash script/preprocess_mysql.sh
```
生成文件：`data/preprocessed/mysql_test/dev/tasks.pkl` 等。

---
## 6. MCTS 搜索运行
脚本：`script/mysql_exp.sh`
```bash
bash script/mysql_exp.sh
# 或:
python -m alphasql.runner.mcts_runner config/mysql_exp.yaml
```
输出：
- `results/mysql_exp/<question_id>.pkl`：每任务的 reasoning 路径 (List[List[MCTSNode]])
- `results/mysql_exp/config.json`：运行时配置快照

---
## 7. 提取最终 SQL
新增脚本：`script/extract_mcts_sqls.py`
示例：
```bash
python script/extract_mcts_sqls.py \
  --mcts_dir results/mysql_exp \
  --output data/mysql_mcts_pred_exec.json \
  --deduplicate --limit-per-task 0
```
输出 JSON 格式：
```json
[
  {"sql_id": "12_1", "sql": "SELECT ..."},
  {"sql_id": "12_2", "sql": "SELECT ..."}
]
```

---
## 8. 执行 SQL 并生成结果
执行器：`code/sql_exe.py`（已扩展 argparse）
```bash
python code/sql_exe.py --mode query \
  --input Alpha-SQL-master/data/mysql_mcts_pred_exec.json \
  --output result/mcts_sql_execution.json \
  --host 127.0.0.1 --port 9030 --user root --password "" --db final_algorithm_competition
```
输出示例：
```json
[
  {
    "sql_id": "12_1",
    "sql": "SELECT ...",
    "status": "success",
    "result": [ {"col": 1} ]
  },
  {
    "sql_id": "12_2",
    "sql": "SELECT ...",
    "status": "error",
    "error_message": "Table not found"
  }
]
```
数值标准化：整型不带小数；浮点保留两位；日期/Decimal 转字符串。

---
## 9. 一键流水线脚本使用
脚本：`script/pipeline_mysql.sh`
作用：自动执行预处理 → MCTS → SQL 提取 → 查询执行。

运行：
```bash
cd Alpha-SQL-master
bash script/pipeline_mysql.sh
```
脚本内部关键变量（可在脚本顶部调整）：
```bash
DATASET="mini_dev_mysql_alpaca_filtered.json"
DB_ROOT_DIR="mysql_db"
MCTS_CONFIG="config/mysql_exp.yaml"
MCTS_RESULT_DIR="results/mysql_exp"
EXTRACT_OUTPUT="data/mysql_mcts_pred_exec.json"
FINAL_EXEC_OUTPUT="../result/mcts_sql_execution.json"
DEDUPLICATE=1
LIMIT_PER_TASK=0
```
数据库连接通过环境变量覆盖：`SR_HOST SR_PORT SR_USER SR_PASSWORD SR_DB`。

---
## 10. 常见问题 & 排查
| 问题 | 可能原因 | 处理建议 |
|------|----------|----------|
| 预处理失败 | 缺少 schema | 在 `mysql_db` 中补充基本表结构或降级相关步骤 |
| MCTS 无输出 SQL | 模型返回空内容 | 提高 `temperature` 或检查 API 额度；打印中间节点内容 |
| 批量执行大量失败 | 表/字段不存在 | 先执行建表脚本 `sql_file/create_table.sql` 并导入数据 |
| API 超时/错误码 | 网络或额度限制 | 查看脚本重试日志；减少并发或缩短 `n` |
| 结果文件为空 | 提取阶段未找到 `final_sql_query` | 确认终止节点类型；调试 `mcts_node.py` 中生成逻辑 |

---
## 11. 全量命令速查
```bash
# 环境
conda activate alphasql

# API 环境变量
export GPTBEST_API_HOST=hk-api.gptbest.vip
export GPTBEST_API_KEY=sk-xxxx

# 预处理
bash script/preprocess_mysql.sh

# MCTS 搜索
bash script/mysql_exp.sh

# 提取 SQL
python script/extract_mcts_sqls.py \
  --mcts_dir results/mysql_exp \
  --output data/mysql_mcts_pred_exec.json \
  --deduplicate

# 执行 SQL
python code/sql_exe.py --mode query \
  --input Alpha-SQL-master/data/mysql_mcts_pred_exec.json \
  --output result/mcts_sql_execution.json \
  --host 127.0.0.1 --port 9030 --user root --password "" --db final_algorithm_competition

# 一键流水线
bash script/pipeline_mysql.sh
```

---
## 结语
通过本指南与新增脚本，可快速完成 Alpha-SQL 在自定义 MySQL 数据集上的零样本 MCTS 搜索与真实数据库验证。如需扩展：
- 增加 SQL 质量评估（执行成功率、覆盖率统计）；
- 增加对生成 SQL 的自动去重/评分策略；
- 加入 WandB 或自定义日志采集。

如需我再加入统计汇总脚本或自动建表逻辑，请继续提出。祝测试顺利！
