# Alpha-SQL 运行日志记录指南

本指南介绍如何启用、查看与理解 Alpha-SQL 在 MySQL 场景中的运行日志，包括：
- 模型交互 (LLM 调用) 日志
- MCTS 搜索阶段日志（rollout / backpropagate / 保存）
- SQL 执行阶段日志（查询/插入结果、错误）
- 一键流水线整体日志文件

---
## 1. 日志功能概述
新增的日志组件位于：`alphasql/utils/run_logger.py`。
通过环境变量控制：
- `ALPHASQL_LOG_DIR`：日志输出目录（默认 `logs`）
- `ALPHASQL_LOG_VERBOSE=1`：启用 DEBUG 级别更详细日志（默认 INFO）
- `ALPHASQL_LOG_FULL=1`：输出完整提示与模型回复、完整 SQL 与结果（默认关闭，避免日志暴涨）

结构化 JSON Lines 附加日志文件：
- `logs/llm_dialogues.jsonl`：每次 LLM 调用的完整对话记录（提示、输出、耗时、策略）
- `logs/sql_exec.jsonl`：每条 SQL 的执行记录（语句、成功/失败、行数，选配完整行数据）

每次运行会生成形如：`logs/run_YYYYmmdd_HHMMSS.log` 的文件。
流水线脚本 `script/pipeline_mysql.sh` 会同时将 stdout/stderr 重定向写入 `logs/pipeline_时间戳.log`。

---
## 2. 启用日志
方式一：直接使用流水线脚本自动开启：
```bash
export ALPHASQL_LOG_VERBOSE=1              # 可选：详细模式
bash script/pipeline_mysql.sh
```
方式二：手动运行各阶段：
```bash
export ALPHASQL_LOG_DIR=logs
export ALPHASQL_LOG_VERBOSE=1
python -m alphasql.runner.mcts_runner config/mysql_exp.yaml
python code/sql_exe.py --mode query --input data/mysql_mcts_pred_exec.json --output result/mcts_sql_execution.json
```

---
## 3. 日志内容结构
### 3.1 LLM 调用 (`openai_llm.py`)
示例条目：
```
2025-11-25 10:12:01 | INFO | LLM HTTP call attempt=1 model=claude-opus-4-5-20251101 len(prompt)=742 prompt_head=SELECT ...
2025-11-25 10:12:02 | INFO | LLM HTTP success latency=1.23s outputs=1 first_len=310
```
字段说明：
- `attempt`：第几次重试
- `latency`：单次请求耗时
- `outputs`：返回 generations 数量
- `first_len`：首条输出长度（字符）
- 错误示例：`LLM HTTP error status=429 ...` / `LLM HTTP exception ...`

此外会在 `logs/llm_dialogues.jsonl` 追加结构化对话记录（当 `ALPHASQL_LOG_FULL=1` 时包含完整提示与输出；关闭时保留截断片段）：
```json
{"ts":"2025-11-25T10:12:02.123","mode":"http","model":"claude-opus-4-5-20251101","strategy":"single","n":1,"latency":1.23,"prompt":"...","outputs":["..."]}
```

### 3.2 MCTS 搜索 (`mcts.py`)
示例：
```
Solve start task_id=3 max_rollout=5
Rollout task=3 step=1/5
Selected leaf type=ROOT depth=0 children=0 N=0
Expanded leaf now children=4
Simulation reached terminal depth=6
Backpropagate start final_sql='SELECT ...'
Backpropagate done reward=0.67
Task=3 done valid_reasoning_paths=2 saving=results/mysql_exp/3.pkl
```
关键事件：
- `Rollout`：每次搜索迭代开始
- `Selected leaf`：UCB 选择结果
- `Expanded leaf`：扩展后的子节点数
- `Simulation reached terminal`：随机模拟到终止节点
- `Backpropagate`：奖励传播（显示最终 SQL 与 reward）

### 3.3 SQL 执行 (`sql_exe.py`)
示例：
```
QUERY_START input=data/mysql_mcts_pred_exec.json output=result/mcts_sql_execution.json
EXECUTE sql_id=3_1 len=245
EXECUTE_SUCCESS sql_id=3_1 rows=12
EXECUTE_ERROR sql_id=3_2 err=(1054, "Unknown column 'xxx'")
QUERY_WRITE_DONE file=result/mcts_sql_execution.json count=2
```
说明：
- `EXECUTE_SUCCESS`：记录返回行数
- `EXECUTE_ERROR`：错误包含数据库异常信息
- 写入完成包含结果总条数

同时会在 `logs/sql_exec.jsonl` 追加结构化记录：
```json
{"ts":"2025-11-25T10:12:05.456","op":"query","sql_id":"3_1","status":"success","statement":"SELECT ...","row_count":12,"rows":[{"col":1}]}
```
当 `ALPHASQL_LOG_FULL=1` 时包含完整 `statement` 与 `rows`；关闭时仅保留摘要。

### 3.4 流水线脚本总体 (`pipeline_mysql.sh`)
示例：
```
[PIPELINE] 日志文件: logs/pipeline_20251125_101200.log
[PIPELINE] 检测到已有 tasks.pkl，跳过预处理。
[PIPELINE] 运行 MCTS 搜索 (配置: config/mysql_exp.yaml)
[PIPELINE] MCTS 搜索结束
[PIPELINE] 提取最终 SQL -> data/mysql_mcts_pred_exec.json
[PIPELINE] 执行收集的 SQL -> ../result/mcts_sql_execution.json
[PIPELINE] 流水线完成。概览：
...
```

---
## 4. 日志查看与分析
查看最新日志：
```bash
ls -lt logs | head
tail -n 50 logs/run_20251125_101201.log
grep 'EXECUTE_ERROR' logs/run_*.log
```
统计成功/失败数量快速脚本示例：
```bash
grep -c 'EXECUTE_SUCCESS' logs/run_*.log
grep -c 'EXECUTE_ERROR' logs/run_*.log
```

导出最终 SQL 与奖励，可通过读取 `results/mysql_exp/<id>.pkl`，并结合日志中的 `Backpropagate` 行定位高 reward path。

---
## 5. 常见问题
| 问题 | 日志线索 | 处理建议 |
|------|----------|----------|
| LLM 长时间无响应 | 没有 success/exception 行 | 检查网络、HOST、KEY；确认未被防火墙阻断 |
| 终止节点 SQL 为空 | Backpropagate final_sql='' | 提高 `temperature` 或检查 prompt 模板是否正确 |
| SQL 全部失败 | EXECUTE_ERROR 高频出现 | 校验表/字段存在；单条复制到客户端测试 |
| 日志文件过多 | logs 目录膨胀 | 定期清理：`find logs -type f -mtime +3 -delete` |

---
## 6. 最佳实践
- 调试阶段开启：`export ALPHASQL_LOG_VERBOSE=1`，完成后关闭降低噪声。
- 需要完整对话与结果时开启：`export ALPHASQL_LOG_FULL=1`；常规运行建议关闭避免日志过大。
- 使用单进程 (`n_processes: 1`) + `ALPHASQL_TASK_LIMIT=3` 先验证流程，再扩展到全量。

---
## 7. 快速完整示例
```bash
export GPTBEST_API_HOST=hk-api.gptbest.vip
export GPTBEST_API_KEY=sk-xxxx
export ALPHASQL_LOG_VERBOSE=1
export ALPHASQL_TASK_LIMIT=2
bash script/pipeline_mysql.sh
# 查看日志
tail -n 100 logs/pipeline_*.log
grep 'EXECUTE_ERROR' logs/run_*.log
```

---
## 8. 后续可扩展
- 将 reward 与 SQL 一起输出到独立 CSV 便于排序
- 引入结构化日志（JSON Lines）利于后期分析
- 集成可观测平台（如 ELK / Loki / Grafana）追踪历史运行

如需我补充自动统计脚本或 JSON Lines 日志格式，请继续提出。祝你调试顺利！
