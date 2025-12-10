# Alpha-SQL 复现与数据集测试指南

本文档总结在本工作区中复现 Alpha-SQL、替换模型以及针对 MySQL 数据集执行与生成结果 JSON 的完整流程。包含：环境准备、模型修改、数据集处理、SQL 批量执行与结果格式化。

## 目录
- [环境与依赖](#环境与依赖)
- [模型配置修改说明](#模型配置修改说明)
- [数据集说明与转换](#数据集说明与转换)
- [批量 SQL 执行脚本增强](#批量-sql-执行脚本增强)
- [执行步骤总览](#执行步骤总览)
- [结果文件格式要求](#结果文件格式要求)
- [完整命令示例](#完整命令示例)
- [常见问题](#常见问题)

## 环境与依赖
参照原始 `README.md`：
1. 创建 AlphaSQL 环境并安装依赖：
    ```bash
    conda create -n alphasql python=3.11 -y
    conda activate alphasql
    pip install -r requirements.txt
    ```
2. 若需本地推理大型模型，可按原文档使用 vLLM；本复现重点在于 API 方式调用与结果生成，不强制要求本地推理。

## 模型配置修改说明
原始配置文件位于 `Alpha-SQL-master/config/mysql_exp.yaml`。我们已将 `mcts_model_kwargs.model` 字段修改为：
```yaml
model: "claude-opus-4-5-20251101"
```
这一步代表：Alpha-SQL 在 MCTS 过程中调用的 LLM 已替换为可通过外部 API 访问的模型。若需再次更换，只需编辑相同字段，并根据服务端需求调整其它参数（如 `temperature`, `n`, `top_p` 等）。

若使用与 `augment_dataset.py` 类似的调用方式，可设置环境变量：
```bash
export GPTBEST_API_HOST=你的HOST
export GPTBEST_API_KEY=你的KEY
export AUGMENT_MODEL=你的模型名
```
然后在相关 LLM 封装（`alphasql/llm_call/openai_llm.py` 或自定义调用层）中切换到目标域名与 Key。

## 数据集说明与转换
- 原始数据集示例：`Alpha-SQL-master/mini_dev_mysql_alpaca_filtered.json`
  - 其对象中 `output` 字段为 SQL；`input` 为 Prompt 组合内容。
- 我们的 SQL 批量执行需要输入形如：
  ```json
  [
    {"sql_id": "sql_1", "sql": "SELECT ..."},
    {"sql_id": "sql_2", "sql": "SELECT ..."}
  ]
  ```

### 新增转换脚本
创建的脚本：`script/build_sql_input.py`，功能：
- 读取原始数据集，抽取每条记录的 `output`（SQL）。
- 生成标准执行输入 JSON。
- 支持 `--limit` 限制条数快速测试；支持 `--drop-semicolon` 去掉尾部分号（某些驱动需要）。

用法示例：
```bash
python script/build_sql_input.py \
  --source mini_dev_mysql_alpaca_filtered.json \
  --target data/mini_dev_mysql_alpaca_filtered_exec.json \
  --limit 20
```

## 批量 SQL 执行脚本增强
我们修改了 `code/sql_exe.py` 增加命令行参数：
- 新增依赖：`argparse`
- 支持参数：`--mode` (`query` | `insert`)，`--input`，`--output`，以及数据库连接信息 `--host --port --user --password --db`。
- 默认模式为 `query`。

修改后可直接执行：
```bash
python code/sql_exe.py --mode query \
  --input data/mini_dev_mysql_alpaca_filtered_exec.json \
  --output result/dataset_exe_result.json \
  --host 127.0.0.1 --port 9030 --user root --password "" --db final_algorithm_competition
```

## 执行步骤总览
1. 准备并启动目标数据库（MySQL / StarRocks）。确保包含所需表结构与数据。若需建表可参考你的 `sql_file/create_table.sql` / 插入数据 `insert_sql.json`。
2. 设置环境变量（可选）：
    ```bash
    export SR_HOST=127.0.0.1
    export SR_PORT=9030
    export SR_USER=root
    export SR_PASSWORD=""
    export SR_DB=final_algorithm_competition
    ```
3. 进入项目目录：
    ```bash
    cd Alpha-SQL-master
    ```
4. 生成用于执行的 SQL 输入：
    ```bash
    python script/build_sql_input.py \
      --source mini_dev_mysql_alpaca_filtered.json \
      --target data/mini_dev_mysql_alpaca_filtered_exec.json
    ```
5. 执行批量查询：
    ```bash
    python ../code/sql_exe.py --mode query \
      --input data/mini_dev_mysql_alpaca_filtered_exec.json \
      --output ../result/dataset_exe_result.json
    ```
    > 注意：路径根据当前工作目录做相对调整。如果在仓库根目录，可直接使用 `code/sql_exe.py` 与 `Alpha-SQL-master/...`。
6. 查看结果文件：`result/dataset_exe_result.json`。

## 结果文件格式要求
参考示例 `upload_example/dataset_exe_result.json`：
```json
[
  {
    "sql_id": "sql_1",
    "sql": "SELECT ...",
    "result": [ {"col": 1}, {"col": 2} ],
    "status": "success"
  },
  {
    "sql_id": "sql_2",
    "sql": "SELECT ...",
    "status": "error",
    "error_message": "..."
  }
]
```
当前实现中：
- 成功会包含 `result` 列表；失败包含 `error_message`。
- 数值类型经过 `normalize_numbers_in_result` 标准化（整型无小数，浮点保留两位）。
- 日期 / 时间 / Decimal 类型通过 `DecimalEncoder` 兼容输出。

## 完整命令示例
假设当前工作目录为仓库根路径：
```bash
# 1. 激活环境
conda activate alphasql

# 2. 生成执行输入文件
python Alpha-SQL-master/script/build_sql_input.py \
  --source Alpha-SQL-master/mini_dev_mysql_alpaca_filtered.json \
  --target Alpha-SQL-master/data/mini_dev_mysql_alpaca_filtered_exec.json \
  --limit 50

# 3. 执行查询
python code/sql_exe.py --mode query \
  --input Alpha-SQL-master/data/mini_dev_mysql_alpaca_filtered_exec.json \
  --output result/dataset_exe_result.json \
  --host $SR_HOST --port $SR_PORT --user $SR_USER --password "$SR_PASSWORD" --db $SR_DB

# 4. 查看结果
cat result/dataset_exe_result.json | head -n 40
```

## 常见问题
- 数据库连接失败：确认端口、账号、库名是否正确；若是 StarRocks 需确认使用 MySQL 兼容端口。
- SQL 执行报错：检查原始数据集中的 SQL 是否依赖不存在的表或字段；可先用 `--limit` 少量测试。
- 大模型调用：若需与 MCTS 过程结合，请在 `alphasql/llm_call/openai_llm.py` 中适配你的 API 协议（参考 `augment_dataset.py`）。
- 结果格式不匹配：确保使用增强后的 `sql_exe.py`，并传入标准化输入 JSON。

## 修改点速览
| 文件 | 修改/新增 | 目的 |
|------|-----------|------|
| `config/mysql_exp.yaml` | 修改模型字段 | 更换为 API 可访问模型 |
| `code/sql_exe.py` | 增加 argparse & CLI | 允许灵活指定输入输出与执行模式 |
| `script/build_sql_input.py` | 新增脚本 | 将原数据集转换为批量执行输入格式 |
| `REPRODUCTION.md` | 新增文档 | 汇总复现与运行步骤 |

如需进一步：可加一个自动管线脚本整合“生成输入+执行+汇总”。欢迎继续需求。 
