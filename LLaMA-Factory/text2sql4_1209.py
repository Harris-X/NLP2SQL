#!/usr/bin/env python3
"""Construct an Alpaca-style Text2SQL dataset from the 1209_wrong dump.

- 读取失败 SQL 列表（index.txt），仅保留未出现在失败列表中的样本。
- 从 sql_submission1208 目录加载对应 SQL 作为 gold output。
- 从 final_dataset.json 取问题与表清单，依据 schema.json 构造 CREATE TABLE 片段写入 prompt。
- 生成字段：question_id, instruction, input, output。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

# 路径配置
BASE_DIR = Path("datasets/1209_wrong")
FAILED_INDEX = BASE_DIR / "index.txt"
FINAL_DATASET_JSON = BASE_DIR / "final_dataset.json"
SCHEMA_JSON = BASE_DIR / "schema.json"
SQL_DIR = BASE_DIR / "sql_submission1208"
OUTPUT_ALPACA = BASE_DIR / "sql_submission1208_correct.json"

# 与 text2sql3.py 保持一致：instruction 不为空，主要内容在 input
PROMPT = "请你接下来一步步思考，写出正确的SQL查询语句以满足用户的需求。"

TEMPLATE = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

"""

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_failed_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    tokens = re.split(r"[\s,]+", text)
    return {t.strip().strip('"').strip("'") for t in tokens if t.strip()}


def map_dtype(dtype_str: str) -> str:
    dtype_str = str(dtype_str).lower()
    if "int" in dtype_str:
        return "BIGINT"
    if any(x in dtype_str for x in ["double", "float", "real"]):
        return "DOUBLE"
    return "TEXT"


def build_create_table_stmt(table_name: str, table_info: Dict[str, Any]) -> str:
    columns = table_info.get("columns", []) or []
    col_defs: List[str] = []
    for col in columns:
        col_name = (col.get("col") or "").strip()
        if not col_name:
            continue
        col_type = col.get("type", "string")
        col_desc = (col.get("description") or "").strip()
        sql_type = map_dtype(col_type)
        comment_part = ""
        if col_desc and col_desc.lower() != "nan":
            safe_desc = col_desc.replace("'", "")
            comment_part = f" COMMENT '{safe_desc}'"
        col_defs.append(f"  `{col_name}` {sql_type}{comment_part}")
    body = ",\n".join(col_defs)
    return f"CREATE TABLE `{table_name}` (\n{body}\n);"


def load_schema_map(path: Path) -> Dict[str, Dict[str, Any]]:
    schema_list = load_json(path)
    return {item.get("table_name"): item for item in schema_list if item.get("table_name")}


def load_success_sql(sql_dir: Path, failed_ids: set[str]) -> Dict[str, str]:
    success_map: Dict[str, str] = {}
    for sql_file in sql_dir.glob("*.sql"):
        sql_id = sql_file.stem
        if sql_id in failed_ids:
            continue
        sql_text = sql_file.read_text(encoding="utf-8").strip()
        if not sql_text:
            continue
        success_map[sql_id] = sql_text
    return success_map


def main() -> None:
    failed_ids = parse_failed_ids(FAILED_INDEX)
    print(f"Failed SQL count: {len(failed_ids)}")

    success_map = load_success_sql(SQL_DIR, failed_ids)
    print(f"Loaded {len(success_map)} successful SQL files from {SQL_DIR}")

    schema_map = load_schema_map(SCHEMA_JSON)
    print(f"Loaded {len(schema_map)} tables from schema")

    final_dataset = load_json(FINAL_DATASET_JSON)
    print(f"Loaded {len(final_dataset)} items from final_dataset")

    records: List[Dict[str, Any]] = []
    processed = 0
    skipped_missing_sql = 0
    missing_tables = 0

    for item in final_dataset:
        sql_id = item.get("sql_id")
        if not sql_id or sql_id in failed_ids:
            continue
        if sql_id not in success_map:
            skipped_missing_sql += 1
            continue

        question = (item.get("question") or "").strip()
        table_list = item.get("table_list") or []
        evidence = (item.get("knowledge") or "").strip()
        sql_text = success_map[sql_id]

        create_table_stmts: List[str] = []
        for table_name in table_list:
            table_info = schema_map.get(table_name)
            if not table_info:
                missing_tables += 1
                continue
            create_table_stmts.append(build_create_table_stmt(table_name, table_info))
        schema_context = "\n".join(create_table_stmts)

        input_text = TEMPLATE.format(
            dialect="MySQL",
            question=question,
            db_schema=schema_context,
            evidence=evidence,
        )

        record = {
            "question_id": sql_id,
            "instruction": PROMPT,
            "input": input_text,
            "output": sql_text,
        }
        records.append(record)
        processed += 1

    print(f"Processed {processed} records. Skipped missing-sql: {skipped_missing_sql}, missing_tables: {missing_tables}")

    OUTPUT_ALPACA.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_ALPACA.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(records)} records to {OUTPUT_ALPACA}")


if __name__ == "__main__":
    main()
