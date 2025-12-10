import json
import os

# Paths
EXE_RESULT_JSON = "datasets/minidev/dataset_exe_result.json"
FINAL_DATASET_JSON = "datasets/minidev/final_dataset.json"
SCHEMA_JSON = "datasets/minidev/schema.json"
OUTPUT_ALPACA = "datasets/minidev/mini_dev_mysql_alpaca_filtered.json"

# 与 text2sql2.py 保持一致：不使用 instruction，全部放入 input 中。
PROMPT = "请你接下来一步步思考，写出正确的SQL查询语句以满足用户的需求。"  # 留空

# 中文模板，加入 evidence（参考信息）段落，最后重复用户问题并以 ```sql 开始代码块
TEMPLATE = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

"""

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def map_dtype(dtype_str):
    dtype_str = str(dtype_str).lower()
    if "int" in dtype_str:
        return "BIGINT"
    elif "double" in dtype_str or "float" in dtype_str or "real" in dtype_str:
        return "DOUBLE"
    else:
        return "TEXT"

def build_create_table_stmt(table_name, table_info):
    """
    Constructs a CREATE TABLE statement from the schema info.
    table_info is a dict with keys: 'table_description', 'columns'
    """
    columns = table_info.get('columns', [])
    
    col_defs = []
    for col in columns:
        col_name = col.get('col', '').strip()
        if not col_name:
            continue
            
        col_type = col.get('type', 'string')
        col_desc = col.get('description', '').strip()
        
        sql_type = map_dtype(col_type)
        
        comment_part = ""
        if col_desc and col_desc.lower() != "nan":
            # Escape single quotes in comment
            safe_desc = col_desc.replace("'", "")
            comment_part = f" COMMENT '{safe_desc}'"
            
        col_defs.append(f"  `{col_name}` {sql_type}{comment_part}")
        
    body = ",\n".join(col_defs)
    return f"CREATE TABLE `{table_name}` (\n{body}\n);"

def main():
    # 1. Load Execution Results and filter for success
    print(f"Loading execution results from {EXE_RESULT_JSON}...")
    exe_results = load_json(EXE_RESULT_JSON)
    
    # Create a lookup for successful SQLs: sql_id -> sql_text
    success_map = {}
    for item in exe_results:
        if item.get("status") == "success" and "sql_id" in item:
            success_map[item["sql_id"]] = item.get("sql", "")
    
    print(f"Found {len(success_map)} successful SQL executions.")

    # 2. Load Schema and build lookup
    print(f"Loading schema from {SCHEMA_JSON}...")
    schema_list = load_json(SCHEMA_JSON)
    schema_map = {}
    for table in schema_list:
        t_name = table.get("table_name")
        if t_name:
            schema_map[t_name] = table

    # 3. Load Final Dataset and process
    print(f"Loading final dataset from {FINAL_DATASET_JSON}...")
    final_dataset = load_json(FINAL_DATASET_JSON)
    
    alpaca_records = []
    processed_count = 0
    missing_table_count = 0
    
    for item in final_dataset:
        sql_id = item.get("sql_id")
        
        # Filter by success status
        if sql_id not in success_map:
            continue
            
        question = item.get("question", "").strip()
        table_list = item.get("table_list", [])
        
        # Get SQL from exe_result (guaranteed to be the one that succeeded)
        sql_text = success_map[sql_id].strip()
        
        # Build Schema Prompt
        create_table_stmts = []
        for table_name in table_list:
            if table_name in schema_map:
                stmt = build_create_table_stmt(table_name, schema_map[table_name])
                create_table_stmts.append(stmt)
            else:
                # print(f"Warning: Table {table_name} not found in schema.")
                missing_table_count += 1
        
        schema_context = "\n".join(create_table_stmts)
        
        evidence = item.get("knowledge", "").strip()
        # 构造模板 prompt
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
        alpaca_records.append(record)
        processed_count += 1

    print(f"Processed {processed_count} records.")
    if missing_table_count > 0:
        print(f"Warning: {missing_table_count} tables were missing from schema.")

    # Save
    with open(OUTPUT_ALPACA, 'w', encoding='utf-8') as f:
        json.dump(alpaca_records, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {OUTPUT_ALPACA}")

if __name__ == "__main__":
    main()
