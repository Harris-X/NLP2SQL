import json
import os
from typing import Optional

import pandas as pd

# 读取数据
df_snow = pd.read_json("Spider2/spider2-snow/spider2-snow.jsonl", lines=True)
df_schema = pd.read_json("Spider2/methods/gold-tables/spider2-snow-gold-tables.jsonl", lines=True)

SCHEMA = True
base_path = "Spider2/spider2-snow/resource/databases/"
sql_base_path = "Spider2/spider2-snow/evaluation_suite/gold/sql/"  # SQL 文件的基础路径


def resolve_schema_path(json_dir: str, expected_name: str) -> Optional[str]:
    """Return existing schema path, tolerating case differences."""
    candidate = os.path.join(json_dir, expected_name)
    if os.path.exists(candidate):
        return candidate

    if not os.path.isdir(json_dir):
        return None

    target = expected_name.lower()
    for entry in os.listdir(json_dir):
        if entry.lower() == target:
            return os.path.join(json_dir, entry)
    return None

# 存储新 DataFrame 的数据（列表形式，每个元素是一行数据）
new_data = []
failed_instances = []

for index, row in df_snow.iterrows():
    instance_id = row['instance_id']
    original_question = row['instruction']
    
    # 1. 获取当前 instance_id 对应的 gold_tables
    # 注意：df_schema["gold_tables"] 可能是列表类型（如 ["CHICAGO.TAXI_TRIPS"]），需先提取字符串
    matches = df_schema[df_schema["instance_id"] == instance_id]["gold_tables"]
    if matches.empty:
        print(f"未找到 gold_tables（instance_id: {instance_id}），跳过。")
        continue

    gold_tables = matches.values[0]
    # 如果 gold_tables 是列表（如 ["xxx"]），取第一个元素；如果是字符串则直接使用
    if isinstance(gold_tables, list):
        gold_tables_str = gold_tables[0]
    else:
        gold_tables_str = gold_tables
    
    # 2. 构建 JSON 文件路径并读取 schema
    parts = gold_tables_str.split(".")
    json_file_name = parts[-1] + ".json"
    json_dir = os.path.join(base_path, *parts[:-1])
    json_path = resolve_schema_path(json_dir, json_file_name)
    if not json_path:
        print(f"未找到 schema 文件（instance_id: {instance_id}）：{os.path.join(json_dir, json_file_name)}")
        continue
    print(f"Processing instance_id: {instance_id}, JSON path: {json_path}")
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            schema_payload = json.load(f)
    except Exception as e:
        print(f"读取 schema 失败（instance_id: {instance_id}）：{e}")
        continue  # 跳过错误行
    
    # 3. 处理 column_names 并与原始问题拼接
    if SCHEMA and isinstance(schema_payload, dict) and schema_payload.get('column_names'):
        # 提取 column_names（假设是列表，如 ["company", "taxi_id"...]）
        column_names = schema_payload['column_names']
        if not isinstance(column_names, (list, tuple)):
            column_names = [str(column_names)]
        else:
            column_names = [str(col) for col in column_names]
        
        # 将列名列表转为字符串（例如："column names: company, dropoff_community_area, taxi_id, ..."）
        columns_str = "column names: " + ", ".join(column_names)
        
        # 拼接原始问题和列名信息（格式可根据需求调整）
        prompt_prefix = (
            "You are an expert data analyst. Please write a MySQL query that satisfies the requirement below.\n"
            # "Requirement: "
        )
        new_question = f"{original_question} {columns_str}" # {prompt_prefix}
    else:
        # 若不需要 schema 或无 column_names，直接使用原始问题
        new_question = original_question
    
    # 3. 读取对应 instance_id 的 SQL 文件内容
    sql_file_name = f"{instance_id}.sql"  # SQL 文件名与 instance_id 一致
    sql_path = os.path.join(sql_base_path, sql_file_name)
    
    try:
        with open(sql_path, "r", encoding="utf-8") as f:
            sql_content = f.read().strip()  # 读取 SQL 内容并去除首尾空白
    except FileNotFoundError:
        print(f"SQL 文件不存在（instance_id: {instance_id}）：{sql_path}")
        failed_instances.append(instance_id)
        continue
    except Exception as e:
        print(f"读取 SQL 失败（instance_id: {instance_id}）：{e}")
        failed_instances.append(instance_id)
        continue


    # 4. 将数据加入新列表（新增 output 列存储 SQL 内容）
    new_data.append({
        "instance_id": instance_id,
        "question": new_question,
        "output": sql_content  # 新增列：SQL 文件内容
    })

# 5. 创建新的 DataFrame
new_df = pd.DataFrame(new_data)

# 取消字符串截断（设置为 None 表示不限制长度）
pd.set_option('display.max_colwidth', None)
# 查看结果（可选）
print(new_df[new_df["instance_id"]=="sf_bq011"]["output"])

# 保存为 JSONL 文件
output_path = "Spider2/spider2-snow/qwen3_sft_dataset.jsonl"
new_df.to_json(output_path, lines=True, orient="records", force_ascii=False)
print(f"成功样本数：{len(new_df)}，已保存至 {output_path}")

prompt = "You are an expert data analyst. Please write a MySQL query that satisfies the requirement below." # \nRequirement: 
# 生成 Alpaca 格式数据
alpaca_records = [
    {
        "instruction": prompt,
        "input": row["question"] ,
        "output": row["output"],
    }
    for _, row in new_df.iterrows()
]

alpaca_output_path = "Spider2/spider2-snow/qwen3_sft_dataset_alpaca.json"
with open(alpaca_output_path, "w", encoding="utf-8") as f:
    json.dump(alpaca_records, f, ensure_ascii=False, indent=2)
print(f"Alpaca 格式样本已保存至 {alpaca_output_path}")

if failed_instances:
    print(f"跳过的 instance_id 共 {len(failed_instances)} 个：{failed_instances[:10]}{'...' if len(failed_instances) > 10 else ''}")