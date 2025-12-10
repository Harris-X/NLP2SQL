import json
import os
import re
import time
import http.client
from typing import List, Dict, Any

# Configuration
# 优先使用经过筛选的高质量数据集
INPUT_FILE = "datasets/minidev/mini_dev_mysql_alpaca_filtered.json"
if not os.path.exists(INPUT_FILE):
    INPUT_FILE = "datasets/minidev/mini_dev_mysql_alpaca.json"

OUTPUT_FILE = "datasets/minidev/mini_dev_mysql_alpaca_augmented.json"

# API Configuration
API_HOST = os.getenv("GPTBEST_API_HOST", "hk-api.gptbest.vip")
API_KEY = os.getenv("GPTBEST_API_KEY", "sk-GMYNUCidV96DStXskUpPqgemoaDur0alDXZkeyiq5E3mXGZn")
MODEL = os.getenv("AUGMENT_MODEL", "gpt-4o-mini") # 使用较强模型进行改写
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5

# Template (Must match text2sql2.py)
TEMPLATE = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

"""

def extract_all_sections(input_text):
    """Extract question, db_schema and evidence from the existing input text."""
    # Pattern:
    # ...【用户问题】\n(.*?)\n\n【数据库schema】\n(.*?)\n\n【参考信息】\n(.*?)\n\n【用户问题】...
    
    q1_pattern = r"【用户问题】\n(.*?)\n\n【数据库schema】"
    schema_pattern = r"【数据库schema】\n(.*?)\n\n【参考信息】"
    evidence_pattern = r"【参考信息】\n(.*?)\n\n【用户问题】"
    
    q_match = re.search(q1_pattern, input_text, re.DOTALL)
    schema_match = re.search(schema_pattern, input_text, re.DOTALL)
    evidence_match = re.search(evidence_pattern, input_text, re.DOTALL)
    
    question = q_match.group(1).strip() if q_match else ""
    db_schema = schema_match.group(1).strip() if schema_match else ""
    evidence = evidence_match.group(1).strip() if evidence_match else ""
    
    return question, db_schema, evidence

def call_llm(messages):
    if not API_KEY:
        print("Error: API_KEY not found.")
        return None

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0.7
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = http.client.HTTPSConnection(API_HOST, timeout=60)
            headers = {
                "Accept": "application/json",
                "Authorization": API_KEY,
                "Content-Type": "application/json"
            }
            conn.request("POST", "/v1/chat/completions", json.dumps(payload).encode("utf-8"), headers)
            res = conn.getresponse()
            raw = res.read().decode("utf-8", errors="ignore")
            
            if res.status != 200:
                print(f"API Error {res.status}: {raw}")
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
                
            data = json.loads(raw)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return content
            
        except Exception as e:
            print(f"Request failed (attempt {attempt}): {e}")
            time.sleep(RETRY_BACKOFF ** attempt)
            
    return None

def parse_json_response(content):
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        # Remove first line if it is ``` or ```json
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        # Remove last line if it is ```
        while lines and lines[-1].strip() == "```":
            lines.pop()
        content = "\n".join(lines)
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {content[:100]}...")
        return None

def augment_record(record, index):
    meta = record.get("meta", {})
    input_text = record.get("input", "")
    original_sql = record.get("output", "")
    
    # Try to get from meta, else extract
    question_cn = meta.get("question_cn")
    evidence_cn = meta.get("evidence_cn")
    db_schema = ""
    
    if not question_cn or not evidence_cn:
        q_ext, s_ext, e_ext = extract_all_sections(input_text)
        if not question_cn: question_cn = q_ext
        if not evidence_cn: evidence_cn = e_ext
        db_schema = s_ext
    else:
        # Still need schema
        _, db_schema, _ = extract_all_sections(input_text)
        
    if not question_cn or not original_sql:
        # print(f"Skipping record {index}: Missing question or SQL")
        return []
        
    question_id = meta.get("question_id", f"aug_idx_{index}")
    db_id = meta.get("db_id", "unknown")
    
    # Construct Prompt
    system_prompt = """You are a data augmentation expert for Text-to-SQL tasks.
Your goal is to generate 3 new variations of the given (Question, SQL) pair based on specific modes.
The input is in Chinese (Question) and SQL.

The modes are:
1. **Keyword Replacement** ("keyword"): Identify a specific entity, number, or value in the Question and SQL. Replace it with a different but plausible value. The structure of the SQL remains the same, only the value changes.
2. **Paraphrasing** ("paraphrase"): Rewrite the Question in Chinese using different words, sentence structure, or synonyms, but keep the meaning EXACTLY the same. The SQL MUST remain unchanged.
3. **Simplification** ("simplification"): Simplify the Question to ask for fewer columns or details (e.g., "List name and age" -> "List name"). Adjust the SQL SELECT clause to match the simplified question.

Output Format:
Return a JSON object with a key "variations" containing a list of objects. Each object should have:
- "mode": "keyword" | "paraphrase" | "simplification"
- "question": The new Chinese question.
- "sql": The new SQL (or original if unchanged).

Example Output:
{
  "variations": [
    {"mode": "keyword", "question": "...", "sql": "..."},
    {"mode": "paraphrase", "question": "...", "sql": "..."},
    {"mode": "simplification", "question": "...", "sql": "..."}
  ]
}
"""

    user_content = f"""
Original Question: {question_cn}
Original SQL: {original_sql}
Schema Context:
{db_schema}
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    response = call_llm(messages)
    if not response:
        return []
        
    data = parse_json_response(response)
    if not data or "variations" not in data:
        return []
        
    new_records = []
    for var in data["variations"]:
        new_q = var.get("question")
        new_sql = var.get("sql")
        mode = var.get("mode")
        
        if not new_q or not new_sql:
            continue
            
        # Reconstruct Input
        new_input = TEMPLATE.format(
            dialect="MySQL",
            question=new_q,
            db_schema=db_schema,
            evidence=evidence_cn
        )
        
        new_record = {
            "instruction": record["instruction"],
            "input": new_input,
            "output": new_sql,
            "meta": {
                "question_id": f"{question_id}_{mode}",
                "db_id": db_id,
                "original_question": meta.get("original_question", question_cn),
                "question_cn": new_q,
                "evidence_cn": evidence_cn,
                "augmentation_mode": mode,
                "parent_id": question_id
            }
        }
        new_records.append(new_record)
        
    return new_records

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return

    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} records. Starting augmentation...")
    
    augmented_data = []
    
    # 限制处理数量用于测试，或者处理全部
    # 这里默认处理全部，但可以设置 LIMIT_AUGMENT 环境变量
    limit = int(os.getenv("LIMIT_AUGMENT", "0"))
    
    count = 0
    for i, record in enumerate(data):
        if limit > 0 and count >= limit:
            break
            
        print(f"Processing {i+1}/{len(data)}: {record.get('meta', {}).get('question_id', f'idx_{i}')}")
        
        # Add original record
        augmented_data.append(record)
        
        # Generate augmentations
        try:
            new_items = augment_record(record, i)
            augmented_data.extend(new_items)
            print(f"  Generated {len(new_items)} variations.")
        except Exception as e:
            print(f"  Error augmenting record {i}: {e}")
        
        count += 1
        
        # Save periodically
        if (i + 1) % 10 == 0:
             with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(augmented_data, f, ensure_ascii=False, indent=2)
                
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(augmented_data, f, ensure_ascii=False, indent=2)
        
    print(f"Done. Total records: {len(augmented_data)}. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
