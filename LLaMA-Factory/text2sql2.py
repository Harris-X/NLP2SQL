import json
import os
import http.client
import time
from http.client import RemoteDisconnected
from typing import List, Dict, Tuple

import pandas as pd


def read_schema_csv(path: str) -> pd.DataFrame:
    """Load a schema CSV, trying multiple encodings to avoid decode errors."""
    for encoding in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 0, f"Failed to decode {path} with tried encodings")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip BOM/空格，标准化列名。"""
    normalized_columns = [str(col).strip().lstrip("\ufeff") for col in df.columns]
    df.columns = normalized_columns
    return df


def build_create_table_stmt(table_name: str, df: pd.DataFrame) -> str:
    """基于 schema CSV 构造 CREATE TABLE 语句，包含列名、类型和注释。"""
    df = normalize_columns(df.copy())

    def get_col(name: str) -> pd.Series | None:
        return df[name] if name in df.columns else None

    col_name_col = get_col("original_column_name")
    col_desc_col = get_col("column_description")
    data_fmt_col = get_col("data_format")
    value_desc_col = get_col("value_description")

    if col_name_col is None:
        raise KeyError("original_column_name")

    col_defs = []
    for idx, raw_name in col_name_col.dropna().astype(str).items():
        name = raw_name.strip()
        if not name:
            continue

        # 类型处理
        dtype = "TEXT"
        if data_fmt_col is not None:
            fmt = str(data_fmt_col.iloc[idx]).strip()
            if fmt and fmt.lower() != "nan":
                fmt_lower = fmt.lower()
                if "int" in fmt_lower:
                    dtype = "BIGINT"
                elif "real" in fmt_lower or "double" in fmt_lower or "float" in fmt_lower:
                    dtype = "DOUBLE"
                else:
                    dtype = "TEXT"

        # 注释处理
        comments = []
        if col_desc_col is not None:
            desc = str(col_desc_col.iloc[idx]).strip()
            if desc and desc.lower() != "nan":
                comments.append(desc)

        if value_desc_col is not None:
            vdesc = str(value_desc_col.iloc[idx]).strip()
            if vdesc and vdesc.lower() != "nan":
                comments.append(vdesc)

        comment_part = ""
        if comments:
            # 简单的转义单引号
            joined_comment = "; ".join(comments).replace("'", "")
            comment_part = f" COMMENT '{joined_comment}'"

        col_defs.append(f"  `{name}` {dtype}{comment_part}")

    body = ",\n".join(col_defs)
    return f"CREATE TABLE `{table_name}` (\n{body}\n);"


SOURCE_JSON = "/root/autodl-tmp/comp/LLaMA-Factory/datasets/minidev/MINIDEV/mini_dev_mysql.json"
COLUMN_DIR = "datasets/minidev/MINIDEV/dev_databases"
OUTPUT_JSONL = "datasets/minidev/mini_dev_mysql_with_schema.jsonl"
OUTPUT_ALPACA = "datasets/minidev/mini_dev_mysql_alpaca.json"
PROMPT = "请你接下来一步步思考，写出正确的SQL查询语句以满足用户的需求。"
TRANSLATION_CACHE = "datasets/minidev/translation_cache.json"

# 通过环境变量配置 API
API_HOST = os.getenv("GPTBEST_API_HOST", "hk-api.gptbest.vip")
API_KEY = os.getenv("GPTBEST_API_KEY", "sk-GMYNUCidV96DStXskUpPqgemoaDur0alDXZkeyiq5E3mXGZn")  # 若为空则跳过翻译
FORCE_RETRANSLATE = os.getenv("FORCE_RETRANSLATE", "0") == "1"
RETRANSLATE_IDS = {i.strip() for i in os.getenv("RETRANSLATE_IDS", "").split(",") if i.strip()}
LIMIT_TRANSLATE = int(os.getenv("LIMIT_TRANSLATE", "0") or 0)  # 0表示不限制条数
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "1") or 1)
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "1.5") or 1.5)
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-5-mini") # claude-haiku-4-5-20251001
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "gpt-5.1")  # 备用模型：用于首次模型翻译效果较差时
MIN_CN_RATIO = float(os.getenv("MIN_CN_RATIO", "0.01"))  # 判定翻译质量的中文占比阈值
SCAN_CACHE_FOR_POOR = os.getenv("SCAN_CACHE_FOR_POOR", "1") == "1"  # 启动时扫描缓存标记差翻译
PRINT_POOR_SAMPLE_LIMIT = int(os.getenv("PRINT_POOR_SAMPLE_LIMIT", "10") or 10)
PREFALLBACK_FOR_POOR = os.getenv("PREFALLBACK_FOR_POOR", "1") == "1"  # 差翻译是否直接跳过主模型优先用备用模型
FIX_POOR_CACHE = os.getenv("FIX_POOR_CACHE", "1") == "1"  # 启动时对缓存中差翻译进行批量fallback重写


# nl2sqlite 中文模板（MySQL 方言）
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

df_mini = pd.read_json(SOURCE_JSON)

# 加载翻译缓存（question_id -> {question_cn, evidence_cn}）
if os.path.isfile(TRANSLATION_CACHE) and not FORCE_RETRANSLATE:
    try:
        with open(TRANSLATION_CACHE, "r", encoding="utf-8") as cf:
            translation_cache: Dict[str, Dict[str, str]] = json.load(cf)
    except Exception:
        translation_cache = {}
else:
    translation_cache = {}

# 扫描已有缓存，识别低中文占比的疑似未正确翻译条目
POOR_TRANSLATION_IDS = set()
## 延后扫描到 chinese_ratio 定义之后


def save_cache() -> None:
    os.makedirs(os.path.dirname(TRANSLATION_CACHE), exist_ok=True)
    with open(TRANSLATION_CACHE, "w", encoding="utf-8") as cf:
        json.dump(translation_cache, cf, ensure_ascii=False, indent=2)


def extract_identifiers(schema_sql: str) -> List[str]:
    """从CREATE TABLE片段中提取表名与列名集合，用于提示模型不要翻译这些标识符。"""
    import re
    identifiers = set()
    # 表名
    for m in re.finditer(r"CREATE TABLE\s+`([^`]+)`", schema_sql):
        identifiers.add(m.group(1))
    # 列名（位于反引号内）
    for m in re.finditer(r"\n\s*`([^`]+)`\s+[A-Z]+", schema_sql):
        identifiers.add(m.group(1))
    # 限制数量避免提示过长
    return sorted(list(identifiers))[:120]

def chinese_ratio(text: str) -> float:
    total = len(text)
    if total == 0:
        return 0.0
    cn = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    return cn / total

UNTRANSLATED_THRESHOLD = 0.3  # 中文占比低于此认为未译

# 现在执行缓存扫描（放在 chinese_ratio 定义之后，避免未定义函数）
if SCAN_CACHE_FOR_POOR and translation_cache:
    for _qid, _entry in translation_cache.items():
        qc = _entry.get("question_cn", "")
        ec = _entry.get("evidence_cn", "")
        if (chinese_ratio(qc) < MIN_CN_RATIO) or (chinese_ratio(ec) < MIN_CN_RATIO):
            POOR_TRANSLATION_IDS.add(str(_qid))
    if POOR_TRANSLATION_IDS:
        print(f"检测到疑似翻译不充分条目数：{len(POOR_TRANSLATION_IDS)} (中文占比 < {MIN_CN_RATIO})")
        for i, pid in enumerate(sorted(POOR_TRANSLATION_IDS)):
            if i >= PRINT_POOR_SAMPLE_LIMIT:
                break
            entry = translation_cache.get(pid, {})
            qc = entry.get("question_cn", "")
            ec = entry.get("evidence_cn", "")
            print(f"  例{ i+1 }: qid={pid}, ratioQ={chinese_ratio(qc):.2f}, ratioE={chinese_ratio(ec):.2f}")
    else:
        print("缓存中未发现明显低中文占比条目。")

def _sanitize_model_content(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        # 去掉第一行 ``` 或 ```json
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()
        content = "\n".join(lines).strip()
    if not content.startswith("{"):
        l = content.find("{")
        r = content.rfind("}")
        if l != -1 and r != -1 and r > l:
            content = content[l:r+1]
    return content

# 若需要在启动阶段就修复缓存中的差翻译（不依赖后续遍历数据集），执行批量重写
if FIX_POOR_CACHE and POOR_TRANSLATION_IDS:
    if not API_KEY:
        print("警告：FIX_POOR_CACHE=1 但没有设置 API_KEY，跳过缓存重写。")
    elif not FALLBACK_MODEL:
        print("警告：FIX_POOR_CACHE=1 但未提供 FALLBACK_MODEL，跳过缓存重写。")
    else:
        print(f"开始使用备用模型 {FALLBACK_MODEL} 修复缓存差翻译，共 {len(POOR_TRANSLATION_IDS)} 条。")
        # 轻量级请求函数（仅fallback，不尝试主模型）
        def _fallback_translate_for_cache(qid: str, original_q: str, original_e: str):
            id_hint = ""  # 缓存阶段不再注入schema以减少token消耗，可根据需要扩展
            sys_parts = [
                "You are a professional translator into Simplified Chinese.",
                "Translate ONLY natural language parts (descriptions, narrative).",
                "KEEP technical identifiers EXACTLY AS ORIGINAL.",
                "Return STRICT JSON with keys question_cn and evidence_cn only."
            ]
            sys_prompt = " \n".join(sys_parts)
            payload_messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps({"question": original_q, "evidence": original_e}, ensure_ascii=False)}
            ]
            body = {"model": FALLBACK_MODEL, "stream": False, "messages": payload_messages}
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    conn = http.client.HTTPSConnection(API_HOST, timeout=60)
                    headers = {"Accept": "application/json", "Authorization": API_KEY, "Content-Type": "application/json"}
                    conn.request("POST", "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode("utf-8"), headers)
                    res = conn.getresponse()
                    raw = res.read().decode("utf-8", errors="ignore")
                    data = json.loads(raw)
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    content = _sanitize_model_content(content)
                    try:
                        parsed = json.loads(content)
                        qc = parsed.get("question_cn") or original_q
                        ec = parsed.get("evidence_cn") or original_e
                    except Exception:
                        qc, ec = original_q, original_e
                        print(f"  [缓存重译] 非JSON格式，qid={qid}，截断：{content[:80]}")
                    if chinese_ratio(qc) > chinese_ratio(original_q) or chinese_ratio(ec) > chinese_ratio(original_e) or (chinese_ratio(qc) >= MIN_CN_RATIO) or (chinese_ratio(ec) >= MIN_CN_RATIO):
                        translation_cache[qid] = {"question_cn": qc, "evidence_cn": ec}
                        return True
                    return False
                except Exception as e:
                    wait = (RETRY_BACKOFF ** (attempt - 1))
                    print(f"  [缓存重译] 请求失败 qid={qid}，第{attempt}次重试，错误：{e}，{wait:.1f}s后重试...")
                    time.sleep(wait)
            return False

        fixed = 0
        for _qid in sorted(POOR_TRANSLATION_IDS):
            entry = translation_cache.get(_qid, {})
            oq = entry.get("question_cn") or entry.get("question") or ""
            oe = entry.get("evidence_cn") or entry.get("evidence") or ""
            if _fallback_translate_for_cache(_qid, oq, oe):
                fixed += 1
            if fixed and fixed % 20 == 0:
                save_cache()
                print(f"  已修复 {fixed} 条，进度自动保存。")
        save_cache()
        print(f"缓存差翻译修复完成，成功提升 {fixed}/{len(POOR_TRANSLATION_IDS)} 条。")

def translate_question_and_evidence(qid: str, question: str, evidence: str, identifiers: List[str], schema_snippet: str) -> Tuple[str, str]:
    """翻译并缓存。支持强制或按ID/自动检测重译；解析代码块；保留schema标识符；备用模型。"""
    need_retranslate = FORCE_RETRANSLATE or (qid in RETRANSLATE_IDS) or (qid in POOR_TRANSLATION_IDS)
    if qid in translation_cache and not need_retranslate:
        cached = translation_cache[qid]
        qc = cached.get("question_cn", question)
        ec = cached.get("evidence_cn", evidence)
        # 满足较高中文比例直接返回
        if chinese_ratio(qc) >= MIN_CN_RATIO or chinese_ratio(ec) >= MIN_CN_RATIO:
            return qc, ec  # 已译且质量较高
        # 否则标记为需要重译
        need_retranslate = True

    if not API_KEY:
        # 无API直接回写原文，保证字段完整
        translation_cache[qid] = {"question_cn": question, "evidence_cn": evidence}
        save_cache()
        return question, evidence

    id_hint = ", ".join(identifiers) if identifiers else ""
    schema_preview = schema_snippet
    sys_parts = [
        "You are a professional translator into Simplified Chinese.",
        "Translate ONLY natural language parts (descriptions, narrative).",
        "KEEP ALL technical identifiers EXACTLY AS ORIGINAL (table names, column names, SQL keywords, numbers, date strings).",
        "Do NOT translate anything inside backticks or listed identifiers.",
    ]
    if id_hint:
        sys_parts.append(f"Identifiers list (do not translate): {id_hint}")
    if schema_preview:
        sys_parts.append("Schema snippet (context only, do not translate identifiers):\n" + schema_preview)
    sys_parts.append("Return STRICT JSON with keys question_cn and evidence_cn only.")
    sys_prompt = " \n".join(sys_parts)

    payload_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps({"question": question, "evidence": evidence}, ensure_ascii=False)},
    ]
    # 初始使用主模型
    req_body_primary = {"model": PRIMARY_MODEL, "stream": False, "messages": payload_messages}

    # 重试与退避，仅在“成功或显著改善”时写入缓存
    last_error = None
    question_cn, evidence_cn = question, evidence
    # 若是差翻译并且偏好直接使用fallback且提供了备用模型，则跳过主模型
    if need_retranslate and PREFALLBACK_FOR_POOR and FALLBACK_MODEL:
        print(f"信息：qid={qid} 因差翻译直接使用备用模型 {FALLBACK_MODEL} 重译。")
        req_body_primary = {"model": FALLBACK_MODEL, "stream": False, "messages": payload_messages}
        primary_is_fallback = True
    else:
        primary_is_fallback = False

    for attempt in range(1, MAX_RETRIES + 1):  # 主模型(或直接fallback)重试
        try:
            conn = http.client.HTTPSConnection(API_HOST, timeout=45)
            headers = {"Accept": "application/json", "Authorization": API_KEY, "Content-Type": "application/json"}
            conn.request("POST", "/v1/chat/completions", json.dumps(req_body_primary, ensure_ascii=False).encode("utf-8"), headers)
            res = conn.getresponse()
            raw = res.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            content = _sanitize_model_content(content)
            try:
                parsed = json.loads(content)
                qc = parsed.get("question_cn") or question
                ec = parsed.get("evidence_cn") or evidence
            except Exception:
                qc, ec = question, evidence
                print(f"警告：翻译结果非JSON格式，qid={qid}，内容截断：{content[:120]}")

            # 成功条件：中文占比明显>threshold 或 比原文更高
            improved = (chinese_ratio(qc) > chinese_ratio(question)) or (chinese_ratio(ec) > chinese_ratio(evidence))
            if chinese_ratio(qc) >= MIN_CN_RATIO or chinese_ratio(ec) >= MIN_CN_RATIO or improved:
                question_cn, evidence_cn = qc, ec
                # 仅在成功或改善时写缓存
                translation_cache[qid] = {"question_cn": question_cn, "evidence_cn": evidence_cn}
                save_cache()
                return question_cn, evidence_cn
        except (RemoteDisconnected, TimeoutError, Exception) as e:
            last_error = e
            wait = (RETRY_BACKOFF ** (attempt - 1))
            print(f"警告：翻译请求失败，qid={qid}，第{attempt}次重试，错误：{e}，{wait:.1f}s后重试...")
            time.sleep(wait)

    # 若主模型未达到质量且存在备用模型，则尝试备用模型
    if FALLBACK_MODEL and not primary_is_fallback:
        print(f"信息：尝试备用模型 {FALLBACK_MODEL} 对 qid={qid} 进行重译……")
        req_body_fallback = {"model": FALLBACK_MODEL, "stream": False, "messages": payload_messages}
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                conn = http.client.HTTPSConnection(API_HOST, timeout=60)
                headers = {"Accept": "application/json", "Authorization": API_KEY, "Content-Type": "application/json"}
                conn.request("POST", "/v1/chat/completions", json.dumps(req_body_fallback, ensure_ascii=False).encode("utf-8"), headers)
                res = conn.getresponse()
                raw = res.read().decode("utf-8", errors="ignore")
                data = json.loads(raw)
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                content = _sanitize_model_content(content)
                try:
                    parsed = json.loads(content)
                    qc = parsed.get("question_cn") or question
                    ec = parsed.get("evidence_cn") or evidence
                except Exception:
                    qc, ec = question, evidence
                    print(f"警告：备用模型返回非JSON，qid={qid}，截断：{content[:120]}")

                improved = (chinese_ratio(qc) > chinese_ratio(question)) or (chinese_ratio(ec) > chinese_ratio(evidence))
                if chinese_ratio(qc) >= MIN_CN_RATIO or chinese_ratio(ec) >= MIN_CN_RATIO or improved:
                    question_cn, evidence_cn = qc, ec
                    translation_cache[qid] = {"question_cn": question_cn, "evidence_cn": evidence_cn}
                    save_cache()
                    return question_cn, evidence_cn
            except (RemoteDisconnected, TimeoutError, Exception) as e:
                last_error = e
                wait = (RETRY_BACKOFF ** (attempt - 1))
                print(f"警告：备用模型翻译失败，qid={qid}，第{attempt}次重试，错误：{e}，{wait:.1f}s后重试...")
                time.sleep(wait)

    # 若全部失败：返回缓存的已译版本（若存在且更好），否则原文；不覆盖已有成功缓存
    cached = translation_cache.get(qid)
    if cached:
        qc = cached.get("question_cn", question)
        ec = cached.get("evidence_cn", evidence)
        if chinese_ratio(qc) >= MIN_CN_RATIO or chinese_ratio(ec) >= MIN_CN_RATIO:
            return qc, ec
    print(f"警告：翻译最终失败，qid={qid}，返回原文。最后错误：{last_error}")
    return question, evidence

records = []
missing_schema_dbs = []
failed_tables = []

processed_count = 0
for _, row in df_mini.iterrows():
    if LIMIT_TRANSLATE and processed_count >= LIMIT_TRANSLATE:
        break
    question_id = row["question_id"]
    question = str(row["question"]).strip()
    sql_text = str(row["SQL"]).strip()
    db_id = row["db_id"]

    column_mini = os.path.join(COLUMN_DIR, db_id, "database_description")
    if not os.path.isdir(column_mini):
        missing_schema_dbs.append(db_id)
        column_content: List[str] = []
    else:
        column_content = []
        for item in sorted(os.listdir(column_mini)):
            if not item.lower().endswith(".csv"):
                continue

            column_path = os.path.join(column_mini, item)
            table_name = os.path.splitext(item)[0]
            try:
                content = read_schema_csv(column_path)
                create_table_sql = build_create_table_stmt(table_name, content)
            except UnicodeDecodeError as err:
                failed_tables.append((column_path, f"decode_error: {err}"))
                continue
            except KeyError:
                failed_tables.append((column_path, "missing_original_column_name"))
                continue

            column_content.append(create_table_sql)

    schema_snippet = "\n".join(column_content) if column_content else ""
    # 从原数据中提取 evidence 字段（若存在），否则为空
    evidence = str(row.get("evidence", "")).strip()

    identifiers = extract_identifiers(schema_snippet)
    # 翻译 question 与 evidence (强制翻译, 保留标识符)
    question_cn, evidence_cn = translate_question_and_evidence(str(question_id), question, evidence, identifiers, schema_snippet)

    # 使用中文模板构造最终的 Prompt（要求输出 MySQL 查询）
    prompt_text = TEMPLATE.format(
        dialect="MySQL",
        question=question_cn,
        db_schema=schema_snippet,
        evidence=evidence_cn,
    )

    records.append(
        {
            "instance_id": f"mini_{question_id}",
            "question_id": question_id,
            "db_id": db_id,
            "question": prompt_text,
            "original_question": question,
            "original_evidence": evidence,
            "question_cn": question_cn,
            "evidence_cn": evidence_cn,
            "output": sql_text,
        }
    )
    processed_count += 1

new_df = pd.DataFrame(records)
pd.set_option("display.max_colwidth", None)

os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
new_df.to_json(OUTPUT_JSONL, lines=True, orient="records", force_ascii=False)

alpaca_records = []
for _, row in new_df.iterrows():
    alpaca_records.append(
        {
            "instruction": PROMPT,
            "input": row["question"],  # 已是中文模板成品
            "output": row["output"],
            "meta": {
                "question_id": row["question_id"],
                "db_id": row["db_id"],
                "original_question": row["original_question"],
                "original_evidence": row["original_evidence"],
                "question_cn": row["question_cn"],
                "evidence_cn": row["evidence_cn"],
            },
        }
    )

with open(OUTPUT_ALPACA, "w", encoding="utf-8") as f:
    json.dump(alpaca_records, f, ensure_ascii=False, indent=2)

print(f"成功样本数：{len(new_df)}，已保存至 {OUTPUT_JSONL}")
print(f"Alpaca 格式样本已保存至 {OUTPUT_ALPACA}")

if missing_schema_dbs:
    unique_missing = sorted(set(missing_schema_dbs))
    print(f"缺少 schema 描述的数据库：{unique_missing}")

if failed_tables:
    print("以下表在解析时出错（仅显示前 10 项）：")
    for info in failed_tables[:10]:
        print(f"  {info[0]} -> {info[1]}")


