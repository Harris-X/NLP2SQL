import json
import os
import re
import time
import http.client
from typing import Dict, Any, List

INPUT_PATH = "mydata/dpo_datasets_cot.json"
OUTPUT_PATH = "mydata/dpo_datasets_cot_paraphrased.json"

API_HOST = os.getenv("GPTBEST_API_HOST", "hk-api.gptbest.vip")
API_KEY = os.getenv("GPTBEST_API_KEY", "sk-GMYNUCidV96DStXskUpPqgemoaDur0alDXZkeyiq5E3mXGZn")
MODEL = os.getenv("PRIMARY_MODEL", "claude-opus-4-5-20251101-thinking")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3") or 3)
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "1.5") or 1.5)
TIMEOUT = int(os.getenv("API_TIMEOUT", "60") or 60)

SYSTEM_PROMPT = (
    "你是SQL任务场景的中文改写助手。\n"
    "给定用户问题，请在保持语义等价的前提下，生成一句新的提问。\n"
    "不要增删信息，不要输出SQL或分析，只输出改写后的问题。"
)

PRIMER = "请改写下面的用户问题，保持含义一致：\n{question}"


def _sanitize_model_content(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()
        content = "\n".join(lines).strip()
    return content


def call_api(question: str) -> str:
    if not API_KEY:
        return "（未调用模型，缺少API_KEY）"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PRIMER.format(question=question)},
    ]
    body = {"model": MODEL, "messages": messages, "stream": False}

    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            conn = http.client.HTTPSConnection(API_HOST, timeout=TIMEOUT)
            headers = {
                "Accept": "application/json",
                "Authorization": API_KEY,
                "Content-Type": "application/json",
            }
            payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
            conn.request("POST", "/v1/chat/completions", payload, headers)
            res = conn.getresponse()
            raw = res.read().decode("utf-8", errors="ignore")
            data = json.loads(raw)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return _sanitize_model_content(content or "")
        except Exception as e:  # noqa: BLE001
            last_error = e
            wait = RETRY_BACKOFF ** (attempt - 1)
            print(f"API 调用失败，第{attempt}次，{e}，{wait:.1f}s 后重试...", flush=True)
            time.sleep(wait)
    return f"（模型多次失败：{last_error}）"


def extract_question(input_text: str) -> str | None:
    # 捕获【用户问题】后到下一个标题或文本结束的内容
    pattern = r"【用户问题】\s*\n(.*?)(?=\n\s*【|\Z)"
    m = re.search(pattern, input_text, flags=re.S)
    if m:
        return m.group(1).strip()
    return None


def replace_question(input_text: str, new_question: str) -> str:
    pattern = r"(【用户问题】\s*\n)(.*?)(?=\n\s*【|\Z)"
    return re.sub(pattern, lambda m: m.group(1) + new_question, input_text, flags=re.S)


def append_json_array(path: str, obj: Dict[str, Any]) -> None:
    text = json.dumps(obj, ensure_ascii=False, indent=2)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("[\n")
            f.write(text)
            f.write("\n]\n")
            f.flush()
            os.fsync(f.fileno())
        return

    with open(path, "r+", encoding="utf-8") as f:
        content = f.read().rstrip()
        if content.endswith("]"):
            content = content[:-1].rstrip()
        if content.endswith("["):
            new_content = content + text + "\n]\n"
        else:
            new_content = content + ",\n" + text + "\n]\n"
        f.seek(0)
        f.write(new_content)
        f.truncate()
        f.flush()
        os.fsync(f.fileno())


def load_source(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    samples = load_source(INPUT_PATH)
    print(f"共 {len(samples)} 条样本，开始改写用户问题并写入 {OUTPUT_PATH}")

    paraphrase_cache: Dict[str, str] = {}
    processed = 0
    skipped = 0

    for idx, item in enumerate(samples, 1):
        input_text = item.get("input", "")
        question = extract_question(input_text)
        if not question:
            skipped += 1
            append_json_array(OUTPUT_PATH, item)
            print(f"[{idx}/{len(samples)}] 未找到用户问题，原样写入。", flush=True)
            continue

        if question in paraphrase_cache:
            new_question = paraphrase_cache[question]
        else:
            new_question = call_api(question)
            paraphrase_cache[question] = new_question

        new_input = replace_question(input_text, new_question)
        new_item = {
            "instruction": item.get("instruction", ""),
            "input": new_input,
            "output": item.get("output", ""),
        }

        append_json_array(OUTPUT_PATH, new_item)
        processed += 1
        print(f"[{idx}/{len(samples)}] 已改写并写入。", flush=True)

    print(f"完成。改写 {processed} 条，未改写 {skipped} 条。")


if __name__ == "__main__":
    main()
