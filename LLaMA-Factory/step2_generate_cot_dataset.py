import json
import os
import time
import http.client
from typing import Dict, Any, List

INPUT_PATH = "mydata/dpo_datasets_all.json"
OUTPUT_PATH = "mydata/dpo_datasets_cot.json"

API_HOST = os.getenv("GPTBEST_API_HOST", "hk-api.gptbest.vip")
API_KEY = os.getenv("GPTBEST_API_KEY", "")
MODEL = os.getenv("PRIMARY_MODEL", "gpt-5.1")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3") or 3)
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "1.5") or 1.5)
TIMEOUT = int(os.getenv("API_TIMEOUT", "60") or 60)

SYSTEM_PROMPT = (
    "你是SQL领域的高级数据助理。\n"
    "给你instruction、input和一个参考答案。假设你没有看到参考答案，\n"
    "仅基于instruction和input用中文生成简洁的思维链（不要输出SQL，不要给最终答案）。\n"
    "输出控制在200字以内，保持条理化。"
)

PRIMER = (
    "下面是instruction、input，以及供对齐质量检查的参考答案（假装没看见参考答案）。\n"
    "请用中文输出思维链，不要给SQL，不要复述参考答案。"
)


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


def call_api(prompt: str) -> str:
    if not API_KEY:
        return "（未调用模型，缺少API_KEY）"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
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


def normalize_sql(sql_text: str) -> str:
    cleaned = sql_text.strip()
    if cleaned and not cleaned.endswith(";"):
        cleaned = cleaned + ";"
    return cleaned


def build_prompt(item: Dict[str, Any]) -> str:
    parts = [PRIMER]
    parts.append(f"Instruction:\n{item.get('instruction', '')}\n")
    parts.append(f"Input:\n{item.get('input', '')}\n")
    parts.append("参考答案（忽略，仅用于对齐）：\n" + item.get("output", ""))
    return "\n".join(parts)


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
    print(f"共 {len(samples)} 条样本，开始生成思维链并追加写入 {OUTPUT_PATH}")

    # 记忆已处理过的样本，避免对完全重复的 instruction/input/output 再次调用模型
    reuse_cache: Dict[tuple[str, str, str], str] = {}
    processed = 0
    for idx, item in enumerate(samples, 1):
        inst = item.get("instruction", "")
        inp = item.get("input", "")
        out = item.get("output", "")
        key = (inst, inp, out)

        if key in reuse_cache:
            combined_output = reuse_cache[key]
        else:
            prompt = build_prompt(item)
            reasoning = call_api(prompt)
            normalized = normalize_sql(out)
            combined_output = f"{reasoning}\n\n```sql\n{normalized}\n```"
            reuse_cache[key] = combined_output

        new_item = {"instruction": inst, "input": inp, "output": combined_output}

        append_json_array(OUTPUT_PATH, new_item)
        processed += 1
        print(f"[{processed}/{len(samples)}] 已写入", flush=True)

    print("完成。")


if __name__ == "__main__":
    main()
