from openai import OpenAI
import dotenv
from typing import List, Optional
from alphasql.llm_call.cost_recoder import CostRecorder
import time
import os
import http.client
import json
from alphasql.utils.run_logger import get_logger
from datetime import datetime
from pathlib import Path

dotenv.load_dotenv(override=True)

DEFAULT_COST_RECORDER = CostRecorder(model="Qwen3-Coder-30B-A3B-Instruct")

MAX_RETRYING_TIMES = 5

# MAX_TIMEOUT = 60

N_CALLING_STRATEGY_SINGLE = "single"
N_CALLING_STRATEGY_MULTIPLE = "multiple"

def call_openai(prompt: str,
                model: str,
                temperature: float = 0.0,
                top_p: float = 1.0,
                n: int = 1,
                max_tokens: int = 512,
                stop: List[str] = None,
                base_url: str = None,
                api_key: str = None,
                n_strategy: str = N_CALLING_STRATEGY_SINGLE,
                cost_recorder: Optional[CostRecorder] = DEFAULT_COST_RECORDER,
                messages: Optional[List[dict]] = None) -> List[str]:
    """统一的 LLM 调用入口。

    优先使用自定义直连接口 (环境变量 GPTBEST_API_HOST / GPTBEST_API_KEY)，
    否则回退到 OpenAI SDK (支持 OPENAI_BASE_URL / OPENAI_API_KEY)。
    返回 contents: List[str] 保持与原逻辑兼容。
    """
    logger = get_logger()
    custom_host = os.getenv("GPTBEST_API_HOST")
    custom_key = os.getenv("GPTBEST_API_KEY")
    full_log = os.getenv("ALPHASQL_LOG_FULL", "0") == "1"
    dialogues_path = os.getenv("ALPHASQL_LLM_LOG_PATH", str(Path(os.getenv("ALPHASQL_LOG_DIR", "logs")) / "llm_dialogues.jsonl"))
    try:
        Path(dialogues_path).parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # 若用户希望覆盖 base_url/api_key，也允许传入参数层面
    sdk_base_url = base_url or os.getenv("OPENAI_BASE_URL")
    sdk_api_key = api_key or os.getenv("OPENAI_API_KEY")

    # 分支1：使用自定义 HTTP 接口
    if custom_host and custom_key:
        attempts = 0
        contents: List[str] = []
        while attempts < MAX_RETRYING_TIMES:
            try:
                start = time.time()
                truncated_prompt = (prompt)
                logger.info(f"LLM HTTP call attempt={attempts+1} model={model} len(prompt)={len(prompt)} prompt_head={truncated_prompt}")
                payload = {
                    "model": model,
                    "messages": messages if messages else [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": False,
                    "max_tokens": max_tokens,
                    "n": n if n_strategy == N_CALLING_STRATEGY_SINGLE else 1
                }
                conn = http.client.HTTPSConnection(custom_host, timeout=120)
                headers = {
                    "Accept": "application/json",
                    "Authorization": custom_key,
                    "Content-Type": "application/json"
                }
                conn.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
                res = conn.getresponse()
                raw = res.read().decode("utf-8", errors="ignore")
                latency = time.time() - start
                if res.status != 200:
                    logger.warning(f"LLM HTTP error status={res.status} latency={latency:.2f}s body={raw}")
                    attempts += 1
                    time.sleep(2 ** attempts)
                    continue
                data = json.loads(raw)
                choices = data.get("choices", [])
                if n > 1 and n_strategy == N_CALLING_STRATEGY_MULTIPLE:
                    # multiple 策略：循环调用 n 次
                    contents.append(choices[0].get("message", {}).get("content", "").strip())
                    # 继续补齐剩余次数
                    for _ in range(n - 1):
                        time.sleep(0.1)
                        conn = http.client.HTTPSConnection(custom_host, timeout=120)
                        conn.request("POST", "/v1/chat/completions", json.dumps(payload), headers)
                        res2 = conn.getresponse()
                        raw2 = res2.read().decode("utf-8", errors="ignore")
                        if res2.status != 200:
                            print(f"[LLM HTTP ERROR MULTI] {res2.status}: {raw2}")
                            continue
                        data2 = json.loads(raw2)
                        c2 = data2.get("choices", [])
                        if c2:
                            contents.append(c2[0].get("message", {}).get("content", "").strip())
                else:
                    contents = [c.get("message", {}).get("content", "").strip() for c in choices]
                # 记录完整对话
                entry = {
                    "ts": datetime.now().isoformat(),
                    "mode": "http",
                    "model": model,
                    "strategy": n_strategy,
                    "n": n,
                    "latency": latency,
                    "prompt": prompt if full_log else (prompt),
                    "messages": messages if (full_log and messages) else None,
                    "outputs": contents if full_log else [o for o in contents]
                }
                try:
                    with open(dialogues_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                logger.info(f"LLM HTTP success latency={latency:.2f}s outputs={len(contents)}")
                break
            except Exception as e:
                logger.error(f"LLM HTTP exception attempt={attempts+1}: {e}")
                attempts += 1
                time.sleep(2 ** attempts)
        if not contents:
            contents = [""]
        return contents

    # 分支2：回退到 OpenAI 官方 SDK
    client = OpenAI()
    if sdk_base_url:
        client.base_url = sdk_base_url
    if sdk_api_key:
        client.api_key = sdk_api_key

    retrying = 0
    contents: List[str] = []
    while retrying < MAX_RETRYING_TIMES:
        try:
            if n == 1 or (n > 1 and n_strategy == N_CALLING_STRATEGY_SINGLE):
                start = time.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=messages if messages else [{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=n,
                    top_p=top_p,
                    stop=stop,
                )
                latency = time.time() - start
                if cost_recorder is not None and hasattr(response, "usage"):
                    cost_recorder.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                contents = [choice.message.content for choice in response.choices]
                # 记录完整对话
                entry = {
                    "ts": datetime.now().isoformat(),
                    "mode": "sdk",
                    "model": model,
                    "strategy": n_strategy,
                    "n": n,
                    "latency": latency,
                    "prompt": prompt if full_log else (prompt),
                    "messages": messages if (full_log and messages) else None,
                    "outputs": contents if full_log else [o for o in contents]
                }
                try:
                    with open(dialogues_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                logger.info(f"LLM SDK success model={model} latency={latency:.2f}s n={len(contents)}")
                break
            elif n > 1 and n_strategy == N_CALLING_STRATEGY_MULTIPLE:
                contents = []
                for _ in range(n):
                    start = time.time()
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages if messages else [{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=1,
                        top_p=top_p,
                        stop=stop,
                    )
                    latency = time.time() - start
                    if cost_recorder is not None and hasattr(response, "usage"):
                        cost_recorder.update_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
                    contents.append(response.choices[0].message.content)
                entry = {
                    "ts": datetime.now().isoformat(),
                    "mode": "sdk",
                    "model": model,
                    "strategy": n_strategy,
                    "n": n,
                    "latency": latency,
                    "prompt": prompt if full_log else (prompt),
                    "outputs": contents if full_log else [o for o in contents]
                }
                try:
                    with open(dialogues_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass
                logger.info(f"LLM SDK multi success model={model} latency_last={latency:.2f}s total={len(contents)}")
                break
            else:
                raise ValueError(f"Invalid n_strategy: {n_strategy} for n: {n}")
        except Exception as e:
            logger.warning(f"LLM SDK error attempt={retrying+1}: {e}")
            retrying += 1
            if retrying == MAX_RETRYING_TIMES:
                contents = [""]
                break
            time.sleep(10)
    return contents

