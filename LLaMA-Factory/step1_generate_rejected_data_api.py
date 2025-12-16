#!/usr/bin/env python3
"""Generate DPO preference data for BIRD text2sql using SiliconFlow Chat Completions API.

This script is API 版本的 `step1_generate_rejected_data.py`：
- 保持 argparse 的参数名和默认值与原脚本一致，方便复用命令行参数。
- 不再本地加载大模型，而是调用 SiliconFlow Chat Completions 接口生成 rejected SQL。

输入：
  - Alpaca SFT 数据集：同原脚本（默认为 /root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets_283.json）。

输出：
  - DPO 数据：同原脚本（默认为 /root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets_283_sample.json），
    格式为 JSON 数组，每个元素形如：
    {
      "instruction": "...",
      "input": "...",
      "chosen": "gold SQL (output)",
      "rejected": "model prediction SQL"
    }

注意：
  - API 端点固定为 https://api.siliconflow.cn/v1/chat/completions
  - API Key 通过环境变量 SILICONFLOW_API_KEY（或 SILICONFLOW_TOKEN）提供。
  - 模型名称可通过环境变量 SILICONFLOW_MODEL 覆盖，未设置时默认使用
    "Qwen/Qwen3-Coder-30B-A3B-Instruct"。
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from tqdm import tqdm


API_URL = "https://api.siliconflow.cn/v1/chat/completions"


def load_alpaca(path: Path, max_samples: int = 0) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expect a list in {path}, got {type(data)}")
    if max_samples and max_samples > 0:
        data = data[: max_samples]
    return data


def build_prompt(instruction: str, input_text: str) -> str:
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    if instruction and input_text:
        return f"{instruction}\n{input_text}"
    if instruction:
        return instruction
    return input_text


def generate_sql_via_api(
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
    api_key: str,
    model_name: str,
    max_retries: int = 3,
) -> List[str]:
    """调用 SiliconFlow Chat Completions API 生成 SQL。

    返回长度为 num_return_sequences 的字符串列表。
    """

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    body: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        # OpenAI 风格参数
        "max_tokens": max_new_tokens,
        "temperature": max(0.0, float(temperature)),
        "top_p": float(top_p),
        "n": max(1, int(num_return_sequences)),
        "stream": False,
    }

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=body, timeout=600)
        except requests.RequestException as e:  # 网络类错误重试
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"SiliconFlow API network error: {e}") from e

        if resp.status_code == 200:
            break

        # 5xx / 429 之类服务端错误，适当重试
        if resp.status_code in {429} or 500 <= resp.status_code < 600:
            last_err = RuntimeError(
                f"SiliconFlow API error: HTTP {resp.status_code} - {resp.text}"
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
        # 其他错误不重试
        raise RuntimeError(
            f"SiliconFlow API error: HTTP {resp.status_code} - {resp.text}"
        )

    data = resp.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError(f"SiliconFlow API returned no choices: {data}")

    results: List[str] = []
    for choice in choices[: max(1, num_return_sequences)]:
        message = choice.get("message") or {}
        content = message.get("content", "")
        if not isinstance(content, str):
            # 兼容 content 可能为 list 结构的情况
            if isinstance(content, list):
                content = "".join(part.get("text", "") for part in content)
            else:
                content = str(content)
        results.append(content.strip())

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BIRD DPO data from SFT dataset and model predictions via SiliconFlow API.",
    )
    # 为保持兼容，parser 的参数与 step1_generate_rejected_data.py 保持一致
    parser.add_argument(
        "--base-model",
        type=str,
        default="/root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct",
        help="(Unused in API mode) Path to base model; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="",
        help="(Unused in API mode) LoRA adapter path; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--sft-data",
        type=str,
        default="/root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets_283.json",
        help="Path to Alpaca SFT dataset used for BIRD text2sql.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets_283_sample.json",
        help="Where to write DPO JSON array.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of samples (0 means use all).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=65536,
        help="Max new tokens when generating SQL.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for generation (0 for greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.6,
        help="Top-p for nucleus sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="(Unused in API mode) Device to run generation on; kept for CLI compatibility.",
    )
    parser.add_argument(
        "--num-rejects",
        type=int,
        default=6,
        help="How many rejected SQL candidates to generate per sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for generation (may be ignored by API).",
    )
    parser.add_argument(
        "--dedup-rejects",
        action="store_true",
        help=(
            "If set, skip writing duplicate rejected SQL candidates "
            "within the same sample."
        ),
    )

    args = parser.parse_args()

    # 读取 API Key 和模型名
    api_key = os.getenv("SILICONFLOW_API_KEY","sk-myohjpmdhpmhiplkytqqgpafgifvlaflysacsvofrzwpznfa")
    
    if not api_key:
        raise RuntimeError(
            "Missing SiliconFlow API key. Please set SILICONFLOW_API_KEY or SILICONFLOW_TOKEN."
        )

    # 默认使用与本地一致的模型名，用户可通过环境变量覆盖
    model_name = os.getenv("SILICONFLOW_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

    sft_path = Path(args.sft_data)
    data = load_alpaca(sft_path, args.max_samples)
    print(f"Loaded {len(data)} SFT samples from {sft_path}.")
    print(f"Using SiliconFlow model: {model_name}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 读取已有输出，实现断点续跑 + 追加
    existing_records: List[Dict[str, Any]] = []
    processed_keys = set()
    if out_path.exists():
        raw = out_path.read_text(encoding="utf-8").strip()
        if raw:
            try:
                # 正常完整 JSON 数组
                existing_records = json.loads(raw)
            except json.JSONDecodeError:
                # 可能是上次中断导致缺少结尾 "]"，尽量修复
                if raw.startswith("["):
                    inner = raw[1:]
                    last_brace = inner.rfind("}")
                    if last_brace != -1:
                        repaired = "[" + inner[: last_brace + 1] + "]"
                        try:
                            existing_records = json.loads(repaired)
                        except json.JSONDecodeError:
                            print(
                                f"[WARN] Failed to repair existing output {out_path}, "
                                "starting from scratch."
                            )
                    else:
                        print(
                            f"[WARN] Existing output {out_path} has no complete records, "
                            "starting from scratch."
                        )
                else:
                    print(
                        f"[WARN] Existing output {out_path} is not a JSON array, "
                        "starting from scratch."
                    )

    if existing_records:
        print(f"Found {len(existing_records)} existing DPO records in {out_path}, will append new ones.")
        for rec in existing_records:
            key = (rec.get("instruction", ""), rec.get("input", ""))
            processed_keys.add(key)

    num_written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write("[\n")
        first_record = True

        # 先把已有记录写回去
        for rec in existing_records:
            if not first_record:
                fout.write(",\n")
            fout.write(json.dumps(rec, ensure_ascii=False, indent=2))
            first_record = False
            num_written += 1

        total = len(data)

        for idx, sample in tqdm(enumerate(data, start=1), total=total):
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            gold_sql = (sample.get("output", "") or "").strip()

            if not gold_sql:
                continue

            key = (instruction, input_text)
            if key in processed_keys:
                # 这一条样本已经在旧文件中生成过，跳过
                continue

            prompt = build_prompt(instruction, input_text)
            seen_rejects = set()
            num_to_generate = max(1, args.num_rejects)

            try:
                preds = generate_sql_via_api(
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_return_sequences=num_to_generate,
                    api_key=api_key,
                    model_name=model_name,
                )[:num_to_generate]
            except Exception as e:
                print(f"[WARN] Failed to generate for sample {idx}: {e}")
                continue

            for ridx, pred_sql in tqdm(
                enumerate(preds), total=len(preds), leave=False
            ):
                if args.dedup_rejects:
                    norm_pred = pred_sql.strip()
                    if norm_pred in seen_rejects:
                        continue
                    seen_rejects.add(norm_pred)

                record = {
                    "instruction": instruction,
                    "input": input_text,
                    "chosen": gold_sql,
                    "rejected": pred_sql,
                }

                if not first_record:
                    fout.write(",\n")
                fout.write(json.dumps(record, ensure_ascii=False, indent=2))
                fout.flush()
                first_record = False
                num_written += 1

            processed_keys.add(key)

            if idx % 10 == 0:
                print(
                    f"Processed {idx}/{len(data)} samples, "
                    f"written {num_written} preference pairs."
                )

        fout.write("\n]\n")
        fout.flush()

    print(f"Done. Wrote {num_written} DPO samples to {out_path}.")


if __name__ == "__main__":
    main()
