#!/usr/bin/env python3
"""LLM-assisted classifier for mini_dev_mysql_alpaca_filtered.json samples.

The script reuses the project's existing LLM access pattern (`LLMClient`) so it
honors the same environment variables (OpenAI-compatible endpoints, GPTBEST
proxy, etc.). Each record from the target JSON dataset is enriched with a
category label drawn from the business taxonomy provided by the user.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..augment_decomposition import LLMClient, safe_json_loads, parse_cn_prompt_blocks

EXAMPLE_CATEGORIES: Dict[str, str] = {
    "留存类": "Retention metrics (次留、7留、回流、新增留存等) and churn analysis.",
    "行为链条": "Multi-step funnels or behavior journeys such as 点击→加入→确认 chains.",
    "用户分群": "User segmentation, cohort splits, or profiling across dimensions.",
    "活跃分布": "Activity/engagement distribution over time, device, mode, or other axes.",
    "活动分析": "Event/campaign performance reviews and general operational analysis.",
}

CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "留存类": ("留存", "回流", "次留", "7留", "留住", "复访"),
    "行为链条": ("链条", "流程", "步骤", "转化", "路径", "点击", "确认", "新增→"),
    "用户分群": ("分群", "分层", "分组", "画像", "类型", "群体"),
    "活跃分布": ("活跃", "参与人数", "分布", "日活", "周活", "活跃天数"),
    "活动分析": ("活动", "玩法", "上线", "运营", "分析", "参与情况"),
}

PROMPT_TEMPLATE = """你是资深数据分析师，需要阅读给定的 SQL 需求并判断它的分析主题。

示例分类（可扩展）：
{category_guide}

分类准则：
1. 重点关注问题本身想回答的商业问题，而不是 SQL 实现细节。
2. 如果问题同时符合多个类别，优先选择排名更靠前且覆盖面更窄的那一个：留存类 > 行为链条 > 用户分群 > 活跃分布 > 活动分析。
3. 当示例类别均不适用时，可自拟更贴切的标签；输出 JSON 时需给出 category（中文短语）和 rationale（20~40 字中文说明）。

待判定的样本：
问题 ID: {question_id}
Instruction: {instruction}
用户需求概述:
{user_question}

数据库/参考摘要:
{schema_text}

候选 SQL:
{sql_snippet}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify mini-dev samples with an LLM")
    parser.add_argument("--input", default="mini_dev_mysql_alpaca_filtered.json", help="Path to source dataset")
    parser.add_argument("--output", default="mini_dev_mysql_alpaca_classified.jsonl", help="Where to write enriched records (JSONL)")
    parser.add_argument("--dataset-output", default=None, help="File to rewrite with category labels (defaults to --input)")
    parser.add_argument("--model", default="Qwen3-Coder-30B-A3B-Instruct", help="LLM model name passed to LLMClient")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature for the LLM call")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap on processed records")
    parser.add_argument("--start-from", type=int, default=0, help="Zero-based offset into the dataset")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional delay (seconds) between LLM calls")
    parser.add_argument("--resume", action="store_true", help="Append to output if it already exists (skip known IDs)")
    parser.add_argument("--dry-run", action="store_true", help="Skip network calls and rely on keyword heuristics")
    parser.add_argument("--offline", action="store_true", help="Instantiate LLMClient in offline mode to force heuristics")
    parser.add_argument("--no-write-back", action="store_true", help="Do not persist category labels back to the dataset JSON")
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data)!r}")
    return data


def write_dataset(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
        f.write("\n")


def truncate(text: str, limit: int = 1600) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def category_guide_text() -> str:
    return "\n".join(
        f"- {name}: {desc}" for name, desc in EXAMPLE_CATEGORIES.items()
    )


def build_messages(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    blocks = parse_cn_prompt_blocks(sample.get("input", ""))
    prompt = PROMPT_TEMPLATE.format(
        category_guide=category_guide_text(),
        question_id=sample.get("question_id", "unknown"),
        instruction=truncate(sample.get("instruction", ""), 400),
        user_question=truncate(blocks.get("user_question") or sample.get("input", ""), 800),
        schema_text=truncate(blocks.get("schema_text"), 600),
        sql_snippet=truncate(sample.get("output", ""), 800),
    )
    return [
        {
            "role": "system",
            "content": "你是擅长手游商业分析的资深分析师，只返回 JSON。",
        },
        {"role": "user", "content": prompt},
    ]


def heuristic_category(source_text: str) -> Tuple[str, str]:
    lowered = (source_text or "").lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category, f"关键词命中: {category}"
    return "活动分析", "默认回退到活动分析"


def normalize_category(candidate: Optional[str]) -> Optional[str]:
    if candidate:
        cleaned = candidate.strip()
        if cleaned:
            return cleaned
    return None


def classify_sample(
    sample: Dict[str, Any],
    llm: Optional[LLMClient],
    temperature: float,
    dry_run: bool,
) -> Dict[str, Any]:
    full_text = "\n".join(
        filter(
            None,
            [sample.get("instruction", ""), sample.get("input", ""), sample.get("output", "")],
        )
    )
    raw_response: Optional[str] = None
    category_source = "heuristic"
    rationale = ""

    if not dry_run and llm is not None:
        try:
            messages = build_messages(sample)
            raw_response = llm.invoke(messages=messages, temperature=temperature, max_tokens=320)
            parsed = safe_json_loads(raw_response)
            candidate = parsed.get("category") if parsed else None
            rationale = (parsed.get("rationale") or parsed.get("reason")) if parsed else ""
            normalized = normalize_category(candidate)
            if normalized:
                category = normalized
                category_source = "LLM"
                rationale = rationale or "LLM 直接给出分类"
            else:
                category, fallback_reason = heuristic_category(full_text)
                category_source = "heuristic"
                rationale = rationale or fallback_reason
        except Exception as exc:
            raw_response = f"__error__: {exc}"
            category, fallback_reason = heuristic_category(full_text)
            category_source = "heuristic"
            rationale = fallback_reason or "LLM 调用失败，使用关键词回退"
    else:
        category, fallback_reason = heuristic_category(full_text)
        category_source = "heuristic"
        rationale = fallback_reason or "Dry-run/离线模式下的关键词分类"

    return {
        **sample,
        "category": category,
        "category_reason": rationale,
        "category_source": category_source,
        "llm_raw": raw_response,
    }


def load_existing(output_path: Path) -> Dict[str, Dict[str, Any]]:
    if not output_path.exists():
        return {}
    cache: Dict[str, Dict[str, Any]] = {}
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and "question_id" in record:
                cache[record["question_id"]] = record
    return cache


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    dataset = load_dataset(input_path)
    dataset_output_path = Path(args.dataset_output) if args.dataset_output else input_path
    resume_map = load_existing(output_path) if args.resume else {}

    llm_client: Optional[LLMClient]
    if args.dry_run or args.offline:
        llm_client = None
    else:
        llm_client = LLMClient(model=args.model, default_temperature=args.temperature, offline=args.offline)

    start = args.start_from
    end = len(dataset) if args.max_records is None else min(len(dataset), start + args.max_records)
    subset = dataset[start:end]

    if not subset:
        print("No records to process after applying start/max constraints", file=sys.stderr)
        return

    mode = "a" if args.resume and output_path.exists() else "w"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with output_path.open(mode, encoding="utf-8") as writer:
        for idx, sample in enumerate(subset, start=start):
            sample_id = sample.get("question_id")
            if sample_id and sample_id in resume_map:
                cached = resume_map[sample_id]
                sample.update({
                    key: cached.get(key)
                    for key in ("category", "category_reason", "category_source", "llm_raw")
                    if key in cached
                })
                continue
            enriched = classify_sample(sample, llm_client, args.temperature, args.dry_run or args.offline)
            writer.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            writer.flush()
            sample.update({
                "category": enriched.get("category"),
                "category_reason": enriched.get("category_reason"),
                "category_source": enriched.get("category_source"),
                "llm_raw": enriched.get("llm_raw"),
            })
            processed += 1
            print(f"[{processed}] {sample_id or idx} -> {enriched['category']} ({enriched['category_source']})")
            if args.sleep:
                time.sleep(args.sleep)

    if not args.no_write_back:
        dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
        write_dataset(dataset_output_path, dataset)
        print(f"Dataset updated with categories at {dataset_output_path}")
    else:
        print("Dataset write-back skipped (--no-write-back)")

    print(f"Finished {processed} new classifications; output at {output_path}")


if __name__ == "__main__":
    main()
