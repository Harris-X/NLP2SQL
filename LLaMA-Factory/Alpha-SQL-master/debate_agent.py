#!/usr/bin/env python3
"""Three-model debate agent for SQL verification and regeneration.

- Loads prompts from final_dataset_processed.json and paired .sql answers.
- Model A highlights correct portions of the SQL; Model B highlights incorrect parts.
- Both models write to a shared history, then re-read the history for a second pass.
- Model C reviews the shared history to decide whether to regenerate SQL; if yes, it proposes a new SQL and the debate loops.
- Stops when Model C decides regeneration is unnecessary or when max rounds are reached.

This script reuses the existing LLMClient utilities from augment_decomposition.py to honor the same
endpoint/env configuration. SQL execution can be optionally enabled; by default it is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import copy

from augment_decomposition import LLMClient, safe_json_loads

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATASET = BASE_DIR / "competion_dataset" / "final_dataset_processed.json"
SQL_DIR = BASE_DIR / "1209_wrong" / "sql_submission1208"
OUTPUT_JSONL = BASE_DIR / "competion_dataset" / "debate_results.jsonl"

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

PROMPT_SKELETON = (
    "该问题需求如下：{instruction}\n"
    "{input_text}\n"
    "问题分类中：由于[{category_reason}],\n"
    "该问题属于[{category}],\n"
    "该问题的求解答案为：\n{sql_text}\n"
)

EVAL_PROMPT = (
    "你是严格的 SQL 评审员，需基于上下文逐条指出答案的正确和错误点。\n"
    "【问题上下文】\n{prompt}\n"
    "【历史记录（含上一轮判定与对话）】\n{history}\n"
    "【输出格式（仅 JSON）】\n"
    "{{\n"
    "  \"correct_parts\": [ {{\"segment\": \"...\", \"reason\": \"...\"}}, ... ],\n"
    "  \"incorrect_parts\": [ {{\"segment\": \"...\", \"reason\": \"...\"}}, ... ],\n"
    "  \"reason\": \"总体评价，20~60 字\"\n"
    "}}\n"
    "要求：每个 segment 引用SQL片段或规则点，reason 说明原因；如信息不足，请写明。"
)

DECISION_PROMPT = (
    "你是裁判模型，需判断是否要重新生成 SQL 并给出修正版。\n"
    "【问题上下文】\n{prompt}\n"
    "【历史记录（含所有评审对话）】\n{history}\n"
    "【输出格式（仅 JSON）】\n"
    "{{\n"
    "  \"regenerate\": true/false,\n"
    "  \"reason\": \"是否重生成的依据，20~60 字\",\n"
    "  \"fixed_sql\": \"当 regenerate=true 时给出新的 SQL，需 MySQL 语法且仅查询；否则可空字符串\"\n"
    "}}\n"
    "若重生成，请确保吸收历史反馈并修正错误；如无法改进，保持 regenerate=false。"
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def read_sql_text(sql_dir: Path, question_id: str) -> str:
    sql_path = sql_dir / f"{question_id}.sql"
    if sql_path.exists():
        return sql_path.read_text(encoding="utf-8").strip()
    return ""


def build_prompt(record: Dict[str, Any], sql_text: str) -> str:
    return PROMPT_SKELETON.format(
        instruction=record.get("instruction", "").strip(),
        input_text=record.get("input", "").strip(),
        category_reason=record.get("category_reason", "未知原因"),
        category=record.get("category", "未知分类"),
        sql_text=sql_text or "(无SQL文本)",
    )


def history_to_text(history: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for idx, entry in enumerate(history, 1):
        lines.append(
            f"[{idx}] model={entry.get('model')} verdict={entry.get('verdict')} "
            f"correct={entry.get('correct_parts')} incorrect={entry.get('incorrect_parts')} reason={entry.get('reason')}"
        )
    return "\n".join(lines) or "(empty)"


def dialogues_to_text(dialogues: List[Dict[str, Any]]) -> str:
    """Render the full dialogue transcript (no truncation) so models receive complete context.

    Returns a plain-text concatenation of all prompt/response pairs and parsed summaries.
    """
    lines: List[str] = []
    for idx, dlg in enumerate(dialogues, 1):
        prompt = (dlg.get("prompt") or "").strip()
        resp = (dlg.get("response_raw") or "").strip()
        lines.append(
            f"[{idx}] model={dlg.get('model')} verdict={dlg.get('verdict')}\n"
            f"PROMPT:\n{prompt}\n\nRESPONSE:\n{resp}\n\nPARSED:\n{json.dumps(dlg.get('parsed', {}), ensure_ascii=False)}\n"
        )
    return "\n".join(lines) or "(empty)"


def parse_json_response(text: str) -> Dict[str, Any]:
    data = safe_json_loads(text) or {}
    if not isinstance(data, dict):
        return {}
    return data


def apply_endpoint_overrides(base_url: str, api_key: str) -> None:
    """Propagate endpoint overrides to environment for LLMClient reuse."""
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


def call_judge(model: LLMClient, prompt: str, dialogues: List[Dict[str, Any]], temperature: float) -> Dict[str, Any]:
    # render full dialogue history for model input
    history_text = dialogues_to_text(dialogues)
    rendered = EVAL_PROMPT.format(prompt=prompt, history=history_text)
    if getattr(model, "offline", False):
        resp = ""
        parsed = {}
    else:
        resp = model.invoke(prompt=rendered, temperature=temperature, max_tokens=512)
        parsed = parse_json_response(resp) or {}
    return {
        "prompt": rendered,
        "history_text": history_text,
        "history_struct": copy.deepcopy(dialogues),
        "raw": resp,
        "parsed": parsed,
        "correct_parts": parsed.get("correct_parts") or [],
        "incorrect_parts": parsed.get("incorrect_parts") or [],
        "reason": parsed.get("reason") or "",
    }


def call_decider(model: LLMClient, prompt: str, dialogues: List[Dict[str, Any]], temperature: float) -> Dict[str, Any]:
    history_text = dialogues_to_text(dialogues)
    rendered = DECISION_PROMPT.format(prompt=prompt, history=history_text)
    if getattr(model, "offline", False):
        resp = ""
        parsed = {}
    else:
        resp = model.invoke(prompt=rendered, temperature=temperature, max_tokens=512)
        parsed = parse_json_response(resp) or {}
    return {
        "prompt": rendered,
        "history_text": history_text,
        "history_struct": copy.deepcopy(dialogues),
        "raw": resp,
        "parsed": parsed,
        "regenerate": bool(parsed.get("regenerate")),
        "reason": parsed.get("reason") or "",
        "fixed_sql": parsed.get("fixed_sql") or "",
    }


# ---------------------------------------------------------------------------
# Debate loop
# ---------------------------------------------------------------------------


def run_debate(
    record: Dict[str, Any],
    model_a: LLMClient,
    model_b: LLMClient,
    model_c: LLMClient,
    sql_dir: Path,
    max_rounds: int,
    temperature: float,
) -> Dict[str, Any]:
    question_id = record.get("question_id") or record.get("sql_id") or "unknown"
    current_sql = read_sql_text(sql_dir, question_id)
    prompt_text = build_prompt(record, current_sql)
    history: List[Dict[str, Any]] = []
    dialogues: List[Dict[str, Any]] = []

    for round_idx in range(max_rounds):
        # include both structured summary and full dialogue transcript
        hist_text = dialogues_to_text(dialogues)

        # First-pass judgments (pass full dialogues list so model receives all prior turns)
        judge_a = call_judge(model_a, prompt_text, dialogues, temperature)
        history.append({
            "model": "A",
            "verdict": "first_pass",
            "correct_parts": judge_a["correct_parts"],
            "incorrect_parts": judge_a["incorrect_parts"],
            "reason": judge_a["reason"],
            "round": round_idx,
        })
        dialogues.append({
            "model": "A",
            "verdict": "first_pass",
            "prompt": judge_a.get("prompt"),
            "history_snapshot": judge_a.get("history_snapshot"),
            "response_raw": judge_a.get("raw"),
            "parsed": {
                "correct_parts": judge_a.get("correct_parts"),
                "incorrect_parts": judge_a.get("incorrect_parts"),
                "reason": judge_a.get("reason"),
            },
        })

        judge_b = call_judge(model_b, prompt_text, dialogues, temperature)
        history.append({
            "model": "B",
            "verdict": "first_pass",
            "correct_parts": judge_b["correct_parts"],
            "incorrect_parts": judge_b["incorrect_parts"],
            "reason": judge_b["reason"],
        })
        dialogues.append({
            "model": "B",
            "verdict": "first_pass",
            "prompt": judge_b.get("prompt"),
            "history_snapshot": judge_b.get("history_snapshot"),
            "response_raw": judge_b.get("raw"),
            "parsed": {
                "correct_parts": judge_b.get("correct_parts"),
                "incorrect_parts": judge_b.get("incorrect_parts"),
                "reason": judge_b.get("reason"),
            },
        })

        # Second-pass with history
        # Second pass: include all accumulated dialogues
        judge_a_2 = call_judge(model_a, prompt_text, dialogues, temperature)
        history.append({
            "model": "A",
            "verdict": "second_pass",
            "correct_parts": judge_a_2["correct_parts"],
            "incorrect_parts": judge_a_2["incorrect_parts"],
            "reason": judge_a_2["reason"],
            "round": round_idx,
        })
        dialogues.append({
            "model": "A",
            "verdict": "second_pass",
            "prompt": judge_a_2.get("prompt"),
            "history_snapshot": judge_a_2.get("history_snapshot"),
            "response_raw": judge_a_2.get("raw"),
            "parsed": {
                "correct_parts": judge_a_2.get("correct_parts"),
                "incorrect_parts": judge_a_2.get("incorrect_parts"),
                "reason": judge_a_2.get("reason"),
            },
        })

        judge_b_2 = call_judge(model_b, prompt_text, dialogues, temperature)
        history.append({
            "model": "B",
            "verdict": "second_pass",
            "correct_parts": judge_b_2["correct_parts"],
            "incorrect_parts": judge_b_2["incorrect_parts"],
            "reason": judge_b_2["reason"],
        })
        dialogues.append({
            "model": "B",
            "verdict": "second_pass",
            "prompt": judge_b_2.get("prompt"),
            "history_snapshot": judge_b_2.get("history_snapshot"),
            "response_raw": judge_b_2.get("raw"),
            "parsed": {
                "correct_parts": judge_b_2.get("correct_parts"),
                "incorrect_parts": judge_b_2.get("incorrect_parts"),
                "reason": judge_b_2.get("reason"),
            },
        })

        # Decision and potential regeneration
        # Decision: pass full dialogues
        decision = call_decider(model_c, prompt_text, dialogues, temperature)
        history.append({
            "model": "C",
            "verdict": "decision",
            "regenerate": decision["regenerate"],
            "reason": decision["reason"],
            "fixed_sql": decision["fixed_sql"],
            "round": round_idx,
        })
        dialogues.append({
            "model": "C",
            "verdict": "decision",
            "prompt": decision.get("prompt"),
            "history_snapshot": decision.get("history_snapshot"),
            "response_raw": decision.get("raw"),
            "parsed": {
                "regenerate": decision.get("regenerate"),
                "reason": decision.get("reason"),
                "fixed_sql": decision.get("fixed_sql"),
            },
        })

        if not decision["regenerate"]:
            break

        if decision["fixed_sql"]:
            current_sql = decision["fixed_sql"].strip()
            prompt_text = build_prompt(record, current_sql)
        else:
            # If no SQL provided, stop to avoid infinite loop.
            break

    return {
        "question_id": question_id,
        "final_sql": current_sql,
        "history": history,
        "dialogues": dialogues,
        "prompt": prompt_text,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_llm_client(model_name: str, temperature: float, offline: bool) -> LLMClient:
    return LLMClient(model=model_name, default_temperature=temperature, offline=offline)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-model SQL debate agent")
    parser.add_argument("--input", default=str(PROCESSED_DATASET), help="Path to processed dataset JSON")
    parser.add_argument("--sql-dir", default=str(SQL_DIR), help="Directory containing reference .sql files")
    parser.add_argument("--output", default=str(OUTPUT_JSONL), help="Where to write debate results JSONL")
    parser.add_argument("--sql-output-dir", default=str(BASE_DIR / "competion_dataset" / "debate_sql"), help="Where to write regenerated SQL files")
    parser.add_argument("--model-a", default="qwen3-coder-480b-a35b-instruct")
    parser.add_argument("--model-b", default="gpt-5.1-thinking")
    parser.add_argument("--model-c", default="claude-opus-4-5-20251101-thinking")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=-1, help="Optional cap on records processed")
    parser.add_argument("--start", type=int, default=0, help="Zero-based start index")
    parser.add_argument("--offline", action="store_true", help="Use offline heuristic mode for all models")
    parser.add_argument("--openai_base_url", default="https://api.gptbest.vip/v1")
    parser.add_argument("--openai_api_key", default="sk-GMYNUCidV96DStXskUpPqgemoaDur0alDXZkeyiq5E3mXGZn")
    args = parser.parse_args()

    dataset = load_dataset(Path(args.input))
    sql_dir = Path(args.sql_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sql_output_dir = Path(args.sql_output_dir)
    sql_output_dir.mkdir(parents=True, exist_ok=True)

    # propagate endpoint overrides for LLMClient (align with augment_decomposition)
    apply_endpoint_overrides(args.openai_base_url, args.openai_api_key)

    subset = dataset[args.start : (len(dataset) if args.limit == -1 else args.start + args.limit)]

    model_a = build_llm_client(args.model_a, args.temperature, args.offline)
    model_b = build_llm_client(args.model_b, args.temperature, args.offline)
    model_c = build_llm_client(args.model_c, args.temperature, args.offline)

    processed = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for record in subset:
            result = run_debate(
                record=record,
                model_a=model_a,
                model_b=model_b,
                model_c=model_c,
                sql_dir=sql_dir,
                max_rounds=args.max_rounds,
                temperature=args.temperature,
            )
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            # persist final SQL as .sql file per question
            qid = result.get("question_id") or f"idx_{processed}"
            if result.get("final_sql"):
                (sql_output_dir / f"{qid}.sql").write_text(result["final_sql"], encoding="utf-8")
            processed += 1
            print(f"[{processed}] {result['question_id']} processed")

    print(f"Done. Wrote {processed} results to {output_path}")


if __name__ == "__main__":
    main()
