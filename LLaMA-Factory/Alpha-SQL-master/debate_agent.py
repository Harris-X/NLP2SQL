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
from typing import Any, Dict, List, Optional, Tuple, Literal, TextIO
import copy

from augment_decomposition import LLMClient, SQLExecutionTool, safe_json_loads

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

EVAL_SYSTEM_PROMPT_A = (
    "你是评审员A，主要负责【业务核心理解】与【需求→数据→口径】的正确性审查。"
    "你要基于【数据库schema】逐表核对：每张表的业务含义、关键字段含义、与问题需求的对应关系。"
    "硬性要求：评审时必须检查 SQL 是否覆盖并使用【数据库schema】中所有表；缺表或冗表都算问题。"
    "如果 schema 中存在同名/相似键字段（如 id、user_id、phone、acct 等），需判断其业务含义是否一致，"
    "若不一致则指出可能的口径/关联错误。"
    "\n\n"
    "【输出格式（仅 JSON）】\n"
    "{\n"
    "  \"incorrect_parts\": [ {\"segment\": \"...\", \"reason\": \"...\"}, ... ],\n"
    "  \"reason\": \"必须以'A(业务):'开头；从业务口径/指标定义/数据链路角度总结，20~80字；不得复述其他评审的句子；信息不足则说明缺口\",\n"
    "  \"understanding_notes\": [\"3~6条要点：你对题目/业务口径/数据链路的可核验理解（不要逐步推理）\"],\n"
    "  \"question_intent_guess\": \"若题意/信息不明，推测题目真实含义（1~2句）\",\n"
    "  \"schema_gap_hypotheses\": [ {\"missing\": \"可能缺失的表/字段/约束\", \"likely_semantics\": \"推测含义\", \"why_needed\": \"为什么需要\", \"related_existing_fields\": [\"table.field\"]} ],\n"
    "  \"missing_question_info_to_fields\": [ {\"missing_info\": \"题目缺失信息点\", \"likely_field\": \"table.field\", \"why\": \"为何相关\"} ],\n"
    "  \"unused_table_backfill\": [ {\"table\": \"schema里但SQL未用的表名\", \"why_schema_includes\": \"该表在业务中的角色\", \"if_should_be_used_then_background\": \"若必须使用它，题目更完整的背景应包含什么\", \"how_to_join_guess\": \"可能的关联键/关联方向（基于schema推断）\"} ],\n"
    "  \"next_round_advice\": \"用于下一轮完善题意/表理解/SQL修正的建议（2~5条，合并成一句话也可）\"\n"
    "}\n"
    "要求：不仅挑问题，还要给出可核验的理解要点与合理推断；不要输出逐步推理过程。"
    "你必须优先讨论【业务口径/指标定义/数据链路】而不是SQL语法。"
)

EVAL_SYSTEM_PROMPT_B = (
    "你是评审员B，主要负责【SQL中的业务理解与关联逻辑】审查（join/where/group/聚合口径/去重/时间窗口）。"
    "你要基于【数据库schema】逐表核对：SQL 使用的表与字段是否匹配其业务含义；表间关联键是否合理。"
    "硬性要求：评审时必须检查 SQL 是否覆盖并使用【数据库schema】中所有表；缺表或冗表都算问题。"
    "特别检查：不同表里的同名/相似键字段是否表达同一含义（例如同为 phone 但口径不同），"
    "若含义不一致或需转换映射却未做，请明确指出。"
    "\n\n"
    "【输出格式（仅 JSON）】\n"
    "{\n"
    "  \"incorrect_parts\": [ {\"segment\": \"...\", \"reason\": \"...\"}, ... ],\n"
    "  \"reason\": \"必须以'B(SQL):'开头；从JOIN/WHERE/GROUP/聚合口径/去重/时间窗口角度总结，20~80字；不得复述其他评审的句子；信息不足则说明缺口\",\n"
    "  \"understanding_notes\": [\"3~6条要点：你对SQL逻辑与业务语义匹配关系的可核验理解（不要逐步推理）\"],\n"
    "  \"question_intent_guess\": \"若题意/信息不明，推测题目真实含义（1~2句）\",\n"
    "  \"schema_gap_hypotheses\": [ {\"missing\": \"可能缺失的表/字段/约束\", \"likely_semantics\": \"推测含义\", \"why_needed\": \"为什么需要\", \"related_existing_fields\": [\"table.field\"]} ],\n"
    "  \"missing_question_info_to_fields\": [ {\"missing_info\": \"题目缺失信息点\", \"likely_field\": \"table.field\", \"why\": \"为何相关\"} ],\n"
    "  \"unused_table_backfill\": [ {\"table\": \"schema里但SQL未用的表名\", \"why_schema_includes\": \"该表在业务中的角色\", \"if_should_be_used_then_background\": \"若必须使用它，题目更完整的背景应包含什么\", \"how_to_join_guess\": \"可能的关联键/关联方向（基于schema推断）\"} ],\n"
    "  \"next_round_advice\": \"用于下一轮完善题意/表理解/SQL修正的建议（2~5条，合并成一句话也可）\"\n"
    "}\n"
    "要求：不仅挑问题，还要给出可核验的理解要点与合理推断；不要输出逐步推理过程。"
    "你必须优先讨论【SQL语义/关联逻辑/聚合与去重】而不是业务口径。"
)

DECISION_SYSTEM_PROMPT = (
    "你是裁判模型，以问题业务的理解为第一要素，判断是否要重新生成 SQL 并给出修正版。注意：只要评审员检查出问题，就必须重新理解业务，对于业务中子表的缺失部分必须要进行合理的推断，同时一步步思考来生成修正的sql。\n\n"
    "【输出格式（仅 JSON）】\n"
    "{\n"
    "  \"regenerate\": true/false,\n"
    "  \"reason\": \"是否重生成的依据，20~60 字\",\n"
    "  \"fixed_sql\": \"当 regenerate=true 时给出新的 SQL，需 MySQL 语法且仅查询；否则可空字符串\"\n"
    "}\n"
    "若重生成，请确保吸收历史反馈并修正错误；如无法改进，保持 regenerate=false。"
)


SQL_FIX_SYSTEM_PROMPT = (
    "你是资深 MySQL 工程师与调试助手。你的任务是修复给定 SQL 的语法/执行错误，使其能在 MySQL 上成功执行。\n"
    "约束：只允许只读查询（SELECT/CTE），禁止 INSERT/UPDATE/DELETE/DDL/事务语句；尽量少改动；不要编造不存在的表/字段。\n\n"
    "【输出格式（仅 JSON）】\n"
    "{\n"
    "  \"fixed_sql\": \"...\",\n"
    "  \"reason\": \"20~60 字，说明修复点\"\n"
    "}\n"
)


HistoryMode = Literal["compact", "full", "none"]


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    # keep the most recent tail; prefix marker to signal truncation
    return "(truncated) ...\n" + text[-max_chars:]


def _short(text: str, max_len: int = 160) -> str:
    s = (text or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max(0, max_len - 3)] + "..."


def _flush_writer(writer: TextIO) -> None:
    """Flush and best-effort fsync to ensure real-time persistence."""
    writer.flush()
    try:
        os.fsync(writer.fileno())
    except Exception:
        # Some streams/filesystems may not support fsync; flush is still useful.
        pass


def _norm_reason(text: str) -> str:
    return " ".join((text or "").strip().split())


def _is_same_reason(a: str, b: str) -> bool:
    na, nb = _norm_reason(a), _norm_reason(b)
    return bool(na) and na == nb


def _is_same_incorrect_parts(a: Any, b: Any) -> bool:
    if not isinstance(a, list) or not isinstance(b, list):
        return False
    try:
        return json.dumps(a, ensure_ascii=False, sort_keys=True) == json.dumps(b, ensure_ascii=False, sort_keys=True)
    except Exception:
        return False


def render_compact_history(
    dialogues: List[Dict[str, Any]],
    max_turns: int,
    max_chars: int,
) -> str:
    """Compact, de-duplicated history for LLM consumption.

    Rules:
    - Do NOT include full prior prompts or raw responses (too long and repetitive).
    - Keep only parsed summaries and decisions.
    - Prefer most recent turns.
    """
    if max_turns <= 0 or max_chars <= 0:
        return ""

    items: List[str] = []
    for dlg in dialogues:
        verdict = dlg.get("verdict")
        model = dlg.get("model")
        if verdict not in {"first_pass", "second_pass", "decision", "sql_exec", "sql_fix"}:
            continue
        parsed = dlg.get("parsed") or {}
        if verdict in {"first_pass", "second_pass"}:
            incorrect_parts = parsed.get("incorrect_parts") or []
            reason = (parsed.get("reason") or "").strip()
            intent_guess = (parsed.get("question_intent_guess") or "").strip()
            next_advice = (parsed.get("next_round_advice") or "").strip()
            notes = parsed.get("understanding_notes") or []
            # cap each segment length to avoid blowup
            compact_parts: List[str] = []
            for part in incorrect_parts[:8]:
                if not isinstance(part, dict):
                    continue
                seg = str(part.get("segment") or "").strip()
                why = str(part.get("reason") or "").strip()
                if seg:
                    seg = seg[:160]
                if why:
                    why = why[:200]
                if seg or why:
                    compact_parts.append(f"- segment: {seg} | reason: {why}")
            block = [f"model={model} verdict={verdict}"]
            if compact_parts:
                block.append("incorrect_parts:\n" + "\n".join(compact_parts))
            if reason:
                block.append(f"reason: {reason[:240]}")
            if intent_guess:
                block.append(f"intent_guess: {intent_guess[:240]}")
            if notes:
                compact_notes: List[str] = []
                for n in notes[:6]:
                    s = str(n).strip()
                    if s:
                        compact_notes.append(f"- {s[:200]}")
                if compact_notes:
                    block.append("understanding_notes:\n" + "\n".join(compact_notes))
            if next_advice:
                block.append(f"next_round_advice: {next_advice[:240]}")
            items.append("\n".join(block))
        elif verdict == "decision":
            regen = bool(parsed.get("regenerate"))
            reason = (parsed.get("reason") or "").strip()
            # fixed_sql is intentionally omitted because current prompt already contains the current SQL
            items.append(f"model={model} verdict=decision regenerate={regen} reason={reason[:240]}")
        elif verdict == "sql_exec":
            status = str(dlg.get("status") or "")
            err = str(dlg.get("error") or "").strip()
            items.append(f"tool=mysql_exec status={status} error={err[:300]}")
        elif verdict == "sql_fix":
            reason = str(dlg.get("reason") or "").strip()
            items.append(f"model={model} verdict=sql_fix reason={reason[:240]}")

    if not items:
        return "(empty)"

    selected = items[-max_turns:]
    text = "\n\n".join(f"[{i + 1}] {blk}" for i, blk in enumerate(selected))
    return _truncate_text(text, max_chars) or "(empty)"

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


def _compact_list_str(value: Any, max_items: int, max_item_chars: int) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for it in value[:max_items]:
        if isinstance(it, str):
            s = it.strip()
            if s:
                out.append(s)
        else:
            s = json.dumps(it, ensure_ascii=False)
            if s:
                out.append(s)
    return out


def build_enrichment_text(judge_parsed_list: List[Dict[str, Any]], max_chars: int) -> str:
    """Build a concise enrichment block from judges' structured hypotheses."""
    blocks: List[str] = []
    for idx, parsed in enumerate(judge_parsed_list, 1):
        if not isinstance(parsed, dict) or not parsed:
            continue
        title = parsed.get("_source") or f"judge_{idx}"
        intent = str(parsed.get("question_intent_guess") or "").strip()
        advice = str(parsed.get("next_round_advice") or "").strip()
        notes = _compact_list_str(parsed.get("understanding_notes"), max_items=6, max_item_chars=220)
        gaps = _compact_list_str(parsed.get("schema_gap_hypotheses"), max_items=4, max_item_chars=260)
        miss_map = _compact_list_str(parsed.get("missing_question_info_to_fields"), max_items=4, max_item_chars=260)
        unused = _compact_list_str(parsed.get("unused_table_backfill"), max_items=4, max_item_chars=260)

        lines: List[str] = [f"[{title}]"]
        if intent:
            lines.append(f"- question_intent_guess: {intent}")
        if notes:
            lines.append("- understanding_notes:")
            lines.extend([f"  - {n}" for n in notes])
        if gaps:
            lines.append("- schema_gap_hypotheses:")
            lines.extend([f"  - {g}" for g in gaps])
        if miss_map:
            lines.append("- missing_question_info_to_fields:")
            lines.extend([f"  - {m}" for m in miss_map])
        if unused:
            lines.append("- unused_table_backfill:")
            lines.extend([f"  - {u}" for u in unused])
        if advice:
            lines.append(f"- next_round_advice: {advice}")
        blocks.append("\n".join(lines))

    text = "\n\n".join(blocks).strip()
    if not text:
        return ""
    return text


def append_dialogue(dialogues: List[Dict[str, Any]], entry: Dict[str, Any], log_path: Optional[Path]) -> None:
    """Append a dialogue entry and stream it to a local log file immediately."""
    dialogues.append(entry)
    if not log_path:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fw:
        fw.write(json.dumps(entry, ensure_ascii=False, indent=2) + "\n\n")
        try:
            _flush_writer(fw)
        except Exception:
            pass


def append_history(history: List[Dict[str, Any]], entry: Dict[str, Any], log_path: Optional[Path]) -> None:
    """Append a history entry and stream it to the same log for richer metadata."""
    history.append(entry)
    if not log_path:
        return
    payload = {"type": "history", **entry}
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fw:
        fw.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n")
        try:
            _flush_writer(fw)
        except Exception:
            pass


def append_exec_result(exec_log_path: Optional[Path], payload: Dict[str, Any]) -> None:
    """Append SQL execution result to a JSONL file with real-time persistence."""
    if not exec_log_path:
        return
    exec_log_path.parent.mkdir(parents=True, exist_ok=True)
    with exec_log_path.open("a", encoding="utf-8") as fw:
        fw.write(json.dumps(payload, ensure_ascii=False) + "\n")
        try:
            _flush_writer(fw)
        except Exception:
            pass


def _second_pass_debate_directive(self_tag: str, other_tag: str, other_focus: str) -> str:
    """Extra constraint to force real debate and avoid self-repetition."""
    return (
        f"【Second-pass 辩论规则】你现在是{self_tag}的second_pass。\n"
        "1) 你的reason与incorrect_parts必须基于历史进行【更新】，不得与first_pass完全一致或高度重复；否则视为失败。\n"
        f"2) 你必须明确指出并反驳/修正{other_tag}至少1条观点（从{other_focus}角度），并补充你自己的新发现。\n"
        "3) 只输出JSON，不要输出逐步推理。"
    )


def _judge_output_invalid_reason(expected_prefix: str) -> str:
    return f"{expected_prefix} (输出无效：未按要求提供JSON/前缀/有效内容)"


def _validate_judge_parsed(parsed: Dict[str, Any], expected_prefix: str) -> str:
    reason = str(parsed.get("reason") or "").strip()
    incorrect_parts = parsed.get("incorrect_parts")
    if not reason:
        return "missing_reason"
    if not reason.startswith(expected_prefix):
        return "bad_prefix"
    if incorrect_parts is not None and not isinstance(incorrect_parts, list):
        return "bad_incorrect_parts_type"
    return ""


def _enforce_judge_output(
    *,
    model: LLMClient,
    system_prompt: str,
    prompt_text: str,
    extra_context: str,
    dialogues: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    history_mode: HistoryMode,
    history_max_turns: int,
    history_max_chars: int,
    expected_prefix: str,
    max_retries: int = 1,
) -> Dict[str, Any]:
    """Call judge and enforce minimally valid JSON+reason prefix; retry with harder instruction if needed."""
    last = call_judge(
        model,
        system_prompt,
        prompt_text,
        extra_context,
        dialogues,
        temperature,
        max_tokens,
        history_mode,
        history_max_turns,
        history_max_chars,
    )

    for _ in range(max_retries + 1):
        parsed = dict(last.get("parsed") or {})
        err = _validate_judge_parsed(parsed, expected_prefix)
        if not err:
            return last
        hard = (
            f"【输出无效:{err}】你必须只输出JSON，且reason必须以'{expected_prefix}'开头，"
            "incorrect_parts必须是数组；不要复述别人的句子；必须给出可核验的差异观点。"
        )
        last = call_judge(
            model,
            system_prompt,
            prompt_text,
            (extra_context + "\n\n" + hard).strip() if extra_context else hard,
            dialogues,
            temperature,
            max_tokens,
            history_mode,
            history_max_turns,
            history_max_chars,
        )

    # Fallback: force a non-empty, correct-prefix reason so downstream logic can proceed.
    parsed = dict(last.get("parsed") or {})
    parsed["reason"] = _judge_output_invalid_reason(expected_prefix)
    if not isinstance(parsed.get("incorrect_parts"), list):
        parsed["incorrect_parts"] = [{"segment": "OUTPUT_FORMAT", "reason": "invalid judge output"}]
    last["parsed"] = parsed
    last["reason"] = parsed.get("reason") or ""
    last["incorrect_parts"] = parsed.get("incorrect_parts") or []
    return last


def _format_exec_context(sql_text: str, exec_result: Dict[str, Any]) -> str:
    status = str(exec_result.get("status") or "")
    error = str(exec_result.get("error") or "")
    rows_sample = exec_result.get("rows_sample") or []
    sample_json = json.dumps(rows_sample, ensure_ascii=False)
    return (
        "【SQL执行结果】\n"
        f"status={status}\n"
        f"rows_sample_len={len(rows_sample)}\n"
        f"rows_sample={sample_json}\n"
        f"error={error or '(empty)'}\n"
        "【判定提示】一般情况下结果不应为空；若rows_sample_len=0请视为未满足题意并提出修正。"
    )


def validate_sql_with_fixes(
    *,
    question_id: str,
    round_idx: int,
    sql_text: str,
    prompt_text: str,
    model_c: LLMClient,
    dialogues: List[Dict[str, Any]],
    history: List[Dict[str, Any]],
    dialogue_log: Optional[Path],
    exec_results_path: Optional[Path],
    sql_tool: Optional[SQLExecutionTool],
    execute_sql: bool,
    sql_timeout: int,
    sql_fix_max_tries: int,
    temperature: float,
    max_tokens: int,
    history_mode: HistoryMode,
    history_max_turns: int,
    history_max_chars: int,
    require_nonempty: bool,
) -> Tuple[str, bool, str, bool]:
    """Execute SQL (and fix on errors). Returns (final_sql, validated, exec_context, empty_result)."""
    if not execute_sql:
        return sql_text, True, "", False
    if sql_tool is None:
        ctx = "【SQL执行不可用】SQLExecutionTool unavailable"
        append_exec_result(
            exec_results_path,
            {
                "question_id": question_id,
                "round": round_idx,
                "attempt": 0,
                "status": "unavailable",
                "error": "SQLExecutionTool unavailable",
                "sql": sql_text,
            },
        )
        return sql_text, False, ctx, False

    current_sql = sql_text
    last_error = ""
    last_exec: Dict[str, Any] = {"status": "", "error": "", "rows_sample": []}
    attempts: List[Dict[str, Any]] = []

    for fix_idx in range(1, max(1, sql_fix_max_tries) + 1):
        exec_result = sql_tool.run(current_sql, timeout=sql_timeout)
        last_exec = exec_result
        print(
            f"[{question_id}] round={round_idx + 1} tool=mysql_exec attempt={fix_idx} "
            f"status={exec_result.get('status')} error={_short(str(exec_result.get('error') or ''), 220)}"
        )
        attempts.append({"attempt": fix_idx, "sql": current_sql, **exec_result})

        append_exec_result(
            exec_results_path,
            {
                "question_id": question_id,
                "round": round_idx,
                "attempt": fix_idx,
                "sql": current_sql,
                "status": exec_result.get("status"),
                "error": exec_result.get("error"),
                "rows_sample": exec_result.get("rows_sample"),
            },
        )
        append_history(
            history,
            {
                "model": "TOOL",
                "verdict": "sql_exec",
                "round": round_idx,
                "attempt": fix_idx,
                "status": exec_result.get("status"),
                "error": exec_result.get("error"),
                "rows_sample": exec_result.get("rows_sample"),
                "sql": current_sql,
            },
            dialogue_log,
        )
        append_dialogue(
            dialogues,
            {
                "model": "TOOL",
                "verdict": "sql_exec",
                "round": round_idx,
                "attempt": fix_idx,
                "status": exec_result.get("status"),
                "error": exec_result.get("error"),
                "rows_sample": exec_result.get("rows_sample"),
                "sql": current_sql,
            },
            dialogue_log,
        )

        if exec_result.get("status") == "success":
            rows_sample = exec_result.get("rows_sample") or []
            empty_result = len(rows_sample) == 0
            exec_context = _format_exec_context(current_sql, exec_result)
            append_dialogue(
                dialogues,
                {
                    "model": "SYS",
                    "verdict": "sql_exec_summary",
                    "round": round_idx,
                    "parsed": {
                        "status": "success",
                        "rows_sample_len": len(rows_sample),
                        "empty_result": empty_result,
                        "require_nonempty": require_nonempty,
                    },
                    "response_raw": exec_context,
                },
                dialogue_log,
            )
            append_exec_result(
                exec_results_path,
                {
                    "question_id": question_id,
                    "round": round_idx,
                    "attempt": 0,
                    "status": "summary",
                    "validated": True,
                    "empty_result": empty_result,
                    "require_nonempty": require_nonempty,
                    "attempts": attempts,
                },
            )
            return current_sql, True, exec_context, empty_result

        last_error = str(exec_result.get("error") or "")
        fix = call_sql_fixer(
            model=model_c,
            prompt=prompt_text,
            bad_sql=current_sql,
            exec_error=last_error,
            dialogues=dialogues,
            temperature=temperature,
            max_tokens=max_tokens,
            history_mode=history_mode,
            history_max_turns=history_max_turns,
            history_max_chars=history_max_chars,
        )
        print(
            f"[{question_id}] round={round_idx + 1} model=C verdict=sql_fix attempt={fix_idx} "
            f"reason={_short(str(fix.get('reason') or ''))} fixed_sql_chars={len((fix.get('fixed_sql') or '').strip())}"
        )
        append_history(
            history,
            {"model": "C", "verdict": "sql_fix", "round": round_idx, "attempt": fix_idx, "reason": fix.get("reason")},
            dialogue_log,
        )
        append_dialogue(
            dialogues,
            {
                "model": "C",
                "verdict": "sql_fix",
                "round": round_idx,
                "attempt": fix_idx,
                "messages": fix.get("messages"),
                "response_raw": fix.get("raw"),
                "parsed": {"fixed_sql": fix.get("fixed_sql"), "reason": fix.get("reason")},
            },
            dialogue_log,
        )

        if fix.get("fixed_sql"):
            current_sql = str(fix.get("fixed_sql") or "").strip()
        else:
            break

    exec_context = _format_exec_context(current_sql, last_exec)
    append_dialogue(
        dialogues,
        {
            "model": "SYS",
            "verdict": "sql_exec_summary",
            "round": round_idx,
            "parsed": {"status": str(last_exec.get("status") or ""), "validated": False, "error": last_error},
            "response_raw": exec_context,
        },
        dialogue_log,
    )
    append_exec_result(
        exec_results_path,
        {
            "question_id": question_id,
            "round": round_idx,
            "attempt": 0,
            "status": "summary",
            "validated": False,
            "last_error": last_error,
            "attempts": attempts,
        },
    )
    return current_sql, False, exec_context, False


def call_sql_fixer(
    model: LLMClient,
    prompt: str,
    bad_sql: str,
    exec_error: str,
    dialogues: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    history_mode: HistoryMode,
    history_max_turns: int,
    history_max_chars: int,
) -> Dict[str, Any]:
    history_text = ""
    if history_mode in {"full", "compact"}:
        # Do not truncate or drop content; keep full history.
        history_text = dialogues_to_text(dialogues)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SQL_FIX_SYSTEM_PROMPT},
        {"role": "user", "content": f"【问题上下文】\n{prompt}"},
    ]
    if history_mode != "none" and history_text:
        messages.append({"role": "user", "content": f"【历史记录（摘要）】\n{history_text}"})
    messages.append(
        {
            "role": "user",
            "content": "\n".join(
                [
                    "【待修复 SQL】",
                    bad_sql,
                    "\n【执行报错】",
                    exec_error or "(empty)",
                ]
            ),
        }
    )

    resp = model.invoke(messages=messages, temperature=temperature, max_tokens=max_tokens)
    parsed = parse_json_response(resp) or {}
    return {
        "messages": copy.deepcopy(messages),
        "raw": resp,
        "parsed": parsed,
        "fixed_sql": (parsed.get("fixed_sql") or "").strip(),
        "reason": (parsed.get("reason") or "").strip(),
    }


def apply_endpoint_overrides(base_url: str, api_key: str) -> None:
    """Propagate endpoint overrides to environment for LLMClient reuse."""
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


def call_judge(
    model: LLMClient,
    system_prompt: str,
    prompt: str,
    extra_context: str,
    dialogues: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    history_mode: HistoryMode,
    history_max_turns: int,
    history_max_chars: int,
) -> Dict[str, Any]:
    history_text = ""
    if history_mode in {"full", "compact"}:
        # Do not truncate or drop content; keep full history.
        history_text = dialogues_to_text(dialogues)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"【问题上下文】\n{prompt}"},
    ]
    if extra_context:
        messages.append(
            {
                "role": "user",
                "content": f"【补充推断/缺失信息假设（用于下一轮完善与SQL修正）】\n{extra_context}",
            }
        )
    if history_mode != "none" and history_text:
        messages.append({"role": "user", "content": f"【历史记录（摘要）】\n{history_text}"})

    resp = model.invoke(messages=messages, temperature=temperature, max_tokens=max_tokens)
    parsed = parse_json_response(resp) or {}
    return {
        "messages": copy.deepcopy(messages),
        "history_text": history_text,
        "history_snapshot": copy.deepcopy(dialogues),
        "raw": resp,
        "parsed": parsed,
        "incorrect_parts": parsed.get("incorrect_parts") or [],
        "reason": parsed.get("reason") or "",
    }


def call_decider(
    model: LLMClient,
    prompt: str,
    extra_context: str,
    dialogues: List[Dict[str, Any]],
    temperature: float,
    max_tokens: int,
    history_mode: HistoryMode,
    history_max_turns: int,
    history_max_chars: int,
) -> Dict[str, Any]:
    history_text = ""
    if history_mode in {"full", "compact"}:
        # Do not truncate or drop content; keep full history.
        history_text = dialogues_to_text(dialogues)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": f"【问题上下文】\n{prompt}"},
    ]
    if extra_context:
        messages.append(
            {
                "role": "user",
                "content": f"【A/B补充推断（请吸收后再裁决/重写SQL）】\n{extra_context}",
            }
        )
    if history_mode != "none" and history_text:
        messages.append({"role": "user", "content": f"【历史记录（摘要）】\n{history_text}"})

    resp = model.invoke(messages=messages, temperature=temperature, max_tokens=max_tokens)
    parsed = parse_json_response(resp) or {}
    return {
        "messages": copy.deepcopy(messages),
        "history_text": history_text,
        "history_snapshot": copy.deepcopy(dialogues),
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
    max_tokens: int,
    dialogue_log: Optional[Path],
    history_mode: HistoryMode,
    history_max_turns: int,
    history_max_chars: int,
    execute_sql: bool,
    sql_timeout: int,
    sql_fix_max_tries: int,
    exec_results_path: Optional[Path],
    require_nonempty: bool,
) -> Dict[str, Any]:
    question_id = record.get("question_id") or record.get("sql_id") or "unknown"
    current_sql = read_sql_text(sql_dir, question_id)
    prompt_text = build_prompt(record, current_sql)
    history: List[Dict[str, Any]] = []
    dialogues: List[Dict[str, Any]] = []
    sql_tool = None
    sql_tool_init_error = ""
    if execute_sql:
        try:
            sql_tool = SQLExecutionTool()
        except Exception as e:
            sql_tool = None
            sql_tool_init_error = f"SQLExecutionTool init failed: {e}"

    # log record start
    append_dialogue(dialogues, {"type": "record_start", "question_id": question_id}, dialogue_log)
    print(f"[{question_id}] start")
    if execute_sql and sql_tool is None:
        append_history(
            history,
            {
                "model": "TOOL",
                "verdict": "sql_exec_unavailable",
                "round": -1,
                "error": sql_tool_init_error or "SQLExecutionTool unavailable",
            },
            dialogue_log,
        )
        append_exec_result(
            exec_results_path,
            {
                "question_id": question_id,
                "round": -1,
                "attempt": 0,
                "status": "unavailable",
                "error": sql_tool_init_error or "SQLExecutionTool unavailable",
            },
        )

    exec_context_for_next_round = ""

    for round_idx in range(max_rounds):
        round_base_dialogues = copy.deepcopy(dialogues)
        # First-pass judgments (pass full dialogues list so model receives all prior turns)
        judge_a = _enforce_judge_output(
            model=model_a,
            system_prompt=EVAL_SYSTEM_PROMPT_A,
            prompt_text=prompt_text,
            extra_context=exec_context_for_next_round,
            dialogues=round_base_dialogues,
            temperature=temperature,
            max_tokens=max_tokens,
            history_mode=history_mode,
            history_max_turns=history_max_turns,
            history_max_chars=history_max_chars,
            expected_prefix="A(业务):",
            max_retries=1,
        )
        parsed_a_1 = dict(judge_a.get("parsed") or {})
        parsed_a_1["_source"] = "A_first_pass"
        print(
            f"[{question_id}] round={round_idx + 1}/{max_rounds} model=A verdict=first_pass "
            f"incorrect_parts={len(judge_a.get('incorrect_parts') or [])} reason={_short(str(judge_a.get('reason') or ''))}"
        )
        append_history(history, {
            "model": "A",
            "verdict": "first_pass",
            "incorrect_parts": judge_a["incorrect_parts"],
            "reason": judge_a["reason"],
            "round": round_idx,
        }, dialogue_log)
        # Append A after both A/B are finished to avoid same-round leakage.
        append_dialogue(dialogues, {
            "model": "A",
            "verdict": "first_pass",
            "messages": judge_a.get("messages"),
            "history_text": judge_a.get("history_text"),
            "history_snapshot": judge_a.get("history_snapshot"),
            "response_raw": judge_a.get("raw"),
            "parsed": parsed_a_1,
        }, dialogue_log)

        judge_b = _enforce_judge_output(
            model=model_b,
            system_prompt=EVAL_SYSTEM_PROMPT_B,
            prompt_text=prompt_text,
            extra_context=exec_context_for_next_round,
            dialogues=round_base_dialogues,
            temperature=temperature,
            max_tokens=max_tokens,
            history_mode=history_mode,
            history_max_turns=history_max_turns,
            history_max_chars=history_max_chars,
            expected_prefix="B(SQL):",
            max_retries=1,
        )
        parsed_b_1 = dict(judge_b.get("parsed") or {})
        parsed_b_1["_source"] = "B_first_pass"
        print(
            f"[{question_id}] round={round_idx + 1}/{max_rounds} model=B verdict=first_pass "
            f"incorrect_parts={len(judge_b.get('incorrect_parts') or [])} reason={_short(str(judge_b.get('reason') or ''))}"
        )
        append_history(history, {
            "model": "B",
            "verdict": "first_pass",
            "incorrect_parts": judge_b["incorrect_parts"],
            "reason": judge_b["reason"],
        }, dialogue_log)
        append_dialogue(dialogues, {
            "model": "B",
            "verdict": "first_pass",
            "messages": judge_b.get("messages"),
            "history_text": judge_b.get("history_text"),
            "history_snapshot": judge_b.get("history_snapshot"),
            "response_raw": judge_b.get("raw"),
            "parsed": parsed_b_1,
        }, dialogue_log)

        enrichment_for_second_pass = build_enrichment_text([parsed_a_1, parsed_b_1], max_chars=0)
        if enrichment_for_second_pass:
            append_dialogue(
                dialogues,
                {
                    "model": "SYS",
                    "verdict": "enrichment",
                    "round": round_idx,
                    "parsed": {"enrichment_chars": len(enrichment_for_second_pass)},
                    "response_raw": enrichment_for_second_pass,
                },
                dialogue_log,
            )

        # Second-pass with history
        # Second pass: include all accumulated dialogues
        a_second_extra = "\n\n".join(
            [
                enrichment_for_second_pass,
                _second_pass_debate_directive("评审员A(业务)", "评审员B", "SQL语义/关联逻辑/聚合与去重"),
                exec_context_for_next_round,
            ]
        ).strip()
        judge_a_2 = _enforce_judge_output(
            model=model_a,
            system_prompt=EVAL_SYSTEM_PROMPT_A,
            prompt_text=prompt_text,
            extra_context=a_second_extra,
            dialogues=dialogues,
            temperature=temperature,
            max_tokens=max_tokens,
            history_mode=history_mode,
            history_max_turns=history_max_turns,
            history_max_chars=history_max_chars,
            expected_prefix="A(业务):",
            max_retries=1,
        )
        parsed_a_2 = dict(judge_a_2.get("parsed") or {})
        parsed_a_2["_source"] = "A_second_pass"

        # Auto-retry once if A second pass repeats its first pass too closely.
        if _is_same_reason(str(parsed_a_1.get("reason") or ""), str(parsed_a_2.get("reason") or "")) and _is_same_incorrect_parts(
            parsed_a_1.get("incorrect_parts"), parsed_a_2.get("incorrect_parts")
        ):
            retry_note = (
                "【检测到复读】你second_pass的reason/incorrect_parts与first_pass相同。"
                "你必须提供与first_pass不同的更新观点，并反驳B至少1点；否则输出无效。"
            )
            judge_a_2_retry = call_judge(
                model_a,
                EVAL_SYSTEM_PROMPT_A,
                prompt_text,
                (a_second_extra + "\n\n" + retry_note).strip(),
                dialogues,
                temperature,
                max_tokens,
                history_mode,
                history_max_turns,
                history_max_chars,
            )
            parsed_a_2_retry = dict(judge_a_2_retry.get("parsed") or {})
            parsed_a_2_retry["_source"] = "A_second_pass_retry"
            # Prefer retry if it changes reason.
            if not _is_same_reason(str(parsed_a_1.get("reason") or ""), str(parsed_a_2_retry.get("reason") or "")):
                judge_a_2 = judge_a_2_retry
                parsed_a_2 = parsed_a_2_retry
        print(
            f"[{question_id}] round={round_idx + 1}/{max_rounds} model=A verdict=second_pass "
            f"incorrect_parts={len(judge_a_2.get('incorrect_parts') or [])} reason={_short(str(judge_a_2.get('reason') or ''))}"
        )
        append_history(history, {
            "model": "A",
            "verdict": "second_pass",
            "incorrect_parts": judge_a_2["incorrect_parts"],
            "reason": judge_a_2["reason"],
            "round": round_idx,
        }, dialogue_log)
        append_dialogue(dialogues, {
            "model": "A",
            "verdict": "second_pass",
            "messages": judge_a_2.get("messages"),
            "history_text": judge_a_2.get("history_text"),
            "history_snapshot": judge_a_2.get("history_snapshot"),
            "response_raw": judge_a_2.get("raw"),
            "parsed": parsed_a_2,
        }, dialogue_log)

        b_second_extra = "\n\n".join(
            [
                enrichment_for_second_pass,
                _second_pass_debate_directive("评审员B(SQL)", "评审员A", "业务口径/指标定义/数据链路"),
                exec_context_for_next_round,
            ]
        ).strip()
        judge_b_2 = _enforce_judge_output(
            model=model_b,
            system_prompt=EVAL_SYSTEM_PROMPT_B,
            prompt_text=prompt_text,
            extra_context=b_second_extra,
            dialogues=dialogues,
            temperature=temperature,
            max_tokens=max_tokens,
            history_mode=history_mode,
            history_max_turns=history_max_turns,
            history_max_chars=history_max_chars,
            expected_prefix="B(SQL):",
            max_retries=2,
        )
        parsed_b_2 = dict(judge_b_2.get("parsed") or {})
        parsed_b_2["_source"] = "B_second_pass"

        # Auto-retry once if B second pass repeats its first pass too closely.
        if _is_same_reason(str(parsed_b_1.get("reason") or ""), str(parsed_b_2.get("reason") or "")) and _is_same_incorrect_parts(
            parsed_b_1.get("incorrect_parts"), parsed_b_2.get("incorrect_parts")
        ):
            retry_note = (
                "【检测到复读】你second_pass的reason/incorrect_parts与first_pass相同。"
                "你必须提供与first_pass不同的更新观点，并反驳A至少1点；否则输出无效。"
            )
            judge_b_2_retry = call_judge(
                model_b,
                EVAL_SYSTEM_PROMPT_B,
                prompt_text,
                (b_second_extra + "\n\n" + retry_note).strip(),
                dialogues,
                temperature,
                max_tokens,
                history_mode,
                history_max_turns,
                history_max_chars,
            )
            parsed_b_2_retry = dict(judge_b_2_retry.get("parsed") or {})
            parsed_b_2_retry["_source"] = "B_second_pass_retry"
            if not _is_same_reason(str(parsed_b_1.get("reason") or ""), str(parsed_b_2_retry.get("reason") or "")):
                judge_b_2 = judge_b_2_retry
                parsed_b_2 = parsed_b_2_retry
        print(
            f"[{question_id}] round={round_idx + 1}/{max_rounds} model=B verdict=second_pass "
            f"incorrect_parts={len(judge_b_2.get('incorrect_parts') or [])} reason={_short(str(judge_b_2.get('reason') or ''))}"
        )
        append_history(history, {
            "model": "B",
            "verdict": "second_pass",
            "incorrect_parts": judge_b_2["incorrect_parts"],
            "reason": judge_b_2["reason"],
        }, dialogue_log)
        append_dialogue(dialogues, {
            "model": "B",
            "verdict": "second_pass",
            "messages": judge_b_2.get("messages"),
            "history_text": judge_b_2.get("history_text"),
            "history_snapshot": judge_b_2.get("history_snapshot"),
            "response_raw": judge_b_2.get("raw"),
            "parsed": parsed_b_2,
        }, dialogue_log)

        enrichment_for_decision = build_enrichment_text(
            [parsed_a_2, parsed_b_2, parsed_a_1, parsed_b_1],
            max_chars=0,
        )

        issues_exist = bool((judge_a_2.get("incorrect_parts") or []) or (judge_b_2.get("incorrect_parts") or []))

        # Decision and potential regeneration
        # Decision: pass full dialogues
        decision = call_decider(
            model_c,
            prompt_text,
            (enrichment_for_decision + "\n\n" + exec_context_for_next_round).strip() if exec_context_for_next_round else enrichment_for_decision,
            dialogues,
            temperature,
            max_tokens,
            history_mode,
            history_max_turns,
            history_max_chars,
        )

        # Hard rule: if any judge flags issues, we must regenerate.
        if issues_exist and (not decision.get("regenerate") or not (decision.get("fixed_sql") or "").strip()):
            forced_note = (
                "【硬性规则】评审员A/B在second_pass给出了incorrect_parts（代表存在问题）。"
                "在本任务设定下：只要存在incorrect_parts，就必须 regenerate=true 并输出 fixed_sql。"
                "此外，必须满足：SQL需要使用schema中的全部表（缺表=错误），且不要用‘无关’来跳过表。"
            )
            decision_forced = call_decider(
                model_c,
                prompt_text,
                (enrichment_for_decision + "\n\n" + forced_note).strip(),
                dialogues,
                temperature,
                max_tokens,
                history_mode,
                history_max_turns,
                history_max_chars,
            )

            # Prefer forced decision if it produces SQL.
            if (decision_forced.get("fixed_sql") or "").strip():
                decision = decision_forced
                decision["regenerate"] = True
            else:
                # As last resort, override regenerate to reflect the hard rule.
                decision["regenerate"] = True

        # If regeneration is required but model didn't provide SQL, ask once more explicitly.
        if decision.get("regenerate") and not (decision.get("fixed_sql") or "").strip():
            must_provide_sql = (
                "【必须输出】你已被要求 regenerate=true，但未提供 fixed_sql。"
                "请现在输出严格符合格式的JSON，并在 fixed_sql 字段给出可执行的MySQL查询SQL（必须使用schema全部表）。"
            )
            decision_retry = call_decider(
                model_c,
                prompt_text,
                (enrichment_for_decision + "\n\n" + must_provide_sql).strip(),
                dialogues,
                temperature,
                max_tokens,
                history_mode,
                history_max_turns,
                history_max_chars,
            )
            if (decision_retry.get("fixed_sql") or "").strip():
                decision = decision_retry
                decision["regenerate"] = True
        print(
            f"[{question_id}] round={round_idx + 1}/{max_rounds} model=C verdict=decision "
            f"regenerate={bool(decision.get('regenerate'))} reason={_short(str(decision.get('reason') or ''))}"
        )
        append_history(history, {
            "model": "C",
            "verdict": "decision",
            "regenerate": decision["regenerate"],
            "reason": decision["reason"],
            "fixed_sql": decision["fixed_sql"],
            "round": round_idx,
        }, dialogue_log)
        append_dialogue(dialogues, {
            "model": "C",
            "verdict": "decision",
            "messages": decision.get("messages"),
            "history_text": decision.get("history_text"),
            "history_snapshot": decision.get("history_snapshot"),
            "response_raw": decision.get("raw"),
            "parsed": {
                "regenerate": decision.get("regenerate"),
                "reason": decision.get("reason"),
                "fixed_sql": decision.get("fixed_sql"),
            },
        }, dialogue_log)

        # Execute (and fix if needed) at the end of every round, using the SQL that will be carried forward.
        if decision.get("regenerate") and (decision.get("fixed_sql") or "").strip():
            candidate_sql = str(decision.get("fixed_sql") or "").strip()
            print(f"[{question_id}] regenerated_sql_chars={len(candidate_sql)}")
        else:
            candidate_sql = current_sql

        validated_sql, validated, exec_ctx, empty_result = validate_sql_with_fixes(
            question_id=question_id,
            round_idx=round_idx,
            sql_text=candidate_sql,
            prompt_text=prompt_text,
            model_c=model_c,
            dialogues=dialogues,
            history=history,
            dialogue_log=dialogue_log,
            exec_results_path=exec_results_path,
            sql_tool=sql_tool,
            execute_sql=execute_sql,
            sql_timeout=sql_timeout,
            sql_fix_max_tries=sql_fix_max_tries,
            temperature=temperature,
            max_tokens=max_tokens,
            history_mode=history_mode,
            history_max_turns=history_max_turns,
            history_max_chars=history_max_chars,
            require_nonempty=require_nonempty,
        )

        exec_context_for_next_round = exec_ctx
        current_sql = validated_sql

        if not validated:
            print(f"[{question_id}] stop: sql not validated")
            break

        # If query succeeded but returned empty and require_nonempty, force regeneration once (bounded).
        if require_nonempty and empty_result:
            append_history(
                history,
                {"model": "TOOL", "verdict": "sql_empty_result", "round": round_idx, "sql": current_sql},
                dialogue_log,
            )
            empty_note = (
                "【硬性规则】SQL执行成功但结果为空（rows_sample_len=0）。一般检索类题目结果不应为空，"
                "请视为未满足题意，必须 regenerate=true 并输出能返回非空结果的 fixed_sql（仍需满足使用schema全部表）。"
            )
            decision_retry2 = call_decider(
                model_c,
                prompt_text,
                (enrichment_for_decision + "\n\n" + exec_context_for_next_round + "\n\n" + empty_note).strip(),
                dialogues,
                temperature,
                max_tokens,
                history_mode,
                history_max_turns,
                history_max_chars,
            )
            if (decision_retry2.get("fixed_sql") or "").strip():
                decision = decision_retry2
                decision["regenerate"] = True
                regen_sql = str(decision.get("fixed_sql") or "").strip()
                print(f"[{question_id}] regenerated_sql_chars={len(regen_sql)}")
                validated_sql, validated, exec_ctx, empty_result = validate_sql_with_fixes(
                    question_id=question_id,
                    round_idx=round_idx,
                    sql_text=regen_sql,
                    prompt_text=prompt_text,
                    model_c=model_c,
                    dialogues=dialogues,
                    history=history,
                    dialogue_log=dialogue_log,
                    exec_results_path=exec_results_path,
                    sql_tool=sql_tool,
                    execute_sql=execute_sql,
                    sql_timeout=sql_timeout,
                    sql_fix_max_tries=sql_fix_max_tries,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    history_mode=history_mode,
                    history_max_turns=history_max_turns,
                    history_max_chars=history_max_chars,
                    require_nonempty=require_nonempty,
                )
                exec_context_for_next_round = exec_ctx
                current_sql = validated_sql
                if not validated:
                    print(f"[{question_id}] stop: sql not validated")
                    break

        if not decision.get("regenerate"):
            print(f"[{question_id}] stop: regenerate=false")
            break

        prompt_text = build_prompt(record, current_sql)

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
    parser.add_argument("--max-tokens", type=int, default=8192, help="Max tokens per LLM call (set to your model's maximum)")
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--limit", type=int, default=-1, help="Optional cap on records processed")
    parser.add_argument("--start", type=int, default=0, help="Zero-based start index")
    parser.add_argument("--offline", action="store_true", help="Use offline heuristic mode for all models")
    parser.add_argument("--history-mode", choices=["compact", "full", "none"], default="compact", help="How to feed history to the LLM")
    parser.add_argument("--history-max-turns", type=int, default=8, help="Max compact history turns to include")
    parser.add_argument("--history-max-chars", type=int, default=8000, help="Max characters for history payload")
    parser.add_argument("--execute-sql", action="store_true", default=True, help="Execute regenerated SQL via MySQL and let model C fix until runnable")
    parser.add_argument("--sql-timeout", type=int, default=60, help="SQL execution timeout (seconds)")
    parser.add_argument("--sql-fix-max-tries", type=int, default=6, help="Max fix attempts for model C after SQL execution errors")
    parser.add_argument("--dialogue-log", default=str(BASE_DIR / "debate_dialogues.log"), help="Path to stream dialogue entries (JSONL); empty to disable")
    parser.add_argument("--exec-results", default=str(BASE_DIR / "competion_dataset" / "debate_exec_results.jsonl"), help="Path to append SQL exec results JSONL; empty to disable")
    parser.add_argument("--require-nonempty", action="store_true", default=True, help="Treat successful-but-empty result as a hard issue and force regeneration")
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
    # Append mode for real-time incremental writes (do not overwrite existing results).
    with output_path.open("a", encoding="utf-8") as writer:
        for record in subset:
            result = run_debate(
                record=record,
                model_a=model_a,
                model_b=model_b,
                model_c=model_c,
                sql_dir=sql_dir,
                max_rounds=args.max_rounds,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                dialogue_log=Path(args.dialogue_log) if args.dialogue_log else None,
                history_mode=args.history_mode,
                history_max_turns=args.history_max_turns,
                history_max_chars=args.history_max_chars,
                execute_sql=args.execute_sql,
                sql_timeout=args.sql_timeout,
                sql_fix_max_tries=args.sql_fix_max_tries,
                exec_results_path=Path(args.exec_results) if args.exec_results else None,
                require_nonempty=args.require_nonempty,
            )
            writer.write(json.dumps(result, ensure_ascii=False) + "\n")
            _flush_writer(writer)
            # persist final SQL as .sql file per question
            qid = result.get("question_id") or f"idx_{processed}"
            if result.get("final_sql"):
                (sql_output_dir / f"{qid}.sql").write_text(result["final_sql"], encoding="utf-8")
            processed += 1
            print(f"[{processed}] {result['question_id']} processed")

    print(f"Done. Wrote {processed} results to {output_path}")


if __name__ == "__main__":
    main()
