# -*- coding: utf-8 -*-

"""Agent-style question decomposition and SQL augmentation.

This refactor mirrors the ``spider-agent-tc`` structure by splitting the logic
into reusable components:

* ``LLMClient`` centralises OpenAI-compatible calls with retries.
* ``StepPlanner`` emits a meta planning step plus concrete sub-steps.
* ``StepExecutor`` treats SQL execution as a tool call and performs sequential
  retries with LLM-based revisions.
* ``AugmentationAgent`` streams dataset variants to disk with a JSON writer,
  collecting the full agent dialogue for later inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from decimal import Decimal

from dotenv import load_dotenv

from alphasql.database.sql_execution import execute_sql_with_pymysql
from alphasql.llm_call.openai_llm import DEFAULT_COST_RECORDER, call_openai

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Prompts (concise, JSON enforced to match spider-agent conventions)
# ---------------------------------------------------------------------------

META_PLAN_PROMPT = """【角色】你是资深数据分析教练，会在编写 SQL 之前拆解任务。
【输入】
- 目标总步数上限：{max_steps}（第 0 步固定为元规划）
- 用户问题：见下文
- 数据库 schema：见下文
- 参考信息：见下文
- 原问题 SQL 答案：见下文
【任务】
1. 产出若干个按顺序排列的子问题标题，覆盖过滤、关联、集合构建、聚合/排序等关键逻辑；
2. 子问题数量可自适应决定，但需落在 {min_sub_steps} 至 {max_sub_steps} 的范围内；
3. 子问题需语义明确、避免重复，必要时引用实体或字段，且与原问题紧密呼应；
4. 每个子问题需自洽，不能依赖原问题的隐含信息，应包含完成该子任务所需的最小完整的语义背景。
【输出格式】
```json
{{"sub_questions": ["子问题1", "子问题2", ...]}}
```
子问题数组长度需满足 {min_sub_steps} ≤ N ≤ {max_sub_steps}。

【用户问题】
{user_question}

【数据库schema】
{schema_text}

【参考信息】
{ref_text}

【原问题SQL答案】
{final_sql}
"""

STEP_SQL_ONE_PROMPT = """【角色】你是资深 MySQL 工程师，需在已有对话上下文中继续解决当前步骤。
【任务说明】
- 当前步骤：{step_desc}
- is_final_step={is_final_step}（false 表示仅返回可复用的中间集合，true 必须给出最终答案）
- 补充提示：{ref_text}
【写作准则】
1. 充分承接既有历史，不要重复粘贴上下文；
2. 当 is_final_step=false 时，输出满足后续需求的最小字段集合，避免聚合/排序；当 is_final_step=true 时，SQL 必须覆盖原问题全部约束；
3. 仅输出 JSON：{{"sql": "..."}}，SQL 使用标准 MySQL 语法且保持可执行。
"""

SCOPE_REVISION_PROMPT = """【情景】当前 SQL 覆盖范围过大或提前产出最终答案。
【目标】仅返回该步骤所需的最小中间结果，禁止聚合或排序。
【必要信息】
- 步骤描述：{step_desc}
- 过度 SQL：{current_sql}
【输出格式】JSON：{{"sql": "..."}}
"""

REVISION_PROMPT = """【情景】步骤 SQL 执行失败，需要基于历史上下文重新修订。
【信息】
- 步骤描述：{step_desc}
- 失败 SQL：{failed_sql}
- 错误信息：{error_msg}
- 额外提示：{ref_text}
【要求】指出问题并给出修正后的 SQL，仅输出 JSON：{{"revised_sql": "..."}}，禁止返回其他内容。
"""

FINAL_SYNTHESIS_PROMPT = """【角色】你是资深 MySQL 工程师，需要将全部已验证的步骤 SQL 汇总为最终答案。
【上下文】
- 用户问题：{user_question}
- 数据库 schema：{schema_text}
- 参考信息：{ref_text}
- 步骤及对应 SQL：{steps_and_sqls}
【要求】在确认所有前置 SQL 可复用的前提下，合成满足原始需求的最终 SQL。仅输出 JSON：{{"final_sql": "..."}}。
"""

REFLECTION_STEP_PROMPT = """【任务】复核步骤 {step_idx} 的 SQL 是否正确。
【步骤描述】
{step_desc}
【当前 SQL】
{current_sql}
【输出规则】
- 若完全正确：输出 {{"is_correct": true}}
- 若需修正：输出 {{"is_correct": false, "revised_sql": "...", "reason": "问题定位"}}，说明错误原因并保持与其他步骤不重复。
"""

EXECUTOR_SYSTEM_PROMPT = "【角色】资深 MySQL 工程师，负责逐步生成/修正子问题 SQL。" \
    "【要求】始终结合对话历史，输出严格 JSON（含 SQL 字段），遵循 MySQL 只读查询规范。"
REFLECTION_SYSTEM_PROMPT = "【角色】资深 MySQL 专家，负责基于历史对话复核并修正每个步骤 SQL。" \
    "【要求】仅输出 JSON 结论或修正，确保与整体意图一致。"


DETECTION_FINAL_STEP_PROMPT = "【角色】资深数据分析专家，负责检测当前步骤的俩个 MYSQL 是否等价 。" \
    "【用户问题】\n{step_desc}\n" \
    "【数据库schema】\n{schema_text}\n" \
    "【参考信息】\n{ref_text}\n" \
    "FINAL GENERATED SQL: {final_generated_sql} ; ORIGINAL FINAL SQL: {original_final_sql} " \
    "【要求】仅输出 JSON 结论，格式：{{'are_equivalent': true}} 或 {{'are_equivalent': false, 'reason': '不等价原因'}} 。"



# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StepPlan:
    desc: str
    independent: bool
    depends_on: List[int]
    meta: bool = False


@dataclass
class StepSQLResult:
    plan: StepPlan
    sql: str
    status: str
    error: Optional[str] = None
    rows_sample: List[Any] = field(default_factory=list)
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    versions: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # Attempt to extract JSON from markdown code blocks
    pattern = r"```(?:json)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        try:
            data = json.loads(match.strip())
            if isinstance(data, dict):
                return data
        except Exception:
            continue

    # Attempt to find the first JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start : end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # Fallback: try parsing the entire text
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None


def parse_cn_prompt_blocks(text: str) -> Dict[str, str]:
    if not text:
        return {"user_question": "", "schema_text": "", "ref_text": ""}
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    def _segment(start: str, end: Optional[str]) -> str:
        s_idx = normalized.find(start)
        if s_idx == -1:
            return ""
        s_idx += len(start)
        e_idx = normalized.find(end, s_idx) if end else -1
        return normalized[s_idx:e_idx if e_idx != -1 else len(normalized)].strip()

    user_q = _segment("【用户问题】\n", "【数据库schema】") or _segment("【用户问题】", "【数据库schema】")
    schema_text = _segment("【数据库schema】\n", "【参考信息】") or _segment("【数据库schema】", "【参考信息】")
    ref_text = _segment("【参考信息】\n", None) or _segment("【参考信息】", None)

    if user_q:
        filtered = [ln.strip() for ln in user_q.split('\n') if not ln.startswith("输出：")]
        user_q = "\n".join([ln for ln in filtered if ln])

    return {
        "user_question": user_q.strip(),
        "schema_text": schema_text.strip(),
        "ref_text": ref_text.strip(),
    }


# ---------------------------------------------------------------------------
# LLM + tool wrappers (spider-agent style abstractions)
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, model: str, default_temperature: float, offline: bool = False, max_retries: int = 6):
        self.model = model
        self.default_temperature = default_temperature
        self.offline = offline
        self.max_retries = max_retries

    def invoke(
        self,
        prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 1024,
        messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        if self.offline:
            raise RuntimeError("LLM not available in offline mode")
        if messages is None and not prompt:
            raise ValueError("LLMClient.invoke requires either prompt or messages")
        temp = temperature if temperature is not None else self.default_temperature
        for attempt in range(1, self.max_retries + 1):
            try:
                responses = call_openai(
                    prompt=prompt or (messages[-1]["content"] if messages else ""),
                    model=self.model,
                    temperature=temp,
                    n=1,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                if responses:
                    return responses[0]
            except Exception as exc:  # pragma: no cover
                print(f"[LLM] call failed ({attempt}/{self.max_retries}): {exc}")
                time.sleep(0.4 * attempt)
        raise RuntimeError("LLM invocation exceeded retry limit")


class SQLExecutionTool:
    def run(self, sql_text: str, timeout: int = 60) -> Dict[str, Any]:
        try:
            result = execute_sql_with_pymysql(sql_text, timeout=timeout)
            return {
                "status": result.result_type.value,
                "error": result.error_message,
                "rows_sample": result.result[:3] if result.result else [],
            }
        except Exception as e:
            return {
                "status": "error",
                "error": f"Tool execution exception: {str(e)}",
                "rows_sample": [],
            }


# ---------------------------------------------------------------------------
# Planning logic
# ---------------------------------------------------------------------------

class StepPlanner:
    def __init__(self, llm_client: Optional[LLMClient]):
        self.llm_client = llm_client

    def plan(self, context: Dict[str, str], min_steps: int, max_steps: int, final_sql: str, offline: bool) -> List[StepPlan]:
        print("[planner] planning steps...")
        min_sub_steps = max(1, min_steps - 1)
        max_sub_steps = max(min_sub_steps, max_steps - 1)

        if offline or not self.llm_client:
            print("[planner] offline mode or no LLM client, using heuristic planning")
            return self._offline_plan(context, min_sub_steps, max_sub_steps, final_sql)

        prompt = META_PLAN_PROMPT.format(
            max_steps=max_steps,
            min_sub_steps=min_sub_steps,
            max_sub_steps=max_sub_steps,
            user_question=context.get("user_question", ""),
            schema_text=context.get("schema_text", ""),
            ref_text=context.get("ref_text", ""),
            final_sql=final_sql,
        )
        print("[planner] prompt {}...".format(prompt))
        try:
            response = self.llm_client.invoke(prompt, max_tokens=1024)
            print("[planner] meta planning response: {}".format(response))
            titles = (safe_json_loads(response) or {}).get("sub_questions", [])
            if isinstance(titles, list) and min_sub_steps <= len(titles) <= max_sub_steps:
                return self._wrap_titles(context, titles)
        except Exception as exc:
            print(f"[planner] meta planning failed, fallback to heuristics: {exc}")
        return self._offline_plan(context, min_sub_steps, max_sub_steps, final_sql)

    def _wrap_titles(self, context: Dict[str, str], titles: List[str]) -> List[StepPlan]:
        plans = [self._meta_step(context.get("user_question", ""))]
        for idx, title in enumerate(titles):
            dep = random.random() < 0.4
            plans.append(
                StepPlan(
                    desc=title,
                    independent=not dep,
                    depends_on=[idx] if dep else [],
                )
            )
        return plans

    def _offline_plan(self, context: Dict[str, str], min_sub_steps: int, max_sub_steps: int, final_sql: str) -> List[StepPlan]:
        user_q = context.get("user_question", "")
        need = max(max_sub_steps, min_sub_steps)
        titles = self._heuristic_titles(final_sql, need)
        plans = [self._meta_step(user_q)]
        for idx, title in enumerate(titles):
            dep = random.random() < 0.4
            plans.append(
                StepPlan(
                    desc=title,
                    independent=not dep,
                    depends_on=[idx] if dep else [],
                )
            )
        return plans

    @staticmethod
    def _meta_step(user_q: str) -> StepPlan:
        return StepPlan(
            desc=f"第0步：决定子问题数量与主题，围绕原问题进行分解：{user_q}",
            independent=True,
            depends_on=[],
            meta=True,
        )

    @staticmethod
    def _heuristic_titles(final_sql: str, need: int) -> List[str]:
        sql = (final_sql or "").lower()
        buckets: List[str] = []
        if re.search(r"dtstatdate|date|statis_date", sql):
            buckets.append("按时间窗口过滤记录")
        if re.search(r"sgamecode|splattype|saccounttype|suseridtype", sql):
            buckets.append("筛选指定游戏/平台/账号类型")
        if " join " in sql:
            buckets.append("关联用户/维度表，补充字段")
        if "distinct" in sql:
            buckets.append("形成去重后的候选集合")
        if re.search(r"group by|count\(|sum\(|avg\(", sql):
            buckets.append("对候选集合进行聚合统计")
        if "order by" in sql:
            buckets.append("按指定字段排序并输出")
        if not buckets:
            buckets = ["识别核心过滤条件", "构建候选集合", "整合并输出结果"]
        deduped: List[str] = []
        for item in buckets:
            if item not in deduped:
                deduped.append(item)
        while len(deduped) < max(1, need):
            deduped.append(f"补充条件细化（步骤{len(deduped)}）")
        return deduped[: max(1, need)]


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

class StepExecutor:
    def __init__(self, llm_client: Optional[LLMClient], sql_tool: SQLExecutionTool, args):
        self.llm_client = llm_client
        self.sql_tool = sql_tool
        self.args = args
        self.chat_history: List[Dict[str, str]] = []

    def reset_history(self, context: Optional[Dict[str, str]] = None, system_prompt: Optional[str] = None):
        self.chat_history = []
        base_prompt = system_prompt or EXECUTOR_SYSTEM_PROMPT
        if base_prompt:
            self.chat_history.append({"role": "system", "content": base_prompt})
        if context:
            seed = self._compose_context_seed(context)
            if seed:
                self.chat_history.append({"role": "user", "content": seed})

    def _history_snapshot(self) -> List[Dict[str, str]]:
        return [msg.copy() for msg in self.chat_history]

    def _chat_completion(self, user_prompt: str, temperature: float, max_tokens: int) -> str:
        if self.llm_client is None:
            raise RuntimeError("LLM client unavailable for chat completion")
        self.chat_history.append({"role": "user", "content": user_prompt})
        response = self.llm_client.invoke(
            temperature=temperature,
            max_tokens=max_tokens,
            messages=self._history_snapshot(),
        )
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    @staticmethod
    def _compose_context_seed(context: Dict[str, str]) -> str:
        parts = []
        uq = context.get("user_question")
        schema = context.get("schema_text")
        ref = context.get("ref_text")
        final_sql = context.get("final_sql")
        if uq:
            parts.append(f"【用户问题】\n{uq}")
        if schema:
            parts.append(f"【数据库schema】\n{schema}")
        if ref:
            parts.append(f"【参考信息】\n{ref}")
        if final_sql:
            parts.append(f"【原问题SQL答案】\n{final_sql}")
        return "\n\n".join(parts)

    def run_step(
        self,
        step: StepPlan,
        context: Dict[str, str],
        final_sql: str,
        is_final_step: bool,
        validated_sqls: List[str],
        dialogues: List[Dict[str, Any]],
        variant_id: str,
    ) -> StepSQLResult:
        if step.meta:
            dialogues.append({"role": "user", "content": "[META] 第0步：高层规划与子问题分解。"})
            dialogues.append(
                {
                    "role": "note",
                    "content": json.dumps({"ack_meta": True}, ensure_ascii=False),
                    "meta": {"exclude_from_history": True, "prompt_type": "META_ACK"},
                }
            )
            return StepSQLResult(step, sql="", status="meta", rows_sample=[])

        enriched_context = context.copy()
        if validated_sqls:
            enriched_context["ref_text"] = (
                context.get("ref_text", "")
                + "\n"
                + json.dumps({"validated_prev_sqls": [sql for sql in validated_sqls if sql]}, ensure_ascii=False)
            ).strip()

        # 第1轮：生成并内部自修正，直到能正常执行或达到最大重试
        sql_candidate = self._generate_sql(step, enriched_context, final_sql, is_final_step, dialogues)
        attempts: List[Dict[str, Any]] = []
        current_sql = sql_candidate
        success = False
        last_error: Optional[str] = None
        last_rows_sample: List[Any] = []

        # 在单轮对话中，允许多次“执行->报错->自修正”尝试
        for attempt in range(1, self.args.max_step_retries + 1):
            exec_result = self.sql_tool.run(current_sql)
            last_rows_sample = exec_result.get("rows_sample", [])
            attempts.append(
                {
                    "attempt": attempt,
                    "sql": current_sql,
                    "status": exec_result["status"],
                    "error": exec_result.get("error"),
                    "rows_sample": exec_result.get("rows_sample"),
                }
            )
            dialogues.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "step_desc": step.desc,
                            "attempt": attempt,
                            **exec_result,
                        },
                        ensure_ascii=False,
                        default=str,
                    ),
                }
            )

            if exec_result["status"] == "success":
                success = True
                break

            last_error = exec_result.get("error")
            if attempt >= self.args.max_step_retries:
                break
            # 自修正阶段：不依赖最终标准答案，只根据执行错误修正当前子问题 SQL
            current_sql = self._revise_sql(step, current_sql, last_error, dialogues, enriched_context)

        status = "success" if success else "error"
        versions: List[Dict[str, Any]] = []
        if success:
            versions.append(
                {
                    "round": 0,
                    "sql": current_sql,
                    "status": "success",
                    "rows_sample": last_rows_sample,
                    "error": None,
                    "reason": None,
                }
            )
        return StepSQLResult(
            step,
            sql=current_sql,
            status=status,
            error=last_error,
            rows_sample=last_rows_sample,
            attempts=attempts,
            versions=versions,
        )

    def _generate_sql(
        self,
        step: StepPlan,
        context: Dict[str, str],
        final_sql: str,
        is_final_step: bool,
        dialogues: List[Dict[str, Any]],
    ) -> str:
        if self.llm_client is None:
            return final_sql if is_final_step else "SELECT DISTINCT vplayerid FROM dws_jordass_matchlog_stat_di LIMIT 50"

        prompt = STEP_SQL_ONE_PROMPT.format(
            step_desc=step.desc,
            ref_text=context.get("ref_text", ""),
            is_final_step=str(is_final_step).lower(),
        )
        # 记录完整的 prompt 方便离线分析
        dialogues.append(
            {
                "role": "user",
                "content": prompt,
                "meta": {
                    "prompt_type": "STEP_SQL_ONE_PROMPT",
                    "is_final_step": is_final_step,
                },
            }
        )
        response = self._chat_completion(prompt, self.args.sqlgen_temperature, 4096) if self.llm_client else ""
        parsed = safe_json_loads(response) or {}
        sql = parsed.get("sql")

        if not sql:
            sql = final_sql if is_final_step else "SELECT DISTINCT vplayerid FROM dws_jordass_matchlog_stat_di WHERE dtstatdate >= '20240101' LIMIT 100"
        elif not is_final_step:
            norm_final = re.sub(r"\s+", " ", final_sql.strip()).lower()
            norm_sql = re.sub(r"\s+", " ", sql.strip()).lower()
            if norm_sql == norm_final:
                scope_prompt = SCOPE_REVISION_PROMPT.format(step_desc=step.desc, current_sql=sql)
                dialogues.append(
                    {
                        "role": "user",
                        "content": scope_prompt,
                        "meta": {
                            "prompt_type": "SCOPE_REVISION_PROMPT",
                        },
                    }
                )
                correction = self._chat_completion(scope_prompt, self.args.sqlgen_temperature, 8192) if self.llm_client else ""
                dialogues.append(
                    {
                        "role": "assistant",
                        "content": correction,
                        "meta": {"prompt_type": "SCOPE_REVISION_PROMPT"},
                    }
                )
                sql = (safe_json_loads(correction) or {}).get("sql", sql)

        dialogues.append(
            {
                "role": "assistant",
                "content": response,
                "meta": {
                    "prompt_type": "STEP_SQL_ONE_PROMPT",
                    "parsed_sql": sql,
                },
            }
        )
        return sql

    def _revise_sql(
        self,
        step: StepPlan,
        failed_sql: str,
        error_msg: Optional[str],
        dialogues: List[Dict[str, Any]],
        context: Dict[str, str],
    ) -> str:
        if self.llm_client is None:
            return failed_sql
        prompt = REVISION_PROMPT.format(
            step_desc=step.desc,
            failed_sql=failed_sql,
            error_msg=error_msg or "Execution error",
            ref_text=context.get("ref_text", ""),
        )
        # 记录完整的 SQL 自修正 prompt
        dialogues.append(
            {
                "role": "user",
                "content": prompt,
                "meta": {
                    "prompt_type": "REVISION_PROMPT",
                    "error_msg": error_msg,
                },
            }
        )
        response = self._chat_completion(prompt, self.args.revise_temperature, 8192)
        revised = (safe_json_loads(response) or {}).get("revised_sql", failed_sql)
        dialogues.append(
            {
                "role": "assistant",
                "content": response,
                "meta": {
                    "prompt_type": "REVISION_PROMPT",
                    "parsed_sql": revised,
                },
            }
        )
        self._log_correction(step, failed_sql, revised, error_msg)
        return revised

    def _log_correction(self, step: StepPlan, prev_sql: str, revised_sql: str, error: Optional[str]):
        path = getattr(self.args, "corrections_file", "")
        if not path:
            return
        payload = {
            "step_desc": step.desc,
            "prev_sql": prev_sql,
            "revised_sql": revised_sql,
            "error": error,
            "timestamp": int(time.time()),
            "model": self.args.model_used,
            "endpoint_type": self.args.endpoint_type_resolved,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


# ---------------------------------------------------------------------------
# Streaming writer
# ---------------------------------------------------------------------------

class StreamJsonArrayWriter:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self._fh = None
        self._first = True

    def __enter__(self):
        Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and has content
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            # Remove the last ']' if present to append
            with open(self.file_path, "rb+") as f:
                f.seek(0, os.SEEK_END)
                # Scan backwards for ']'
                pos = f.tell()
                found = False
                # Scan last 128 bytes
                scan_len = min(pos, 128)
                f.seek(-scan_len, os.SEEK_END)
                tail = f.read(scan_len)
                
                # Find last ']'
                last_bracket = tail.rfind(b']')
                if last_bracket != -1:
                    # Calculate absolute position
                    truncate_pos = pos - scan_len + last_bracket
                    f.seek(truncate_pos)
                    f.truncate()
                    found = True
                    
                    # Check if the file is now effectively empty (just '[')
                    # This is a simple heuristic
                    if truncate_pos <= 5: # e.g. "[\n"
                        f.seek(0)
                        content = f.read().strip()
                        if content == b'[':
                            self._first = True
                        else:
                            self._first = False
                    else:
                        self._first = False
                else:
                    # No closing bracket found, assume appending to list
                    self._first = False
            
            self._fh = open(self.file_path, "a", encoding="utf-8")
        else:
            self._fh = open(self.file_path, "w", encoding="utf-8")
            self._fh.write("[\n")
            self._first = True
        
        return self

    def write(self, obj: Dict[str, Any]):
        if not self._fh:
            raise RuntimeError("writer not opened")
        if not self._first:
            self._fh.write(",\n")
        else:
            self._first = False
        self._fh.write(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default))
        self._fh.flush()

    def __exit__(self, exc_type, exc, tb):
        if self._fh:
            self._fh.write("\n]\n")
            self._fh.close()
            self._fh = None


# ---------------------------------------------------------------------------
# Augmentation agent orchestrator
# ---------------------------------------------------------------------------

class AugmentationAgent:
    def __init__(self, args):
        self.args = args
        llm_client = None if args.offline else LLMClient(args.llm_model, args.decompose_temperature)
        self.planner = StepPlanner(llm_client)
        self.executor = StepExecutor(llm_client, SQLExecutionTool(), args)

        if args.offline:
            self.reflection_client = None
            self.detection_client = None
        else:
            ref_model = args.reflection_model if args.reflection_model else args.llm_model
            if ref_model == args.llm_model:
                self.reflection_client = llm_client
            else:
                self.reflection_client = LLMClient(ref_model, default_temperature=0.1)

            detection_model = args.detection_model if args.detection_model else args.llm_model
            if detection_model == args.llm_model:
                self.detection_client = llm_client
            elif detection_model == ref_model and self.reflection_client is not None:
                self.detection_client = self.reflection_client
            else:
                self.detection_client = LLMClient(detection_model, default_temperature=0.0)

    def process_dataset(self, records: List[Dict[str, Any]]):
        total = 0
        with StreamJsonArrayWriter(self.args.output_file) as writer:
            for idx, record in enumerate(records):
                variants = self._process_record(record)
                for variant in variants:
                    writer.write(variant)
                    total += 1
                print(f"[augment] processed sample {idx + 1}/{len(records)} with {len(variants)} variants")
        print(f"[augment] total variants: {total}")

    def _process_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        instruction = record.get("instruction") or record.get("question") or ""
        context_raw = parse_cn_prompt_blocks(record.get("input", ""))
        if not context_raw.get("user_question"):
            context_raw["user_question"] = instruction
        final_sql = record.get("sql") or record.get("output") or ""
        if not final_sql or not context_raw.get("user_question"):
            return []
        context_raw["final_sql"] = final_sql

        variants: List[Dict[str, Any]] = []
        for vidx in range(self.args.variants_per_question):
            steps = self.planner.plan(
                context_raw,
                self.args.min_steps,
                self.args.max_steps,
                final_sql,
                self.args.offline,
            )
            if not steps:
                print("[augment] planner returned empty steps, skipping variant")
                continue
            n_steps = len(steps)
            dialogues: List[Dict[str, Any]] = []
            self._append_note_dialogue(
                dialogues,
                f"endpoint={self.args.endpoint_type_resolved}, model={self.args.model_used}, base_url={self.args.openai_base_url}",
            )

            # 记录执行代理的系统提示与上下文种子，确保对话日志完整
            dialogues.append({"role": "system", "content": EXECUTOR_SYSTEM_PROMPT})
            seed_for_log = self.executor._compose_context_seed(context_raw)
            if seed_for_log:
                dialogues.append({"role": "system", "content": seed_for_log, "meta": {"type": "context_seed"}})

            self.executor.reset_history(context_raw)

            validated_sqls: List[str] = []
            step_results: List[StepSQLResult] = []
            print("step", steps)
            for idx_step, step in enumerate(steps):
                res = self.executor.run_step(
                    step,
                    context_raw,
                    final_sql,
                    is_final_step=self._is_last_non_meta(steps, idx_step),
                    validated_sqls=validated_sqls,
                    dialogues=dialogues,
                    variant_id=f"{record.get('question_id', 'unknown')}-v{vidx}",
                )
                step_results.append(res)
                validated_sqls.append(res.sql)

            # Reflection phase: use ground truth to refine steps
            # 每一轮基于上一轮的 step_results，形成串联的反思改写链
            for r_idx in range(self.args.reflection_rounds):
                step_results = self._reflect_and_refine(context_raw, steps, step_results, final_sql, dialogues, r_idx)
            validated_sqls = [r.sql for r in step_results]

            final_sql_generated = self._synthesize_final(context_raw, steps, validated_sqls, final_sql, dialogues)
            final_sql_match, final_sql_reason = self._evaluate_final_sql(
                context_raw,
                final_sql_generated,
                final_sql,
                dialogues,
            )
            step_sql_versions = [
                {
                    "step_index": idx,
                    "versions": [entry.copy() for entry in res.versions],
                }
                for idx, res in enumerate(step_results)
            ]

            variant = {
                "original_id": record.get("question_id"),
                "variant_id": f"{record.get('question_id', 'unknown')}-v{vidx}",
                "question": context_raw.get("user_question", ""),
                "schema": context_raw.get("schema_text", ""),
                "reference": context_raw.get("ref_text", ""),
                "final_sql": final_sql,
                "step_plan": [step.__dict__ for step in steps],
                "step_sqls": [res.sql for res in step_results],
                "step_exec_status": [res.status for res in step_results],
                "step_exec_errors": [res.error for res in step_results],
                "step_rows_sample": [res.rows_sample for res in step_results],
                "step_sql_versions": step_sql_versions,
                "dialogues": dialogues,
                "dialogues_history_clean": self._dialogues_history_view(dialogues),
                "verification": {
                    "all_steps_success": all(r.status in {"success", "meta"} for r in step_results),
                    "failed_steps": [idx for idx, r in enumerate(step_results) if r.status == "error"],
                    # 标记哪些步骤之间的SQL在归一化后是重复的，方便后处理过滤或重跑
                    "duplicate_step_sql_pairs": [
                        [i, j]
                        for i in range(len(step_results))
                        for j in range(i + 1, len(step_results))
                        if self._normalize_sql(step_results[i].sql) == self._normalize_sql(step_results[j].sql)
                    ],
                },
                "n_steps": n_steps,
                "final_sql_generated": final_sql_generated,
                "final_sql_match": final_sql_match,
                "final_sql_match_reason": final_sql_reason,
                "llm_model": self.args.model_used,
                "endpoint_type": self.args.endpoint_type_resolved,
                "openai_base_url": self.args.openai_base_url,
            }
            variants.append(variant)
        return variants

    def _reflect_and_refine(
        self,
        context: Dict[str, str],
        steps: List[StepPlan],
        step_results: List[StepSQLResult],
        final_sql: str,
        dialogues: List[Dict[str, Any]],
        round_idx: int = 0
    ) -> List[StepSQLResult]:
        if self.args.offline or self.reflection_client is None:
            return step_results

        print(f"[augment] starting reflection phase round {round_idx + 1} (step-by-step)...")
        new_results = []
        # 预先构造所有步骤及SQL，作为反思提示的一部分，帮助模型减少不同step之间的完全重复
        all_steps_bundle = json.dumps(
            {
                "steps": [s.desc for s in steps],
                "step_sqls": [r.sql for r in step_results],
            },
            ensure_ascii=False,
        )
        
        for idx, (step, res) in enumerate(zip(steps, step_results)):
            if step.meta:
                new_results.append(res)
                continue
            
            reflection_history: List[Dict[str, str]] = [
                {"role": "system", "content": REFLECTION_SYSTEM_PROMPT}
            ]
            dialogues.append(
                {
                    "role": "system",
                    "content": REFLECTION_SYSTEM_PROMPT,
                    "meta": {
                        "prompt_type": "REFLECTION_SYSTEM_PROMPT",
                        "round": round_idx + 1,
                        "step_index": idx,
                    },
                }
            )
            seed = self._compose_reflection_seed(context, final_sql, all_steps_bundle)
            if seed:
                reflection_history.append({"role": "user", "content": seed})
                dialogues.append(
                    {
                        "role": "system",
                        "content": seed,
                        "meta": {
                            "prompt_type": "REFLECTION_CONTEXT_SEED",
                            "round": round_idx + 1,
                            "step_index": idx,
                        },
                    }
                )

            # Prompt for this step（带上标准答案 SQL + 所有步骤SQL，仅用于评估/修正当前子 SQL，不直接替换）
            prompt = REFLECTION_STEP_PROMPT.format(
                step_idx=idx,
                step_desc=step.desc,
                current_sql=res.sql,
            )

            dialogues.append(
                {
                    "role": "user",
                    "content": prompt,
                    "meta": {
                        "prompt_type": "REFLECTION_STEP_PROMPT",
                        "round": round_idx + 1,
                        "step_index": idx,
                    },
                }
            )

            try:
                reflection_history.append({"role": "user", "content": prompt})
                response = self.reflection_client.invoke(
                    temperature=0.1,
                    max_tokens=2048,
                    messages=[msg.copy() for msg in reflection_history],
                )
                reflection_history.append({"role": "assistant", "content": response})
                dialogues.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "meta": {
                            "prompt_type": "REFLECTION_STEP_PROMPT",
                            "round": round_idx + 1,
                            "step_index": idx,
                        },
                    }
                )
                
                data = safe_json_loads(response)
                if data and not data.get("is_correct") and "revised_sql" in data:
                    diagnosis_reason = data.get("reason")
                    # 在单轮反思中，允许多次修正+执行尝试，受 max_step_retries 控制
                    max_try = max(1, getattr(self.args, "max_step_retries", 2))
                    base_attempts = res.attempts.copy()
                    current_sql = res.sql
                    last_error = res.error
                    exec_result = {"status": res.status, "error": res.error, "rows_sample": res.rows_sample}

                    for local_try in range(1, max_try + 1):
                        revised_sql = data.get("revised_sql", "").strip() or current_sql
                        # 若修正后与当前 SQL 完全一致，则不再进入无意义循环
                        if not revised_sql or self._normalize_sql(revised_sql) == self._normalize_sql(current_sql):
                            break

                        print(f"[augment] Step {idx} revised by reflection (round {round_idx+1}, try {local_try}).")
                        current_sql = revised_sql
                        exec_result = self.executor.sql_tool.run(current_sql)

                        base_attempts.append({
                            "attempt": f"reflection-r{round_idx+1}-t{local_try}",
                            "sql": current_sql,
                            "status": exec_result["status"],
                            "error": exec_result.get("error"),
                            "rows_sample": exec_result.get("rows_sample"),
                        })

                        if exec_result["status"] == "success":
                            last_error = None
                            break

                        last_error = exec_result.get("error")
                        if local_try >= max_try:
                            break

                        # 基于最新错误再请求一次反思模型，保持在同一轮对话语境下连续修正
                        follow_prompt = REFLECTION_STEP_PROMPT.format(
                            user_question=context.get("user_question", ""),
                            schema_text=context.get("schema_text", ""),
                            final_sql=final_sql,
                            all_steps_and_sqls=all_steps_bundle,
                            step_idx=idx,
                            step_desc=step.desc,
                            current_sql=current_sql,
                        )
                        supplementary: List[str] = []
                        if diagnosis_reason:
                            supplementary.append(f"上一轮诊断：{diagnosis_reason}")
                        if last_error:
                            supplementary.append(f"最新执行错误：{last_error}")
                        if supplementary:
                            follow_prompt = f"{follow_prompt}\n【补充线索】\n" + "\n".join(supplementary)
                        dialogues.append(
                            {
                                "role": "user",
                                "content": follow_prompt,
                                "meta": {
                                    "prompt_type": "REFLECTION_STEP_PROMPT_CONTINUE",
                                    "round": round_idx + 1,
                                    "step_index": idx,
                                    "try": local_try + 1,
                                    "last_error": last_error,
                                },
                            }
                        )
                        reflection_history.append({"role": "user", "content": follow_prompt})
                        follow_resp = self.reflection_client.invoke(
                            temperature=0.1,
                            max_tokens=2048,
                            messages=[msg.copy() for msg in reflection_history],
                        )
                        reflection_history.append({"role": "assistant", "content": follow_resp})
                        dialogues.append(
                            {
                                "role": "assistant",
                                "content": follow_resp,
                                "meta": {
                                    "prompt_type": "REFLECTION_STEP_PROMPT_CONTINUE",
                                    "round": round_idx + 1,
                                    "step_index": idx,
                                    "try": local_try + 1,
                                },
                            }
                        )
                        data = safe_json_loads(follow_resp) or {}
                        diagnosis_reason = data.get("reason")

                    version_history = [entry.copy() for entry in res.versions]
                    if exec_result["status"] == "success":
                        version_history.append(
                            {
                                "round": round_idx + 1,
                                "sql": current_sql,
                                "status": "success",
                                "rows_sample": exec_result.get("rows_sample", []),
                                "error": None,
                                "reason": diagnosis_reason,
                            }
                        )

                    new_res = StepSQLResult(
                        plan=res.plan,
                        sql=current_sql,
                        status=exec_result["status"],
                        error=last_error,
                        rows_sample=exec_result.get("rows_sample", []),
                        attempts=base_attempts,
                        versions=version_history,
                    )
                    new_results.append(new_res)
                else:
                    new_results.append(res)

            except Exception as e:
                print(f"[augment] Reflection error at step {idx}: {e}")
                new_results.append(res)
                
        return new_results

    @staticmethod
    def _is_last_non_meta(steps: List[StepPlan], idx: int) -> bool:
        non_meta_indices = [i for i, step in enumerate(steps) if not step.meta]
        return bool(non_meta_indices) and idx == non_meta_indices[-1]

    def _synthesize_final(self, context: Dict[str, str], steps: List[StepPlan], validated_sqls: List[str], fallback_final: str, dialogues: List[Dict[str, Any]]) -> str:
        if self.args.offline or self.executor.llm_client is None:
            return fallback_final
        bundle = json.dumps(
            {
                "steps": [step.__dict__ for step in steps],
                "step_sqls": validated_sqls,
            },
            ensure_ascii=False,
        )
        prompt = FINAL_SYNTHESIS_PROMPT.format(
            user_question=context.get("user_question", ""),
            schema_text=context.get("schema_text", ""),
            ref_text=context.get("ref_text", ""),
            steps_and_sqls=bundle,
        )
        dialogues.append(
            {
                "role": "user",
                "content": prompt,
                "meta": {"prompt_type": "FINAL_SYNTHESIS_PROMPT"},
            }
        )
        response = self.executor.llm_client.invoke(prompt=prompt, temperature=self.args.sqlgen_temperature, max_tokens=1024)
        dialogues.append(
            {
                "role": "assistant",
                "content": response,
                "meta": {"prompt_type": "FINAL_SYNTHESIS_PROMPT"},
            }
        )
        return (safe_json_loads(response) or {}).get("final_sql", fallback_final)

    def _evaluate_final_sql(
        self,
        context: Dict[str, str],
        generated_sql: str,
        reference_sql: str,
        dialogues: List[Dict[str, Any]],
    ) -> Tuple[bool, Optional[str]]:
        if not generated_sql or not reference_sql:
            return False, "missing_sql"
        if self.args.offline or self.detection_client is None:
            fallback_match = self._normalize_sql(generated_sql) == self._normalize_sql(reference_sql)
            reason = "offline_fallback"
            return fallback_match, reason

        prompt = DETECTION_FINAL_STEP_PROMPT.format(
            step_desc=context.get("user_question", ""),
            schema_text=context.get("schema_text", ""),
            ref_text=context.get("ref_text", ""),
            final_generated_sql=generated_sql,
            original_final_sql=reference_sql,
        )
        dialogues.append(
            {
                "role": "user",
                "content": prompt,
                "meta": {"prompt_type": "FINAL_SQL_DETECTION_PROMPT"},
            }
        )

        detection_error: Optional[str] = None
        try:
            response = self.detection_client.invoke(prompt=prompt, temperature=0.0, max_tokens=512)
            dialogues.append(
                {
                    "role": "assistant",
                    "content": response,
                    "meta": {"prompt_type": "FINAL_SQL_DETECTION_PROMPT"},
                }
            )
            parsed = safe_json_loads(response) or {}
            verdict = parsed.get("are_equivalent")
            if isinstance(verdict, bool):
                return verdict, parsed.get("reason")
        except Exception as exc:
            detection_error = str(exc)
            self._append_note_dialogue(
                dialogues,
                json.dumps({"detection_error": detection_error}, ensure_ascii=False),
                meta={"prompt_type": "FINAL_SQL_DETECTION_ERROR"},
            )

        fallback_match = self._normalize_sql(generated_sql) == self._normalize_sql(reference_sql)
        fallback_reason = f"detection_failed:{detection_error}" if detection_error else "detection_unparsed"
        return fallback_match, fallback_reason

    @staticmethod
    def _append_note_dialogue(dialogues: List[Dict[str, Any]], content: str, meta: Optional[Dict[str, Any]] = None) -> None:
        entry_meta = {"exclude_from_history": True}
        if meta:
            entry_meta.update(meta)
        dialogues.append({
            "role": "note",
            "content": content,
            "meta": entry_meta,
        })

    @staticmethod
    def _dialogues_history_view(dialogues: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        allowed_roles = {"system", "user", "assistant"}
        filtered: List[Dict[str, str]] = []
        for entry in dialogues:
            meta = entry.get("meta") or {}
            if meta.get("exclude_from_history"):
                continue
            role = entry.get("role")
            if role in allowed_roles:
                filtered.append({"role": role, "content": entry.get("content", "")})
        return filtered

    @staticmethod
    def _normalize_sql(sql: str) -> str:
        return re.sub(r"\s+", " ", (sql or "").strip()).lower()

    @staticmethod
    def _compose_reflection_seed(context: Dict[str, str], final_sql: str, all_steps_bundle: str) -> str:
        sections = []
        if context.get("user_question"):
            sections.append(f"【用户问题】\n{context['user_question']}")
        if context.get("schema_text"):
            sections.append(f"【数据库schema】\n{context['schema_text']}")
        if final_sql:
            sections.append(f"【标准答案SQL】\n{final_sql}")
        if context.get("ref_text"):
            sections.append(f"【参考信息】\n{context['ref_text']}")
        if all_steps_bundle:
            sections.append(f"【所有步骤及SQL】\n{all_steps_bundle}")
        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def load_input_dataset(path: str, limit: int) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        records = [{"question_id": int(k), "question": "", "sql": v} for k, v in data.items()]
    else:
        records = data
    if limit != -1:
        records = records[:limit]
    return records


def resolve_endpoint(args) -> None:
    base_url = args.openai_base_url or os.getenv("OPENAI_BASE_URL", "")
    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY", "")
    if args.openai_base_url:
        os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    if args.openai_api_key:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key

    parsed = urlparse(base_url) if base_url else None
    inferred = "unknown"
    if parsed and parsed.hostname:
        inferred = "vllm" if parsed.hostname in {"127.0.0.1", "localhost"} else "online"
    if args.offline:
        inferred = "offline"

    args.endpoint_type_resolved = args.endpoint_type if args.endpoint_type != "auto" else inferred
    args.model_used = "offline-rules" if args.offline else args.llm_model
    args.openai_base_url = base_url
    args.openai_api_key = api_key

    print(f"[augment] endpoint_type={args.endpoint_type_resolved}, model={args.model_used}, base_url={base_url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--variants_per_question", type=int, default=1)
    parser.add_argument("--max_step_retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--llm_model", default="qwen3-coder-480b-a35b-instruct")
    parser.add_argument("--decompose_temperature", type=float, default=0.2)
    parser.add_argument("--sqlgen_temperature", type=float, default=0.3)
    parser.add_argument("--revise_temperature", type=float, default=0.1)
    parser.add_argument("--reflection_rounds", type=int, default=0)
    parser.add_argument("--reflection_model", default="gpt-5.1-thinking")
    parser.add_argument("--detection_model", default="gpt-5.1-thinking")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--endpoint_type", choices=["auto", "online", "vllm", "offline"], default="auto")
    parser.add_argument("--corrections_file", default="corrections_log.jsonl")
    parser.add_argument("--openai_base_url", default="https://api.gptbest.vip/v1")
    parser.add_argument("--openai_api_key", default="sk-GMYNUCidV96DStXskUpPqgemoaDur0alDXZkeyiq5E3mXGZn")
    args = parser.parse_args()

    resolve_endpoint(args)
    records = load_input_dataset(args.input_file, args.limit)
    agent = AugmentationAgent(args)
    agent.process_dataset(records)
    DEFAULT_COST_RECORDER.print_profile()


if __name__ == "__main__":
    main()
