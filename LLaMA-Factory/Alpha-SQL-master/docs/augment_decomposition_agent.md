# Augment Decomposition Agent Flow

This note records the new agent-style implementation that mirrors the `spider-agent-tc` pipeline.

## Components

1. **LLMClient**
   - Thin wrapper over `call_openai` with configurable temperature and retry count.
   - Disabled entirely in `--offline` mode so the rest of the pipeline can still run with heuristics.

2. **StepPlanner**
   - Produces a meta step plus ordered sub-steps.
   - Prefers the `META_PLAN_PROMPT`; if the LLM fails or offline mode is enabled it falls back to `_heuristic_titles`, which inspects the final SQL to infer filters, joins, aggregations, etc.

3. **StepExecutor**
   - Treats database execution as a “tool" via `SQLExecutionTool.run`.
   - Sequentially generates per-step SQL, executes it, and—on failure—invokes the revision prompt before retrying (bounded by `--max_step_retries`).
   - Logs every attempt to the dialogue transcript and optionally appends corrections to `--corrections_file`.

4. **AugmentationAgent**
   - Streams dataset variants using `StreamJsonArrayWriter` so augmentation never buffers the full output.
   - For each source record, it creates `variants_per_question` runs with random step counts between `min_steps` and `max_steps`.
   - Keeps the system/user/tool/assistant trace so we can inspect or replay each conversation.
   - Synthesises a final SQL candidate from validated steps (or falls back to the ground-truth SQL when offline).

## Execution Flow

```
records = load_input_dataset(...)
resolve_endpoint(args)
agent = AugmentationAgent(args)
agent.process_dataset(records)
```

Each variant emits a record with:
- `step_plan`, `step_sqls`, execution status/errors
- `dialogues` (full trace)
- `verification` flags for quick filtering
- `final_sql_generated` + match boolean

This keeps the Alpha-SQL augmentation pipeline aligned with the spider-agent tool orchestration model.
