# Text2SQL Preference Data Pipeline

This document is the single source of truth for generating and maintaining Text2SQL DPO preference data in this repo. Keep it updated when the workflow or code changes.

## Prerequisites
- Python environment with `transformers`, `torch`, and `peft` installed.
- Base model weights available locally (default: `/root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct`).
- Optional LoRA adapter path (can be empty to skip).
- SFT/Alpaca-style dataset with `instruction`, `input`, and `output` (gold SQL) fields.
- Sufficient GPU memory to load the base model (30B). Use `--device cpu` only if unavoidable (very slow).

## Step 1: Generate DPO Preference Data with Multiple Rejects
Script: `step1_generate_rejected_data.py`

### Key arguments
- `--base-model`: Path to the base model.
- `--adapter-path`: Path to LoRA adapter (leave empty to disable).
- `--sft-data`: Input SFT dataset path.
- `--output`: Output JSONL/JSON path for preference data.
- `--num-rejects`: Number of rejected SQLs to sample per input (default 1).
- `--dedup-rejects`: Skip duplicate rejects within a sample when set.
- `--seed`: Base RNG seed for reproducibility (applied per-sample/reject).
- `--max-new-tokens`: Generation cap; lower if you hit OOM or timeouts.
- `--temperature` / `--top-p`: Sampling controls; raise temperature/top-p for more diversity.
- `--max-samples`: Limit inputs (0 = use all).
- `--device`: `cuda`, `cuda:0`, etc. (falls back to CPU if not CUDA).

### Example command
```bash
cd /root/autodl-tmp/comp/LLaMA-Factory
python step1_generate_rejected_data.py \
  --base-model /root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct \
  --adapter-path "/root/autodl-tmp/comp/LLaMA-Factory/saves/qwen3-coder-30b/lora/bird_32_sft" \
  --sft-data /root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets.json \
  --output /root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets_multi.jsonl \
  --num-rejects 3 \
  --dedup-rejects \
  --seed 1234 \
  --max-new-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.9 \
  --device cuda
```

### What the script does
- Loads the base model (and LoRA if provided), sets the base seed once.
- For each sample, builds a chat-style prompt from `instruction` + `input`.
- Generates `num_rejects` predicted SQLs with per-reject seeds: `seed + sample_index*1000 + reject_index`.
- Optionally drops duplicate rejects (exact string match) within the same sample when `--dedup-rejects` is set.
- Writes one JSON line **per reject** with fields: `instruction`, `input`, `chosen` (gold SQL), `rejected` (model SQL).

### Common issues and fixes
- **OOM while loading/generating**: Reduce `--max-new-tokens`, increase `--top-p` modestly, or switch to a smaller model/LoRA. Ensure `device_map="auto"` has enough free VRAM.
- **Adapter missing**: If `--adapter-path` is wrong, the script will warn and continue with base model only.
- **No GPU available**: Set `--device cpu` (slow) or point to a CUDA device. Check `nvidia-smi` for free memory.
- **Duplicates still appear**: Dedupe is exact-string; SQL that differs by whitespace will be kept. Consider post-processing normalization if needed.
- **Long generations / timeouts**: Lower `--max-new-tokens` or temperature; use greedy decoding (`--temperature 0 --top-p 1`).

## Maintenance Notes
- When changing any script parameters or adding new steps, update this file.
- Keep examples in sync with defaults inside `step1_generate_rejected_data.py`.
- Record any new known issues or troubleshooting steps here.
