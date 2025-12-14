#!/usr/bin/env python3
"""Generate DPO preference data for BIRD text2sql using Qwen3-Coder-30B-A3B-Instruct + LoRA.

Input:
  - Alpaca SFT data: data/bird_mini_text2sql_alpaca.json
  - Base model: Qwen3-Coder-30B-A3B-Instruct
  - (Optional) SFT LoRA adapter: saves/qwen3-coder-30b/lora/bird_sft

Output:
  - DPO data: data/bird_mini_text2sql_dpo.jsonl (alpaca preference format)

Format (per line JSON):
  {
    "instruction": "...",
    "input": "...",
    "chosen": "gold SQL (output)",
    "rejected": "model prediction SQL"
  }

Here we treat gold SQL as chosen (preferred), model prediction as rejected.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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


@torch.no_grad()
def generate_sql(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.1,
    top_p: float = 0.7,
    num_return_sequences: int = 1,
) -> List[str]:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        pad_token_id=tokenizer.eos_token_id,
        num_return_sequences=max(1, num_return_sequences),
    )

    input_len = inputs["input_ids"].shape[1]
    results: List[str] = []
    for seq in outputs:
        generated_ids = seq[input_len:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        results.append(text.strip())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BIRD DPO data from SFT dataset and model predictions.")
    parser.add_argument(
        "--base-model",
        type=str,
        default="/root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct",
        help="Path to base Qwen3-Coder-30B-A3B-Instruct model.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="", # /root/autodl-tmp/comp/LLaMA-Factory/saves/qwen3-coder-30b/lora/bird_32_sft
        help="Path to LoRA adapter (optional). If empty, use base model only.",
    )
    parser.add_argument(
        "--sft-data",
        type=str,
        default="/root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets.json",
        help="Path to Alpaca SFT dataset used for BIRD text2sql.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets_1214.json",
        help="Where to write DPO JSONL.",
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
        default=0.5,
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
        help="Device to run generation on (e.g., cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--num-rejects",
        type=int,
        default=10,
        help="How many rejected SQL candidates to generate per sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for generation.",
    )
    parser.add_argument(
        "--dedup-rejects",
        action="store_true",
        help="If set, skip writing duplicate rejected SQL candidates within the same sample.",
    )
    args = parser.parse_args()

    sft_path = Path(args.sft_data)
    data = load_alpaca(sft_path, args.max_samples)
    print(f"Loaded {len(data)} SFT samples from {sft_path}.")

    print(f"Loading base model from {args.base_model}...")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if args.device.startswith("cuda") else None,
        trust_remote_code=True,
    )

    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if adapter_path.exists():
            print(f"Loading LoRA adapter from {adapter_path}...")
            model = PeftModel.from_pretrained(model, adapter_path)
            model.eval()
        else:
            print(f"Warning: adapter path {adapter_path} does not exist, skip loading adapter.")

    if not args.device.startswith("cuda"):
        model.to(args.device)
    
    model.eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write("[\n")
        first_record = True
        total = len(data)
        for idx, sample in tqdm(enumerate(data, start=1), total=total):
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            gold_sql = sample.get("output", "").strip()

            if not gold_sql:
                continue

            prompt = build_prompt(instruction, input_text)
            seen_rejects = set()
            num_to_generate = max(1, args.num_rejects)
            seed_value = args.seed + idx * 1000
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)

            preds = generate_sql(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=num_to_generate,
            )[:num_to_generate]

            for ridx, pred_sql in tqdm(enumerate(preds), total=len(preds), leave=False):
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

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(data)} samples, written {num_written} preference pairs.")

        fout.write("\n]\n")
        fout.flush()

    print(f"Done. Wrote {num_written} DPO samples to {out_path}.")


if __name__ == "__main__":
    main()
