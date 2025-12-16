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
from typing import List, Dict, Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.hparams.data_args import DataArguments


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
    temperature: float = 0.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
) -> str:
    """使用 LLaMA-Factory 挂载过的 chat template 构造输入。

    get_template_and_fix_tokenizer 已经根据 `template="qwen3_nothink"`
    修正了 tokenizer.chat_template，这里直接用 apply_chat_template
    与训练时保持一致的格式，然后再送入 generate。
    """

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 显式提供 attention_mask，避免 pad==eos 时推理出错
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

    do_sample = temperature > 0
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.0,
        top_p=top_p if do_sample else 1.0,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][len(inputs["input_ids"][0]) :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


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
        default="/root/autodl-tmp/comp/LLaMA-Factory/saves/qwen3-coder-30b/lora/dpo_32_sft",
        help="Path to LoRA adapter (optional). If empty, use base model only.",
    )
    parser.add_argument(
        "--sft-data",
        type=str,
        default="/root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets_cot_sft.json",
        help="Path to Alpaca SFT dataset used for BIRD text2sql.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/root/autodl-tmp/comp/LLaMA-Factory/mydata/test_dpo_datasets_cot_sft.json",
        help="Where to write DPO JSONL.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
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
        default=0.0,
        help="Sampling temperature for generation (0 for greedy).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p for nucleus sampling.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (1.0 means no penalty).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run generation on (e.g., cuda, cuda:0, cpu).",
    )
    args = parser.parse_args()

    sft_path = Path(args.sft_data)
    data = load_alpaca(sft_path, args.max_samples)
    print(f"Loaded {len(data)} SFT samples from {sft_path}.")

    print(f"Loading base model from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # 使用与训练完全一致的 template 设置（qwen3_nothink）
    data_args = DataArguments(template="qwen3_nothink")
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

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

    # 使用行缓冲，便于实时写入
    with out_path.open("w", encoding="utf-8", buffering=1) as fout:
        for idx, sample in enumerate(data, start=1):
            instruction = sample.get("instruction", "")
            input_text = sample.get("input", "")
            gold_sql = sample.get("output", "").strip()

            if not gold_sql:
                continue

            prompt = build_prompt(instruction, input_text)
            pred_sql = generate_sql(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

            # 构造偏好数据：gold 作为 chosen，model 作为 rejected
            record = {
                "instruction": instruction,
                "input": input_text,
                "chosen": gold_sql,
                "rejected": pred_sql,
            }
            # 美化 JSON 输出：使用缩进，使每个键值对单独占一行
            json_str = json.dumps(record, ensure_ascii=False, indent=2)

            # 实时写入文件
            fout.write(json_str + "\n")
            fout.flush()

            # 同步打印到控制台
            print(json_str, flush=True)
            num_written += 1

            if idx % 10 == 0:
                print(f"Processed {idx}/{len(data)} samples, written {num_written} preference pairs.")

    print(f"Done. Wrote {num_written} DPO samples to {out_path}.")


if __name__ == "__main__":
    main()
