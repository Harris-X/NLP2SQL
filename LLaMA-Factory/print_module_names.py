
import torch
from transformers import AutoModelForCausalLM

model_name_or_path = "/root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct"

print(f"Loading model: {model_name_or_path}")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype=torch.bfloat16
    )

    print("\\n--- Model Named Modules ---")
    for name, module in model.named_modules():
        print(name)
    print("--- End of Modules ---\\n")

except Exception as e:
    print(f"An error occurred: {e}")

