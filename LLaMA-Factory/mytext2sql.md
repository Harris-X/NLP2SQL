# Text2SQL 偏好数据流水线（DPO）

本文是本仓库生成与维护 Text2SQL DPO 偏好数据的唯一权威文档，流程或代码更新时请同步维护。

## 前置条件
- Python 环境已安装 `transformers`、`torch`、`peft`。
- 本地已下载基座模型权重（默认：`/root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct`）。
- 可选 LoRA 适配器路径（留空则不加载）。
- 输入 SFT/Alpaca 样式数据，字段包含 `instruction`、`input`、`output`（gold SQL）。
- 充足显存可加载 30B 模型；无 GPU 时可用 `--device cpu`（速度很慢）。

## 第 1 步：生成多拒答的 DPO 偏好数据
脚本：`step1_generate_rejected_data.py`

### 核心参数
- `--base-model`：基座模型路径。
- `--adapter-path`：LoRA 路径，留空则不用 LoRA。
- `--sft-data`：输入 SFT 数据集路径。
- `--output`：输出 JSON/JSONL 路径（本脚本写成 JSON 数组，含缩进）。
- `--num-rejects`：每条输入采样的拒答数量。
- `--dedup-rejects`：开启后，同一条样本内去重（精确匹配）。
- `--seed`：基准随机种子（按样本/批次派生）。
- `--max-new-tokens`：生成长度上限；OOM 或过慢时下调。
- `--temperature` / `--top-p`：采样多样性控制；想更分散可提高。
- `--max-samples`：限制处理条数，0 表示全量。
- `--device`：如 `cuda`、`cuda:0`；无 CUDA 回落 CPU。

### 示例命令
```bash
cd /root/autodl-tmp/comp/LLaMA-Factory
python step1_generate_rejected_data.py \
  --base-model /root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct \
  --adapter-path "/root/autodl-tmp/comp/LLaMA-Factory/saves/qwen3-coder-30b/lora/bird_32_sft" \
  --sft-data /root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets.json \
  --output /root/autodl-tmp/comp/LLaMA-Factory/mydata/dpo_datasets_multi.json \
  --num-rejects 3 \
  --dedup-rejects \
  --seed 1234 \
  --max-new-tokens 4096 \
  --temperature 0.7 \
  --top-p 0.9 \
  --device cuda
```

### 脚本行为
- 加载基座模型（如提供 LoRA 则一并加载），全局设定基准 seed。
- 对每条样本构造对话式 prompt（`instruction`+`input`）。
- 一次调用批量生成 `num_rejects` 条拒答；同一条样本的 seed 派生自 `seed + idx*1000`。
- 可选去重：同样本内完全相同的 SQL 会跳过。
- 以 JSON 数组格式写盘，每条记录含 `instruction`、`input`、`chosen`（gold SQL）、`rejected`（模型 SQL），并带缩进便于查看。

### 常见问题
- **显存/OOM**：调低 `--max-new-tokens`，或换小模型/低温度；确保 `device_map="auto"` 有足够余量。
- **LoRA 路径错误**：会警告并仅用基座模型继续。
- **无 GPU**：可用 `--device cpu`，但速度很慢；先用 `nvidia-smi` 确认可用卡。
- **去重仍有重复**：当前为精确匹配；如空白差异需额外归一化。
- **生成太慢**：调低 `--max-new-tokens`、`--num-rejects`，或降低温度。

### 对原始的输出通过大模型生成思维链来进行学习

- 运行`step2_generate_cot_dataset.py` 脚本

## 第 2 步：用 LLaMA Factory 进行 DPO 训练

### 数据格式
- 第 1 步输出为 JSON 数组，每个元素字段：`instruction`、`input`、`chosen`、`rejected`。确保文件路径在训练配置中可访问。

数据集的自定义配置：修改 data/dataset_info.json文件，添加如下内容：
```json
"my_sft": {
    "file_name": "my_sft.json",
    "formatting": "alpaca",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "my_dpo": {
    "file_name": "my_dpo.json",
    "ranking": true,
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  }
```

### 最小 DPO 配置示例（YAML）
案例：`examples/train_lora/llama3_lora_dpo.yaml`，将下面的YAML保存为 `configs/dpo_text2sql.yaml`：
```yaml
### model
model_name_or_path: /root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct
adapter_name_or_path: /path/to/your/lora/adapter
trust_remote_code: true
template: qwen3

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 32
lora_target: all
pref_beta: 0.1
pref_loss: sigmoid  # choices: [sigmoid (dpo), orpo, simpo]

### dataset
dataset: my_dpo
cutoff_len: 65536
max_samples: 1500  # 0 表示用完整数据集
overwrite_cache: true
preprocessing_num_workers: 2

### output
output_dir: saves/qwen3-coder-30b/lora/bird_dpo
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none  # choices: [none, wandb, tensorboard, swanlab, mlflow]

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 5.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

```

### 运行命令
```bash
cd /root/autodl-tmp/comp/LLaMA-Factory
llamafactory-cli train configs/dpo_text2sql.yaml
# 或使用 python -m 方式：
# python -m llamafactory.cli.train configs/dpo_text2sql.yaml
```

### 关键参数说明
- `model_name_or_path` / `adapter_name_or_path`：基座与可选 LoRA。
- `template`：需匹配模型（如 qwen）；决定 prompt 构造。
- `finetuning_type`：全参/LoRA/QLoRA 取决于显存与需求。
- `dataset.path`：指向第 1 步生成的偏好数据；`type: preference` 告知含 chosen/rejected。
- `dpo_beta`：越大对偏好差异更敏感；过大可能不稳定。
- `dpo_loss_type`：`sigmoid` 通常更稳；`dpo` 为原始公式。
- `max_seq_length` / `max_target_length`：根据 SQL 长度与显存调整。
- `per_device_train_batch_size` 与 `gradient_accumulation_steps`：共同决定有效 batch；显存不够时减小前者、增大后者。
- `save_steps`：频繁保存会占磁盘且耗时；酌情调大。

### 建议与排查
- 显存紧张：优先用 LoRA/QLoRA，减小 batch 和序列长度。
- 收敛慢或梯度不稳：尝试降低学习率或调低 `dpo_beta`。
- 数据质量：确保 `chosen` 是可靠 gold SQL，`rejected` 为模型生成的劣质/次优 SQL，避免反向信号噪声。
- 如果需要混合多数据源，可在 YAML 中加入更多 dataset 条目，或用权重采样（`sampling_strategy`）。

## 维护说明
- 若脚本或参数有改动，请同步更新本文件。
- 示例命令与默认值请保持与 `step1_generate_rejected_data.py`、DPO 配置一致。
- 新的已知问题或排查技巧请追加在相应小节。
