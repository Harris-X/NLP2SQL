# Qwen3-Coder-30B BIRD Text2SQL: SFT + DPO 全流程

本文记录如何在本仓库中，基于 BIRD mini text2sql 数据，对 `Qwen3-Coder-30B-A3B-Instruct` 进行：

1. 指令监督微调（SFT，LoRA）
2. 生成偏好数据（DPO 数据）
3. 基于偏好数据的 DPO 训练

并总结关键配置修改点与常见问题，方便复现与扩展。

---

## 1. 数据准备

### 1.1 SFT 数据（已完成）

- 源：BIRD mini-dev MySQL 数据集 + schema CSV
- 通过自定义脚本 `text2sql2.py` 生成 Alpaca 格式的 text2sql 数据：
  - 路径：`data/bird_mini_text2sql_alpaca.json`
  - 每条样本结构：
    ```json
    {
      "instruction": "通用 text2sql 提示（系统说明）",
      "input": "自然语言问题 + 各表列名信息",
      "output": "gold SQL"
    }
    ```

### 1.2 在 LLaMA-Factory 中注册数据集

文件：`data/dataset_info.json`

已添加：

```json
"bird_mini_text2sql": {
  "file_name": "bird_mini_text2sql_alpaca.json",
  "formatting": "alpaca",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
},
"bird_mini_text2sql_dpo": {
  "file_name": "bird_mini_text2sql_dpo.jsonl",
  "ranking": true,
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

说明：
- `bird_mini_text2sql`：用于 SFT 的 Alpaca 格式监督数据；`prompt/query/response` 分别对 `instruction/input/output`。
- `bird_mini_text2sql_dpo`：用于 DPO 的偏好数据；`ranking: true` 表示 pairwise 偏好。

更多格式说明参考：`data/README_zh.md` 中的 *Alpaca 格式* 与 *偏好数据集* 章节。

---

## 2. SFT（LoRA）训练

### 2.1 训练配置

文件：`examples/train_lora/qwen3_coder_30b_bird_sft.yaml`

关键字段（简要）：

```yaml
### model
model_name_or_path: /root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct
trust_remote_code: true
template: qwen3_nothink

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: q_proj,k_proj

### dataset
dataset: bird_mini_text2sql
cutoff_len: 65536
max_samples: 1000
preprocessing_num_workers: 2

### train
output_dir: saves/qwen3-coder-30b/lora/bird_sft
per_device_train_batch_size: 1
num_train_epochs: 3
learning_rate: 2e-4
bf16: true
gradient_checkpointing: true
```

### 2.2 启动 SFT 训练

```bash
cd /root/autodl-tmp/comp/LLaMA-Factory
conda activate qwen

llamafactory-cli train examples/train_lora/qwen3_coder_30b_bird_sft.yaml
```

训练完成后，SFT LoRA 权重保存在：

```text
saves/qwen3-coder-30b/lora/bird_sft/
```

---

## 3. DPO 偏好数据的“正确”构造方式

### 3.1 chosen / rejected 应该怎么选？

**最推荐、最稳妥的做法：**

- `chosen`：数据集中**已经存在的高质量 gold 答案**（例如 gold SQL）。
- `rejected`：
  - 来自当前 SFT 模型、旧模型或基座模型的预测结果；
  - 要求：比 gold 明显更差（错误或不完全正确）。

> 这样做的优点：
> - 训练目标清晰：模型被强烈拉向 gold，而远离自身或旧模型的错误习惯；
> - 稳定性好，避免“错误答案优于正确答案”的标签噪声；
> - 可以多轮迭代：SFT → DPO → 新模型 → 再造偏好数据。

你的场景（BIRD text2sql）：

- **chosen**：`bird_mini_text2sql_alpaca.json` 中的 `output`（gold SQL）。
- **rejected**：用当前模型（基座+SFT LoRA）在 `instruction+input` 上生成的 SQL 预测。

### 3.2 是否“必须”用自己微调出来的模型做 rejected？

不是必须，但**很推荐**：

- 如果完全用基座模型做 `rejected`：
  - 可以视作“让 post-training 模型优于 base”；
  - 但你其实已经用 SFT 适应了 text2sql，再用 SFT 模型来产生错误，更贴近模型当前的真实 error pattern。
- 你现在的做法：
  - 基座 + BIRD SFT → 一个初步 text2sql 能力的 LoRA；
  - 用这个 LoRA 生成错误 SQL 作为 `rejected`，gold 作为 `chosen`；
  - 再做 DPO，相当于在 gold 附近进行“对比式对齐”，是非常合理的。

---

## 4. 生成 BIRD DPO 数据

### 4.1 脚本位置与用途

文件：`scripts/generate_bird_dpo.py`

作用：

- 读取已有的 SFT 数据 `data/bird_mini_text2sql_alpaca.json`；
- 用 Qwen3-Coder-30B-A3B-Instruct + LoRA（`bird_sft`）对每条样本生成一个 SQL 预测；
- 将 gold SQL 作为 `chosen`，模型预测作为 `rejected`，写入 DPO JSONL：
  - `data/bird_mini_text2sql_dpo.jsonl`

输出格式（每行一个 JSON）：

```json
{
  "instruction": "...",
  "input": "...",
  "chosen": "gold SQL",
  "rejected": "model predicted SQL"
}
```

### 4.2 脚本参数

脚本核心参数（默认值已对齐当前路径）：

```bash
python scripts/generate_bird_dpo.py \
  --base-model /root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct \
  --adapter-path /root/autodl-tmp/comp/LLaMA-Factory/saves/qwen3-coder-30b/lora/bird_sft \
  --sft-data /root/autodl-tmp/comp/LLaMA-Factory/data/bird_mini_text2sql_alpaca.json \
  --output /root/autodl-tmp/comp/LLaMA-Factory/data/bird_mini_text2sql_dpo.jsonl \
  --max-samples 0 \
  --max-new-tokens 512 \
  --temperature 0.1 \
  --top-p 0.7 \
  --device cuda
```

注意：
- 若 `--adapter-path` 为空或路径不存在，则只使用基座模型作为生成方；
- `max-samples`=0 表示用完整 SFT 数据集（注意显存和时间压力，可先设为 200 做 smoke test）。

---

## 5. BIRD DPO 训练配置

### 5.1 DPO 训练 yaml

文件：`examples/train_lora/qwen3_coder_30b_bird_dpo.yaml`

核心内容：

```yaml
### model
model_name_or_path: /root/autodl-tmp/comp/LLaMA-Factory/Qwen3-Coder-30B-A3B-Instruct
trust_remote_code: true

### method
stage: dpo
do_train: true
finetuning_type: lora
lora_rank: 8
lora_target: q_proj,k_proj
pref_beta: 0.1
pref_loss: sigmoid  # DPO 默认

### dataset
dataset: bird_mini_text2sql_dpo
template: qwen3_nothink
cutoff_len: 65536
max_samples: 1000
overwrite_cache: true
preprocessing_num_workers: 2

### output
output_dir: saves/qwen3-coder-30b/lora/bird_dpo
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 1
learning_rate: 5.0e-6
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null
```

### 5.2 启动 DPO 训练

```bash
cd /root/autodl-tmp/comp/LLaMA-Factory
conda activate qwen

# 先确保已经生成 DPO 数据
python scripts/generate_bird_dpo.py --max-samples 500

# 然后启动 DPO 训练
llamafactory-cli train examples/train_lora/qwen3_coder_30b_bird_dpo.yaml
```

> 如果你希望在 DPO 的基础上继续沿用 SFT LoRA 权重，有两种做法：
> 1. 直接在 `qwen3_coder_30b_bird_dpo.yaml` 中设置 `resume_from_checkpoint: saves/qwen3-coder-30b/lora/bird_sft`；
> 2. 或者在命令行添加 `--resume_from_checkpoint` 指向 SFT 输出目录。
>
> 具体方式需结合当前 LLaMA-Factory 版本对 LoRA+resume 的支持情况，可按需要调整。

---

## 6. 常见问题与排坑

### 6.1 “Cannot find valid samples, check data/README.md”

原因（已解决）：
- `dataset_info.json` 中 BIRD SFT 数据集 `columns` 写成了：
  ```json
  "columns": {
    "instruction": "instruction",
    "input": "input",
    "output": "output"
  }
  ```
- 但内部的 Alpaca 转换器需要的是 `prompt/query/response` 字段；
- 导致 `_response` 为空，被视为“无有效样本”。

修复方式：

```json
"columns": {
  "prompt": "instruction",
  "query": "input",
  "response": "output"
}
```

### 6.2 DPO 数据列名不匹配

症状：
- DPO 阶段训练时报类似“找不到 chosen / rejected”等错误。

检查：
- `data/README_zh.md` 中偏好数据格式要求：
  ```json
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected"
  }
  ```
- 确保 `bird_mini_text2sql_dpo` 对应配置与之完全一致。

### 6.3 上下文长度和 OOM

- Qwen3-Coder-30B-A3B-Instruct 支持长上下文，但训练时：
  - 建议先将 `cutoff_len` 设为如 `8192` 或 `16384` 做尝试；
  - 若显存吃紧，可：
    - 减小 `per_device_train_batch_size`；
    - 提高 `gradient_accumulation_steps`；
    - 确认 `bf16: true` 且显卡支持。

### 6.4 生成的 rejected 与 chosen 太接近

- 若模型已经很强，可能部分预测与 gold 很接近甚至相同：
  - 可以在生成脚本中增加简单过滤逻辑，跳过 `pred_sql == gold_sql` 的样本；
  - 或按 SQL 规范化/去空格后再比较是否一样。

---

## 7. 总结与下一步

- SFT：用 BIRD text2sql 的 Alpaca 数据对 Qwen3-Coder-30B-A3B-Instruct 进行 LoRA 微调，建立基本 text2sql 能力；
- DPO 数据：
  - gold SQL 作为 `chosen`；
  - 当前模型预测 SQL 作为 `rejected`；
  - 使用 `scripts/generate_bird_dpo.py` 生成 `data/bird_mini_text2sql_dpo.jsonl`；
- DPO 训练：使用 `examples/train_lora/qwen3_coder_30b_bird_dpo.yaml`，在偏好信号上进一步对齐模型输出。

后续你可以：
- 在更大的 BIRD 数据上扩展 SFT 和 DPO；
- 将 DPO 后的模型接入你已有的 Spider2 / Snowflake 评测流水线，统一对比 SFT 前后、DPO 前后的执行正确率；
- 尝试多轮 DPO（用 DPO 后模型重新生成 rejected）。
