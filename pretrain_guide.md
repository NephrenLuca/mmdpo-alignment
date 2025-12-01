## 基于 Safe-RLHF 与 Safe-MM-DPO 的预训练与对齐步骤（运行手册）

本手册给出从 **拉取代码 → 准备环境与数据 → 训练评判员模型 → Safe-MM-DPO 对齐训练 →（可选）蒸馏数据准备** 的端到端可执行步骤，假设：

- **代码托管**在 GitHub 仓库 `git@github.com:<your_name>/humanAlignment.git`
- 最终运行于一台已安装好 GPU 驱动 / CUDA 的机器上，并可安装：
  - **Python 3.10.10**
  - **PyTorch 2.6.x**
- 基础模型为 `mistralai/Mistral-7B-v0.1`

如无特殊说明，下述命令均在 **目标训练机器** 上执行。

---

## 1. 拉取代码与创建运行环境

### 1.1 从 GitHub 拉取项目

```bash
# 在目标机器上选择合适的工作目录
cd /path/to/your/workspace

# 克隆项目代码
git clone git@github.com:<your_name>/humanAlignment.git
cd humanAlignment
```

> 若使用 HTTPS，可改为：
> ```bash
> git clone https://github.com/<your_name>/humanAlignment.git
> ```

### 1.2 创建并激活 Python 3.10.10 环境

推荐使用 conda（或系统自带 3.10.10 + venv），目标是让运行环境满足：

- Python：**3.10.10**

```bash
# 使用 conda（推荐）
conda create -n humanAlignment python=3.10
conda activate humanAlignment

# 如需严格指定 3.10.10，可在有该版本镜像的环境中：
# conda create -n humanAlignment python=3.10.10
```

若不使用 conda，也可以用 venv（前提是系统有 python3.10）：

```bash
# 示例：使用系统 python3.10 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate
```

### 1.3 安装 PyTorch 2.6 与核心依赖

#### 1.3.1 安装 PyTorch 2.6

目标版本：

- **torch==2.6.\***（与 Python 3.10.10 兼容）

根据是否需要 GPU 以及 CUDA 版本选择安装命令。以下为 **CPU 版本**示例：

```bash
pip install "torch==2.6.*" "torchvision" "torchaudio"
```

> 如需 GPU 版本，请前往 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/) 选择：
> - PyTorch Build: Stable (2.6)
> - Language: Python
> - Compute Platform: 对应的 CUDA 版本（如 CUDA 12.x）
>
> 然后复制页面给出的 `pip` 或 `conda` 命令执行。

#### 1.3.2 安装项目依赖

确保 `requirements.txt` 中 `torch==2.6.*`，然后：

```bash
pip install -r requirements.txt
```

### 1.4 验证环境

```bash
python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available())"
```

> 期望输出：
> - Python 版本为 3.10.x（最好是 3.10.10）
> - torch 版本为 2.6.x
> - `torch.cuda.is_available()` 在 GPU 环境中应为 `True`

---

## 2. 下载基础模型：Mistral-7B-v0.1

本项目使用 `mistralai/Mistral-7B-v0.1` 作为基础 Causal LM。

- **合理性**：
  - 7B 模型在能力与资源之间折中良好，适合作为 Safe-RLHF / DPO 对齐实验的基础模型；
  - 与 `transformers>=4.35.0`、**Python 3.10.10 + torch 2.6** 组合兼容；
  - 可与 LoRA / PEFT、量化等技术配合，以适应不同显存条件。

### 2.1 创建模型目录

```bash
mkdir -p models/base
```

### 2.2 使用 Hugging Face CLI 下载模型权重

```bash
pip install huggingface_hub

huggingface-cli download mistralai/Mistral-7B-v0.1 \
    --local-dir models/base/Mistral-7B-v0.1
```

### 2.3 使用 Python 代码下载（可选）

```bash
python - << 'PY'
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mistralai/Mistral-7B-v0.1"
cache_dir = "models/base"

print("Downloading model and tokenizer...")
tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
print("Done.")
PY
```

> **资源提示**：Mistral-7B 全参数加载对显存要求较高，建议至少单卡 40GB 或多卡环境；低显存场景建议后续采用 LoRA + 量化。

---

## 3. 准备 BeaverTails 偏好数据

本项目使用 Safe RLHF 相关仓库中的 `BeaverTails` 数据来构造双维度偏好对：

- 有帮助性（helpful）
- 无害性（harmless / safety）

### 3.1 获取 Safe-RLHF 数据仓库

```bash
cd /path/to/your/workspace/humanAlignment

mkdir -p data
cd data
git clone https://github.com/PKU-Alignment/safe-rlhf.git
cd ..
```

### 3.2 整理原始数据到 `data/raw`

具体数据文件路径需根据 `safe-rlhf` 仓库实际结构调整，以下为示意：

```bash
mkdir -p data/raw

# 示例：将 BeaverTails 相关数据复制到 raw 目录
cp data/safe-rlhf/path/to/beavertails_*.jsonl data/raw/
```

### 3.3 运行预处理脚本，生成偏好对与数据集划分

假设已实现 `src/scripts/prepare_data.py`，其职责包括：

- 读取 `data/raw` 中原始 BeaverTails 数据；
- 构造 `(prompt, chosen_response, rejected_response)` 结构的偏好对；
- 按 helpful / harmless 维度分别输出；
- 划分 `train / val / test` 集合。

示例命令（需根据实际脚本参数调整）：

```bash
python -m src.scripts.prepare_data \
    --input_dir data/raw \
    --output_dir data \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

生成的数据目录期望结构示例：

- `data/train/helpful_pairs.jsonl`
- `data/val/helpful_pairs.jsonl`
- `data/test/helpful_pairs.jsonl`
- `data/train/harmless_pairs.jsonl`
- `data/val/harmless_pairs.jsonl`
- `data/test/harmless_pairs.jsonl`

### 3.4 快速检查数据样例（可选）

```bash
python - << 'PY'
import json, os

sample_path = os.path.join("data", "train", "helpful_pairs.jsonl")
with open(sample_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        print(json.loads(line))
PY
```

---

## 4. 训练评判员模型：Helpful-RM 与 Harmless-RM

本阶段目标是训练两个独立的奖励模型：

- **Helpful-RM**：评估回答的“有帮助性”
- **Harmless-RM**：评估回答的“无害性”（或有害成本）

二者结构可以与基础模型类似（如在 Mistral-7B 上加一个标量输出头），但只用于计算奖励差值，不参与最终策略梯度。

### 4.1 准备奖励模型配置 `configs/rm_config.yaml`

配置文件中应包括：

- **基础模型与 tokenizer 路径**：
  - `base_model_path: models/base/Mistral-7B-v0.1`
- **数据路径**：
  - Helpful：`data/train/helpful_pairs.jsonl`、`data/val/helpful_pairs.jsonl`
  - Harmless：`data/train/harmless_pairs.jsonl`、`data/val/harmless_pairs.jsonl`
- **训练超参数**（参考 `dev.md` 第 5.1 小节）：
  - 学习率：`1e-5 ~ 5e-5`
  - batch size：根据显存调整（如 8–32）
  - epoch 数：1–3
  - `max_grad_norm` 等

### 4.2 训练 Helpful-RM

```bash
python -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task helpful \
    --output_dir models/helpful_rm
```

### 4.3 训练 Harmless-RM

```bash
python -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task harmless \
    --output_dir models/harmless_rm
```

### 4.4 验证奖励模型质量（可选）

可以在验证集上做简单统计：

- 计算在偏好对上 `score_chosen > score_rejected` 的比例；
- 查看损失收敛情况；
- 手动 inspect 部分样本的评分是否合理。

---

## 5. Safe-MM-DPO 对齐训练

本阶段是核心训练环节，使用 Helpful-RM 与 Harmless-RM 提供的奖励边际，结合 MM-DPO 与拉格朗日乘子，实现对“有帮助性”与“无害性”的联合对齐。

### 5.1 准备 DPO 配置 `configs/dpo_config.yaml`

建议配置包含以下关键字段（详见 `dev.md` 5.2.3）：

```yaml
λ_init: 1.0
w: 0.5
k: 0.5
β_ori: 0.1
learning_rate: 5e-7
batch_size: 16
kl_coeff: 0.1
λ_lr: 0.01
epochs: 2
gradient_accumulation_steps: 4
max_grad_norm: 1.0
warmup_steps: 100
save_steps: 500
eval_steps: 250

policy_model_path: models/base/Mistral-7B-v0.1
ref_model_path: models/base/Mistral-7B-v0.1
helpful_rm_path: models/helpful_rm
harmless_rm_path: models/harmless_rm

train_helpful_path: data/train/helpful_pairs.jsonl
train_harmless_path: data/train/harmless_pairs.jsonl
val_helpful_path: data/val/helpful_pairs.jsonl
val_harmless_path: data/val/harmless_pairs.jsonl
```

> 实际字段名需与 `src/training/train_safe_mm_dpo.py` 中的解析逻辑一致。

### 5.2 启动 Safe-MM-DPO 训练

```bash
python -m src.training.train_safe_mm_dpo \
    --config configs/dpo_config.yaml \
    --output_dir models/aligned \
    --logging_dir logs/training/safe_mm_dpo
```

### 5.3 监控训练过程

建议在训练脚本中记录以下指标，并通过 TensorBoard / W&B 进行可视化：

- **损失相关**：
  - `loss_total`
  - `loss_helpful`
  - `loss_safety`
  - `loss_kl`（如启用 KL 约束）
- **奖励相关**：
  - `reward_helpful_mean`
  - `reward_safety_mean`
  - `delta_helpful_mean`
  - `delta_safety_mean`
- **动态参数**：
  - `lambda`
  - `beta_helpful_mean`
  - `beta_safety_mean`
- **模型表现**：
  - `kl_divergence`
  - （可选）`perplexity`

### 5.4 检查点与恢复训练

- 建议在 `checkpoints/dpo_checkpoints/` 下定期保存检查点：
  - 模型权重
  - 优化器状态
  - 当前 epoch 与 step
  - 当前 `lambda` 值与配置
- 如训练中断，可通过：

```bash
python -m src.training.train_safe_mm_dpo \
    --config configs/dpo_config.yaml \
    --output_dir models/aligned \
    --logging_dir logs/training/safe_mm_dpo \
    --resume_from_checkpoint checkpoints/dpo_checkpoints/step_xxx
```

> 具体参数名称需与脚本实现保持一致。

---

## 6. （可选）为知识蒸馏生成数据

若后续希望将对齐后的 Mistral-7B 蒸馏到更小的模型（如 TinyLlama-1.1B），可以先使用对齐模型生成高质量回答，作为 dSFT 训练数据。

### 6.1 生成蒸馏数据

假设实现了脚本 `src/scripts/generate_distillation_data.py`，其功能为：

- 从给定提示集合中采样；
- 使用对齐后的模型 `models/aligned` 生成回答；
- 可选：用奖励模型过滤低分样本；
- 输出 `(prompt, response)` 形式的 JSONL 文件。

示例命令：

```bash
python -m src.scripts.generate_distillation_data \
    --model_path models/aligned \
    --input_prompts data/train/prompts_for_distill.jsonl \
    --output_path data/distill/aligned_responses.jsonl \
    --num_samples 50000
```

后续关于学生模型选择与 dSFT 训练流程，可参考 `dev.md` 第 7 章“知识蒸馏流程”。

---

## 7. 快速检查与常见问题

### 7.1 最小检查清单

在正式长时间训练前，建议先完成以下快速检查：

1. **环境**：
   - `python --version` 显示为 3.10.x
   - `python -c "import torch; print(torch.__version__)"` 显示 2.6.x
2. **模型加载**：
   - 能在脚本中成功 `from_pretrained("models/base/Mistral-7B-v0.1")`
3. **数据**：
   - `data/train/helpful_pairs.jsonl` / `harmless_pairs.jsonl` 不为空；
   - 随机打印几条，字段名与训练脚本预期一致。
4. **训练脚本**：
   - 分别用极小数据量（如几百条）跑 1–2 个 step，确保没有 OOM / shape 错误。

### 7.2 常见问题提示

- **显存不足（OOM）**：
  - 减小 `batch_size`；
  - 增大 `gradient_accumulation_steps`；
  - 启用 `gradient_checkpointing`；
  - 使用 4bit/8bit 量化 + LoRA。
- **损失不下降或训练不稳定**：
  - 降低 `learning_rate`；
  - 检查数据是否对齐（prompt、chosen、rejected 是否合理）；
  - 检查 RM 输出的奖励边际分布是否正常。
- **λ 值发散**：
  - 调小 `λ_lr`；
  - 检查有害性成本估计是否过于极端（可能是 Harmless-RM 输出异常）。

---

本运行手册与 `dev.md`、`demo.md` 相互补充：

- **`demo.md`**：聚焦于项目整体设计与技术路线；
- **`dev.md`**：提供模块设计与训练流程的高层说明；
- **`pretrain_guide.md`（本文件）**：面向“直接在目标机器上跑起来”的操作手册。


