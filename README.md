## 基于 Safe-RLHF 与 Safe-MM-DPO 的语言模型价值对齐项目

本项目实现了一个围绕 **Mistral-7B-v0.1** 的完整价值对齐流程，结合 **Safe-RLHF** 的约束优化思想与 **MM-RLHF / MM-DPO** 的动态奖励缩放机制，在“有帮助性”（helpfulness）与“无害性”（safety）之间实现更精细的平衡，并提供从数据准备、评判员训练、Safe-MM-DPO 对齐到（可选）知识蒸馏的端到端代码。

本仓库适配 **Python 3.10.10 + PyTorch 2.6** 环境，推荐通过 GitHub 同步代码，在目标训练机器上运行。

---

## 1. 实现原理概述

### 1.1 Safe RLHF：目标解耦与约束优化

传统 RLHF 往往用一个奖励模型同时刻画“有帮助”和“无害”，容易混淆两个目标。本项目借鉴 **Safe RLHF** 思想，将对齐目标拆分为两个维度：

- **Helpful-RM**：评估回答在任务完成、信息量、清晰度等方面的“有帮助性”；
- **Harmless-RM**：评估回答在是否包含有害、违法、偏见内容等方面的“无害性”（或有害成本）。

理想优化目标可以表述为一个带约束的期望优化问题：

\[
\underset{\theta}{\text{maximize}} \; \mathbb{E}[R_{\text{helpful}}(y, x)] \quad
\text{s.t.} \quad \mathbb{E}[C_{\text{harm}}(y, x)] \le 0
\]

其中约束项通过 **拉格朗日乘子 \(\lambda\)** 转化为无约束优化，并在训练中动态更新，实现对安全性的自适应控制。

### 1.2 MM-DPO：动态奖励缩放的偏好对训练

在实践中直接做三阶段 RL（训练 RM/CM + RL）成本高昂，项目采用 **MM-RLHF / MM-DPO** 思路，在 **DPO（Direct Preference Optimization）** 框架上引入 **动态缩放因子 \(\beta(\delta)\)**：

- DPO 的核心是利用偏好对 \((x, y_w, y_l)\) 优化策略 \(\pi_\theta\)，绕过显式 reward 模型；
- MM-DPO 在 DPO 损失中加入 **动态样本权重**，根据奖励边际 \(\delta = r(y_w) - r(y_l)\) 调整每个样本的影响力：

\[
\beta(\delta) = \beta_{\text{ori}} \left( 1 + w (1 - e^{-k \delta}) \right)
\]

奖励差距越大（高置信度偏好对），\(\beta(\delta)\) 越大，模型从这些“更确定”的样本中学得更多。

### 1.3 Safe-MM-DPO：双评判员 + 动态缩放 + 拉格朗日平衡

本项目的核心算法是 **Safe-MM-DPO**，融合上述两种思想：

1. **双评判员（双 Reward Models）**：
   - 在相同偏好对数据上，分别训练：
     - `Helpful-RM`：输出 \(R_{\text{helpful}}(y, x)\)
     - `Harmless-RM`：输出 \(R_{\text{harmless}}(y, x)\)
   - 对每个偏好对 \((x, y_w, y_l)\) 计算两个奖励边际：
     - \(\delta_H = R_{\text{helpful}}(y_w, x) - R_{\text{helpful}}(y_l, x)\)
     - \(\delta_S = R_{\text{harmless}}(y_w, x) - R_{\text{harmless}}(y_l, x)\)

2. **双通道 MM-DPO 损失**：
   - 在 **同一个策略模型** 上，分别构造：
     - 帮助性损失 \(\mathcal{L}_H\)：基于 \(\delta_H\) 和 \(\beta_H(\delta_H)\)
     - 无害性损失 \(\mathcal{L}_S\)：基于 \(\delta_S\) 和 \(\beta_S(\delta_S)\)

3. **拉格朗日乘子 \(\lambda\)：自动平衡帮助性与安全性**：
   - 总损失：
     \[
     \mathcal{L}_{\text{Total}} = \frac{1}{1+\lambda} \mathcal{L}_H + \frac{\lambda}{1+\lambda} \mathcal{L}_S + \text{KL 正则}
     \]
   - 训练中根据有害性成本近似值 \(J_C\) 更新 \(\lambda\)：
     \[
     \lambda \leftarrow \lambda + \lambda_{\text{lr}} (J_C - \text{threshold})
     \]
   - 当模型更“危险”时（有害性成本偏高），\(\lambda\) 增大，使得 \(\mathcal{L}_S\) 权重提高，强化安全性约束；反之则更多关注提升有帮助性。

---

## 2. 项目结构与关键模块

简化的目录结构如下（详见 `dev.md`）：

```text
humanAlignment/
├── configs/
│   ├── rm_config.yaml         # 奖励模型训练配置
│   └── dpo_config.yaml        # Safe-MM-DPO 训练配置
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── reward_model.py    # 奖励模型封装（基于 AutoModelForSequenceClassification）
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── data_preprocessor.py
│   ├── scripts/
│   │   ├── __init__.py
│   │   └── prepare_data.py    # 数据预处理 CLI
│   └── training/
│       ├── __init__.py
│       ├── train_rm.py        # Helpful-RM / Harmless-RM 训练脚本
│       └── train_safe_mm_dpo.py # Safe-MM-DPO 对齐训练脚本
├── requirements.txt
├── dev.md
├── demo.md
└── pretrain_guide.md          # 运行手册：端到端预训练步骤
```

**关键模块说明**：

- `src/data_processing/data_preprocessor.py` / `src/scripts/prepare_data.py`  
  将 BeaverTails 风格的原始数据转换为 `(prompt, chosen_response, rejected_response)` 偏好对，并划分为 `train/val/test`，分别输出 helpful / harmless 两套 jsonl。

- `src/models/reward_model.py`  
  基于 `AutoModelForSequenceClassification(num_labels=1)` 实现 RewardModel 封装，输出标量奖励分数。

- `src/training/train_rm.py`  
  从 jsonl 偏好对训练 Helpful-RM / Harmless-RM，使用 pairwise ranking loss：
  \[
  \text{loss} = -\log\sigma(score_{\text{chosen}} - score_{\text{rejected}})
  \]

- `src/training/train_safe_mm_dpo.py`  
  实现 Safe-MM-DPO 训练循环，是本项目的算法核心（代码中有详细注释标示双评判员、动态 β、λ 更新等创新点）。

---

## 3. 环境准备

### 3.1 基础环境

- Python 3.10.10（建议）
- PyTorch 2.6.x（根据目标机器 CUDA 版本安装）

创建环境（示例）：

```bash
conda create -n humanAlignment python=3.10
conda activate humanAlignment

pip install -r requirements.txt
```

若在 GPU 环境，请参考 PyTorch 官方安装页面为目标 CUDA 版本安装 `torch==2.6.*`。

### 3.2 下载基础模型

```bash
mkdir -p models/base

pip install huggingface_hub

huggingface-cli download mistralai/Mistral-7B-v0.1 \
    --local-dir models/base/Mistral-7B-v0.1
```

---

## 4. 使用方法：端到端训练流程

### 步骤 1：准备数据

假设你已经将 Safe-RLHF 仓库克隆到 `data/safe-rlhf`，并把 BeaverTails 相关文件复制到 `data/raw`，可运行：

```bash
python -m src.scripts.prepare_data \
    --input_dir data/raw \
    --output_dir data \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

生成的关键文件包括：

- `data/train/helpful_pairs.jsonl`
- `data/val/helpful_pairs.jsonl`
- `data/train/harmless_pairs.jsonl`
- `data/val/harmless_pairs.jsonl`

### 步骤 2：训练两个奖励模型（Helpful-RM & Harmless-RM）

`configs/rm_config.yaml` 已给出推荐配置，可以直接使用：

```bash
# Helpful-RM
python -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task helpful \
    --output_dir models/helpful_rm

# Harmless-RM
python -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task harmless \
    --output_dir models/harmless_rm
```

训练结束后，会在 `models/helpful_rm` 和 `models/harmless_rm` 下保存最佳 checkpoint。

### 步骤 3：Safe-MM-DPO 对齐训练

`configs/dpo_config.yaml` 已包含推荐超参与路径，可以直接运行：

```bash
python -m src.training.train_safe_mm_dpo \
    --config configs/dpo_config.yaml \
    --output_dir models/aligned \
    --logging_dir logs/training/safe_mm_dpo
```

训练过程中：

- 使用策略模型 `π_θ` 与参考模型 `π_ref` 计算偏好对上的 log π(y|x)；
- 用 Helpful-RM / Harmless-RM 分别计算奖励边际 δ_H / δ_S；
- 基于 δ_H / δ_S 计算动态 β_H / β_S，并构造两个 MM-DPO 损失；
- 通过拉格朗日乘子 λ 动态平衡帮助性与安全性损失；
- 可选地引入 KL 约束，避免策略偏离参考模型过远。

每个 epoch 结束后，对齐后的策略会保存到 `models/aligned/epoch_<n>`。

---

## 5. （可选）知识蒸馏与评估

### 5.1 蒸馏数据生成

在得到对齐后的模型 `models/aligned` 后，可以通过脚本（待实现示例接口已在 `pretrain_guide.md` 中说明）生成 dSFT 蒸馏数据：

```bash
python -m src.scripts.generate_distillation_data \
    --model_path models/aligned \
    --input_prompts data/train/prompts_for_distill.jsonl \
    --output_path data/distill/aligned_responses.jsonl \
    --num_samples 50000
```

之后可以选择如 TinyLlama-1.1B 作为学生模型，在上述数据上进行监督微调。

### 5.2 评估

评估流程在 `dev.md` 中有详细说明，主要包括：

- **BeaverTails Safety Benchmark**：评估安全率（拒绝有害请求的比例）；
- **MT-Bench**：用 GPT-4 等模型作为裁判评估有帮助性与对话质量。

---

## 6. 总结与扩展方向

本项目提供了一条从理论到实现的完整路径：

- 理论上，将 Safe RLHF 的约束优化与 MM-DPO 的动态样本重权结合，提出 Safe-MM-DPO；
- 工程上，围绕 Mistral-7B 构建了完整的数据处理、奖励模型训练与策略对齐训练代码；
- 提供了可扩展的配置与脚本接口，方便在其他基础模型或数据集上复用。

后续可以扩展的方向包括：

- 引入更精细的成本建模（例如多类别有害性标签）；
- 在多语言或多模态场景下验证 Safe-MM-DPO 的效果；
- 与其他 RLHF / DPO 变体对比，系统分析安全性与有用性的 trade-off。

如需修改或扩展，请优先阅读 `dev.md`（开发文档）与 `pretrain_guide.md`（运行手册），它们与本 README 一起构成项目的完整说明。 


