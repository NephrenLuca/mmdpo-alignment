# 开发文档：基于Safe-RLHF与MM-DPO的语言模型价值对齐

## 目录

1. [项目结构](#1-项目结构)
2. [远程开发环境配置](#2-远程开发环境配置)
3. [环境依赖与安装](#3-环境依赖与安装)
4. [数据准备流程](#4-数据准备流程)
5. [训练流程](#5-训练流程)
6. [模型评估流程](#6-模型评估流程)
7. [知识蒸馏流程（可选）](#7-知识蒸馏流程可选)
8. [监控与调试](#8-监控与调试)
9. [最佳实践与注意事项](#9-最佳实践与注意事项)

---

## 1. 项目结构

### 1.1 目录组织

建议的项目目录结构如下：

```
humanAlignment/
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 预处理后的数据
│   ├── train/                 # 训练集
│   ├── val/                   # 验证集
│   └── test/                  # 测试集（用于最终评估）
├── models/                    # 模型存储目录
│   ├── base/                  # 基础模型（Mistral-7B）
│   ├── helpful_rm/            # Helpful奖励模型
│   ├── harmless_rm/           # Harmless奖励模型
│   ├── aligned/               # 对齐后的模型
│   └── distilled/             # 蒸馏后的模型
├── src/                       # 源代码目录
│   ├── data_processing/       # 数据处理模块
│   │   ├── __init__.py
│   │   ├── dataset_loader.py  # 数据集加载器
│   │   └── data_preprocessor.py # 数据预处理
│   ├── models/                # 模型定义模块
│   │   ├── __init__.py
│   │   ├── reward_model.py    # 奖励模型架构
│   │   └── policy_model.py    # 策略模型架构
│   ├── training/              # 训练模块
│   │   ├── __init__.py
│   │   ├── train_rm.py        # 奖励模型训练
│   │   ├── train_safe_mm_dpo.py # Safe-MM-DPO训练
│   │   └── train_distillation.py # 知识蒸馏训练
│   ├── evaluation/            # 评估模块
│   │   ├── __init__.py
│   │   ├── mt_bench_eval.py   # MT-Bench评估
│   │   └── safety_eval.py     # 安全性评估
│   ├── utils/                 # 工具函数
│   │   ├── __init__.py
│   │   ├── config.py          # 配置管理
│   │   ├── logger.py          # 日志管理
│   │   └── metrics.py         # 指标计算
│   └── scripts/               # 可执行脚本
│       ├── prepare_data.py
│       ├── train_all.sh
│       └── evaluate_all.sh
├── configs/                   # 配置文件目录
│   ├── rm_config.yaml         # 奖励模型配置
│   ├── dpo_config.yaml        # DPO训练配置
│   └── eval_config.yaml       # 评估配置
├── logs/                      # 日志目录
│   ├── training/              # 训练日志
│   └── evaluation/            # 评估日志
├── checkpoints/               # 检查点目录
│   ├── rm_checkpoints/
│   ├── dpo_checkpoints/
│   └── distillation_checkpoints/
├── results/                   # 结果存储目录
│   ├── training_curves/       # 训练曲线图
│   ├── evaluation_results/    # 评估结果
│   └── model_comparisons/     # 模型对比结果
├── requirements.txt           # Python依赖
├── setup.sh                   # 环境设置脚本
├── demo.md                    # 项目计划书
└── development.md             # 本开发文档
```

### 1.2 关键文件说明

- **数据处理模块**：负责加载和预处理BeaverTails数据集，将其转换为训练所需的格式
- **模型定义模块**：定义奖励模型和策略模型的结构
- **训练模块**：实现Safe-MM-DPO算法的核心训练逻辑
- **评估模块**：实现MT-Bench和BeaverTails安全基准的评估
- **配置管理**：使用YAML文件管理所有超参数，便于实验管理

---

## 2. 开发与运行环境配置

### 2.1 开发方式与代码同步（GitHub）

本项目推荐在**本地开发 + GitHub 同步代码**的方式下进行，最终在目标训练/推理机器上拉取代码并运行。

1. **本地开发环境**：
   - 在本地机器上使用 Cursor 打开项目仓库
   - 使用 `git` 管理代码版本
   - 通过 GitHub 仓库进行备份与协作

2. **GitHub 工作流示例**：
   ```bash
   # 首次推送
   git init
   git remote add origin git@github.com:<your_name>/<your_repo>.git
   git add .
   git commit -m "init humanAlignment project"
   git push -u origin main

   # 本地与 GitHub 同步
   git pull origin main   # 拉取远程更新
   git push origin main   # 推送本地更改
   ```

3. **在目标机器上运行代码**：
   - 在目标机器（Python 3.10.10 + PyTorch 2.6 环境）上安装好依赖
   - 使用 `git clone` 或 `git pull` 从 GitHub 获取最新代码
   - 在该机器上执行训练与评估脚本

4. **大文件管理（可选）**：
   - 对于模型权重、数据集等大文件，建议：
     - 使用 Git LFS 或单独的对象存储（如 OSS、S3）
     - 或直接在目标机器上通过 `huggingface_hub` / `datasets` 在线下载

### 2.2 目标运行环境要求

确保最终运行本项目的目标机器（训练/推理机器）满足以下要求：

- **硬件**：
  - GPU：至少1块具有16GB+显存的GPU（推荐多块GPU用于并行训练）
  - 内存：至少64GB RAM
  - 存储：至少500GB可用空间（用于模型、数据和检查点）

- **软件**：
  - 操作系统：Linux（Ubuntu 20.04+推荐）
  - Python：**3.10.10**（推荐；至少 3.10.x）
  - PyTorch：**2.6.x**（与 Python 3.10.10 兼容）
  - CUDA：根据 PyTorch 2.6 官方安装指引选择对应版本（如使用 GPU）
  - cuDNN：与CUDA版本匹配

---

## 3. 环境依赖与安装

### 3.1 Python环境管理

推荐使用 `conda` 或 `venv` 创建独立的Python环境，**目标运行环境为 Python 3.10.10**（本项目以该版本进行开发与测试）：

```bash
# 使用conda（推荐）
conda create -n humanAlignment python=3.10  # 实际运行环境建议使用 3.10.10
conda activate humanAlignment

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3.2 核心依赖包

创建 `requirements.txt` 文件，包含以下主要依赖（建议在目标机器上与 PyTorch 2.6 版本保持兼容）：

```
# 深度学习框架
torch==2.6.*
transformers>=4.35.0
accelerate>=0.24.0
peft>=0.6.0  # 用于LoRA等参数高效微调

# RLHF相关
trl>=0.7.0  # Transformers Reinforcement Learning
datasets>=2.14.0

# 数据处理
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# 评估工具
openai>=1.0.0  # 用于MT-Bench评估（如果需要API调用）

# 工具库
tqdm>=4.66.0
wandb>=0.15.0  # 可选：用于实验跟踪
tensorboard>=2.14.0
pyyaml>=6.0
tqdm>=4.66.0

# 其他
sentencepiece>=0.1.99
protobuf>=3.20.0
```

### 3.3 安装步骤

1. **安装PyTorch**（根据是否使用GPU以及CUDA版本选择具体命令，以下为示意）：
   ```bash
   # CPU 版本示例（在目标机器上）
   pip install "torch==2.6.*" "torchvision" "torchaudio"

   # 如需 GPU 版本，请参考 PyTorch 官方安装页面，
   # 根据具体 CUDA 版本生成对应的 pip/conda 安装命令：
   # https://pytorch.org/get-started/locally/
   ```

2. **安装其他依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **验证安装**：
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   python -c "import transformers; print(transformers.__version__)"
   ```

### 3.4 模型下载

在开始训练前，需要下载基础模型。当前方案选择 **Mistral-7B-v0.1** 作为基础 Causal LM：

- **合理性说明**：
  - 7B 规模模型在**能力与资源消耗**之间取得较好平衡，适合做 Safe-RLHF / DPO 类对齐实验
  - `mistralai/Mistral-7B-v0.1` 在 Hugging Face 上维护良好，兼容 `transformers` 4.35+，可在 **Python 3.10.10 + PyTorch 2.6** 环境下正常加载
  - 结合 LoRA / PEFT 等参数高效微调方法，单机多卡或单卡大显存（如 80GB）即可完成训练；低显存场景可考虑 4bit 量化 + LoRA
  - 如希望进一步降低资源占用，可在后续蒸馏阶段选择更小的学生模型（如 TinyLlama-1.1B）

```bash
# 创建模型目录
mkdir -p models/base

# 使用Hugging Face CLI下载（需要先安装：pip install huggingface_hub）
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir models/base/Mistral-7B-v0.1

# 或使用Python脚本下载（在 Python 3.10.10 + torch 2.6 环境中）
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('mistralai/Mistral-7B-v0.1', cache_dir='models/base')"
```

---

## 4. 数据准备流程

### 4.1 数据集获取

1. **BeaverTails数据集**：
   - 从Safe RLHF的GitHub仓库获取：`PKU-Alignment/safe-rlhf`
   - 或从Hugging Face Datasets加载（如果可用）
   - 数据集应包含以下字段：
     - `prompt`: 用户提示
     - `response_chosen`: 被选中的回答（有帮助/无害）
     - `response_rejected`: 被拒绝的回答
     - `helpful_label`: 有帮助性标签（0/1）
     - `harmless_label`: 无害性标签（0/1）

2. **数据格式要求**：
   - 偏好对格式：`(prompt, chosen_response, rejected_response)`
   - 需要分别准备"有帮助"和"无害"两个维度的偏好对

### 4.2 数据预处理流程

1. **数据加载**：
   - 从原始数据源加载数据
   - 验证数据完整性和格式
   - 统计数据集大小和分布

2. **数据清洗**：
   - 移除空值或格式错误的数据
   - 过滤过短或过长的样本（可选）
   - 去重处理

3. **数据分割**：
   - 训练集：80%
   - 验证集：10%（用于训练过程中的快速评估）
   - 测试集：10%（用于最终评估，特别是安全性评估）

4. **数据转换**：
   - 将数据转换为模型输入格式
   - 应用tokenization
   - 创建DataLoader

### 4.3 数据验证

在开始训练前，进行数据验证：

- 检查数据分布（有帮助/无害标签的分布）
- 验证偏好对的质量（随机采样检查）
- 确认数据加载速度（避免I/O瓶颈）

---

## 5. 训练流程

### 5.1 阶段一：训练评判员模型（Reward Models）

#### 5.1.1 训练Helpful-RM

**目标**：训练一个专门评估回答"有帮助性"的奖励模型。

**流程**：
1. **模型初始化**：
   - 基于Mistral-7B的架构
   - 在序列末尾添加一个标量输出头（用于输出奖励分数）
   - 或使用序列分类头

2. **数据准备**：
   - 使用"有帮助"维度的偏好对
   - 格式：`(prompt, chosen_response, rejected_response)`
   - 标签：chosen=1, rejected=0（或使用相对排名）

3. **训练配置**：
   - 学习率：1e-5 到 5e-5
   - 批次大小：16-32（根据GPU显存调整）
   - 训练轮数：1-3个epoch
   - 使用AdamW优化器
   - 应用梯度裁剪（max_grad_norm=1.0）

4. **损失函数**：
   - 使用对比损失（ranking loss）：
     ```
     loss = -log(sigmoid(score_chosen - score_rejected))
     ```

5. **训练监控**：
   - 记录训练损失和验证损失
   - 监控准确率（chosen > rejected的比例）
   - 保存最佳检查点

#### 5.1.2 训练Harmless-RM

**流程**：与Helpful-RM相同，但使用"无害"维度的偏好对。

**注意事项**：
- 两个RM可以并行训练（如果资源充足）
- 使用相同的超参数配置
- 分别保存到不同的目录

### 5.2 阶段二：Safe-MM-DPO对齐训练

这是项目的核心训练阶段。

#### 5.2.1 模型初始化

1. **策略模型**：
   - 加载Mistral-7B作为初始策略模型 `π_θ`
   - 同时作为参考模型 `π_ref`（冻结参数，不更新）

2. **评判员模型**：
   - 加载训练好的Helpful-RM和Harmless-RM
   - 设置为评估模式（`eval()`），不参与梯度计算

3. **拉格朗日乘子**：
   - 初始化 `λ = λ_init`（默认1.0）
   - 作为可训练参数或手动更新

#### 5.2.2 训练循环实现

**每个训练步骤**：

1. **前向传播**：
   - 从批次中获取偏好对 `(x, y_w, y_l)`
   - 使用策略模型计算 `π_θ(y_w|x)` 和 `π_θ(y_l|x)`
   - 使用参考模型计算 `π_ref(y_w|x)` 和 `π_ref(y_l|x)`

2. **计算奖励边际**：
   - 使用Helpful-RM计算：`δ_H = R_helpful(y_w, x) - R_helpful(y_l, x)`
   - 使用Harmless-RM计算：`δ_S = R_harmless(y_w, x) - R_harmless(y_l, x)`

3. **动态缩放因子**：
   - 计算 `β_H = β_ori * (1 + w * (1 - exp(-k * δ_H)))`
   - 计算 `β_S = β_ori * (1 + w * (1 - exp(-k * δ_S)))`

4. **计算损失**：
   - 帮助性损失 `L_H`：使用MM-DPO公式，应用 `β_H`
   - 无害性损失 `L_S`：使用MM-DPO公式，应用 `β_S`
   - 总损失：`L_Total = (1/(1+λ)) * L_H + (λ/(1+λ)) * L_S`

5. **反向传播与更新**：
   - 计算梯度并更新策略模型参数
   - 更新拉格朗日乘子 `λ`：
     ```
     λ = λ + λ_lr * (J_C - threshold)
     ```
     其中 `J_C` 是当前批次的有害性成本期望，`threshold=0`

6. **KL散度约束**（可选）：
   - 添加KL散度项防止策略偏离参考模型过远：
     ```
     L_KL = kl_coeff * KL(π_θ || π_ref)
     ```

#### 5.2.3 训练配置

使用项目计划书中推荐的配置：

```yaml
# dpo_config.yaml
λ_init: 1.0
w: 0.5
k: 0.5
β_ori: 0.1
learning_rate: 5e-7
batch_size: 16
kl_coeff: 0.1
λ_lr: 0.01
epochs: 2
gradient_accumulation_steps: 4  # 有效批次大小 = 16 * 4 = 64
max_grad_norm: 1.0
warmup_steps: 100
save_steps: 500
eval_steps: 250
```

#### 5.2.4 训练监控指标

记录以下指标：

- **损失指标**：
  - `loss_total`: 总损失
  - `loss_helpful`: 帮助性损失
  - `loss_safety`: 安全性损失
  - `loss_kl`: KL散度损失（如果使用）

- **奖励指标**：
  - `reward_helpful_mean`: 平均帮助性奖励
  - `reward_safety_mean`: 平均安全性奖励
  - `delta_helpful_mean`: 平均帮助性奖励边际
  - `delta_safety_mean`: 平均安全性奖励边际

- **动态参数**：
  - `lambda`: 拉格朗日乘子的当前值
  - `beta_helpful_mean`: 平均帮助性缩放因子
  - `beta_safety_mean`: 平均安全性缩放因子

- **模型指标**：
  - `kl_divergence`: 策略与参考模型的KL散度
  - `perplexity`: 困惑度（可选）

#### 5.2.5 检查点管理

- **保存策略**：
  - 定期保存（每N步）
  - 保存最佳检查点（基于验证集上的综合指标）
  - 保存最终检查点

- **检查点内容**：
  - 模型权重
  - 优化器状态
  - 当前epoch和步数
  - 当前λ值
  - 训练配置

### 5.3 阶段三：超参数调优（可选）

如果时间允许，进行快速超参数搜索：

1. **网格搜索配置**：
   - `λ_init`: [0.5, 1.0, 2.0]
   - `w`: [0.3, 0.5, 1.0]
   - 共9个组合

2. **快速评估策略**：
   - 每个配置只训练1个epoch
   - 在小型验证集（100个样本）上评估
   - 评估指标：安全率（目标>80%）和MT-Bench得分

3. **最佳配置选择**：
   - 选择安全率>80%且MT-Bench得分最高的配置
   - 使用该配置进行完整训练（2个epoch）

---

## 6. 模型评估流程

### 6.1 评估准备

1. **模型加载**：
   - 加载对齐后的模型检查点
   - 设置为评估模式

2. **测试集准备**：
   - 加载BeaverTails测试集（用于安全性评估）
   - 准备MT-Bench测试集（用于有帮助性评估）

### 6.2 BeaverTails安全性评估

**目标**：评估模型拒绝生成有害内容的能力。

**流程**：
1. **测试提示**：
   - 使用BeaverTails测试集中的提示
   - 这些提示设计用于诱发有害回答

2. **生成回答**：
   - 对每个提示，使用模型生成回答
   - 设置合适的生成参数（temperature, max_length等）

3. **安全性判断**：
   - 使用预训练的安全分类器（或人工标注）
   - 判断回答是否包含有害内容

4. **计算指标**：
   - **安全率** = (拒绝有害请求的数量) / (总测试样本数)
   - 目标：安全率显著高于基础模型Mistral-7B

5. **详细分析**：
   - 按类别统计安全率（如：暴力、偏见、非法活动等）
   - 分析失败案例，了解模型的薄弱环节

### 6.3 MT-Bench评估

**目标**：评估模型的"有帮助性"和对话质量。

**流程**：
1. **测试集加载**：
   - 从LMSYS获取MT-Bench测试集
   - 包含多轮对话场景

2. **回答生成**：
   - 对每个测试提示，生成模型回答
   - 保持与原始评估设置一致的生成参数

3. **评分**：
   - 使用GPT-4作为裁判（通过API调用）
   - 或使用本地评估脚本（如果可用）
   - 每个回答获得1-10分的评分

4. **计算指标**：
   - **平均得分**：所有回答的平均分
   - **分维度得分**：不同类别（如：写作、数学、推理等）的平均分

5. **对比分析**：
   - 与基础模型Mistral-7B对比
   - 目标：得分不显著下降，甚至有所提升

### 6.4 综合评估报告

生成包含以下内容的评估报告：

- **执行摘要**：主要发现和结论
- **详细指标**：
  - 安全性指标（安全率、分类别安全率）
  - 有帮助性指标（MT-Bench得分、分维度得分）
- **对比分析**：
  - 与基础模型的对比
  - 训练前后的对比
- **可视化**：
  - 安全性评估结果图表
  - MT-Bench得分分布
  - 训练曲线（如果包含）

---

## 7. 知识蒸馏流程（可选）

如果时间允许，实施知识蒸馏以创建轻量级模型。

### 7.1 数据生成

1. **使用对齐模型生成数据**：
   - 从训练集中采样提示
   - 使用对齐后的模型生成高质量回答
   - 过滤低质量回答（基于奖励模型分数）

2. **数据格式**：
   - 格式：`(prompt, response)`
   - 目标：创建高质量的监督学习数据集

### 7.2 学生模型选择

- **推荐**：TinyLlama-1.1B 或类似的小型模型
- **考虑因素**：参数量、性能基线、训练时间

### 7.3 监督微调（dSFT）

1. **训练配置**：
   - 学习率：1e-4 到 5e-4
   - 批次大小：32-64
   - 训练轮数：3-5个epoch

2. **损失函数**：
   - 标准语言建模损失（交叉熵）

3. **训练监控**：
   - 记录训练损失和验证损失
   - 监控困惑度

### 7.4 蒸馏后评估

- 在相同的测试集上评估蒸馏后的模型
- 对比蒸馏前后的性能
- 分析性能损耗和效率提升

---

## 8. 监控与调试

### 8.1 训练监控

#### 8.1.1 实时监控工具

1. **TensorBoard**：
   - 记录所有训练指标
   - 可视化训练曲线
   - 监控GPU利用率

2. **Weights & Biases (可选)**：
   - 实验跟踪和对比
   - 超参数调优
   - 团队协作

3. **命令行监控**：
   - 使用 `watch -n 1 nvidia-smi` 监控GPU
   - 使用 `htop` 监控CPU和内存

#### 8.1.2 关键指标监控

- **训练稳定性**：
  - 损失是否正常下降
  - 梯度是否爆炸或消失（检查梯度范数）
  - λ值的变化趋势

- **资源使用**：
  - GPU利用率（目标>80%）
  - 内存使用（避免OOM）
  - 训练速度（samples/sec）

### 8.2 常见问题与调试

#### 8.2.1 训练问题

1. **损失不下降**：
   - 检查学习率是否过大或过小
   - 验证数据加载是否正确
   - 检查模型初始化

2. **内存不足（OOM）**：
   - 减小批次大小
   - 使用梯度累积
   - 启用梯度检查点（gradient checkpointing）
   - 使用混合精度训练（FP16/BF16）

3. **训练不稳定**：
   - 减小学习率
   - 增加梯度裁剪强度
   - 检查数据质量

4. **λ值异常**：
   - 如果λ增长过快，减小`λ_lr`
   - 如果λ不更新，检查更新逻辑

#### 8.2.2 评估问题

1. **评估速度慢**：
   - 使用批量生成
   - 减少生成长度
   - 使用更快的生成策略（如beam search vs sampling）

2. **结果异常**：
   - 验证测试集是否正确加载
   - 检查模型是否正确加载
   - 确认评估脚本逻辑

### 8.3 日志管理

1. **日志级别**：
   - DEBUG：详细调试信息
   - INFO：一般信息（训练进度、指标）
   - WARNING：警告信息
   - ERROR：错误信息

2. **日志内容**：
   - 训练步骤和指标
   - 检查点保存信息
   - 错误和异常堆栈
   - 配置信息

3. **日志轮转**：
   - 避免日志文件过大
   - 定期归档旧日志

---

## 9. 最佳实践与注意事项

### 9.1 训练最佳实践

1. **渐进式训练**：
   - 先在小型数据集上验证代码正确性
   - 再在完整数据集上训练

2. **检查点策略**：
   - 频繁保存检查点（避免训练中断导致进度丢失）
   - 保留多个检查点（用于回滚）

3. **实验管理**：
   - 为每个实验创建独立的配置文件和输出目录
   - 记录实验配置和结果
   - 使用版本控制管理代码

4. **资源管理**：
   - 监控资源使用，避免影响其他任务
   - 使用资源限制（如CUDA_VISIBLE_DEVICES）

### 9.2 代码质量

1. **模块化设计**：
   - 将功能拆分为独立模块
   - 便于测试和调试

2. **配置管理**：
   - 使用配置文件（YAML）管理超参数
   - 避免硬编码

3. **错误处理**：
   - 添加适当的异常处理
   - 提供有意义的错误信息

4. **代码注释**：
   - 为关键函数和类添加文档字符串
   - 解释复杂的算法逻辑

### 9.3 数据安全与隐私

1. **数据保护**：
   - 确保训练数据的安全存储
   - 遵守数据使用协议

2. **模型安全**：
   - 定期评估模型的安全性
   - 记录模型的行为和限制

### 9.4 远程训练注意事项

1. **连接稳定性**：
   - 使用 `tmux` 或 `screen` 保持会话
   - 配置SSH keepalive
   - 考虑使用VPN（如果需要）

2. **数据传输**：
   - 大文件传输使用 `rsync`（支持断点续传）
   - 考虑使用压缩减少传输时间

3. **任务管理**：
   - 长时间训练使用 `nohup` 或 `tmux`
   - 设置自动重启机制（如果可能）

4. **备份策略**：
   - 定期备份重要检查点
   - 使用版本控制管理代码

### 9.5 性能优化

1. **训练加速**：
   - 使用混合精度训练（FP16/BF16）
   - 启用数据并行（多GPU）
   - 优化数据加载（多进程、预取）

2. **内存优化**：
   - 使用梯度检查点
   - 及时释放不需要的变量
   - 使用CPU卸载（如果适用）

3. **I/O优化**：
   - 使用SSD存储数据
   - 预加载常用数据到内存
   - 优化数据预处理流程

---

## 10. 开发时间线建议

### 阶段一：环境搭建与数据准备（1-2天）
- SSH连接配置
- 环境安装
- 数据获取与预处理

### 阶段二：评判员模型训练（2-3天）
- Helpful-RM训练
- Harmless-RM训练
- 模型验证

### 阶段三：Safe-MM-DPO训练（3-5天）
- 核心训练逻辑实现
- 初始训练与调试
- 超参数调优（如果时间允许）

### 阶段四：模型评估（1-2天）
- BeaverTails安全性评估
- MT-Bench评估
- 结果分析与报告

### 阶段五：知识蒸馏（可选，2-3天）
- 数据生成
- 学生模型训练
- 蒸馏后评