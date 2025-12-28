# Safe-MM-DPO 训练完整操作指南

本指南提供从数据处理到模型训练的完整流程说明。

## 目录

1. [环境准备](#环境准备)
2. [数据处理](#数据处理)
3. [训练 Reward Model](#训练-reward-model)
4. [训练 Safe-MM-DPO](#训练-safe-mm-dpo)
5. [监控和管理训练](#监控和管理训练)
6. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装依赖

```bash
# 确保已安装所有依赖
pip install -r requirements.txt

# 关键依赖包括：
# - torch >= 2.0.0
# - transformers >= 4.30.0
# - peft (用于LoRA)
# - datasets (用于下载数据)
# - huggingface_hub
```

### 2. 准备基础模型

确保基础模型已下载到 `models/base/Mistral-7B-v0.1/`：

```bash
# 如果模型不存在，需要从 Hugging Face 下载
# 或使用已有的模型路径
```

### 3. 准备原始数据

将 PKU-SafeRLHF 数据集下载到 `data/raw/` 目录：

```bash
# 方法1：使用脚本下载
python scripts/download_safe_rlhf_data.py \
    --output_dir data/raw \
    --splits train test

# 方法2：使用 Hugging Face CLI
huggingface-cli download PKU-Alignment/PKU-SafeRLHF \
    --repo-type dataset \
    --local-dir data/raw/pku_saferlhf
```

数据格式要求：
- 文件格式：JSONL
- 必需字段：`prompt`, `response_0`, `response_1`
- 可选字段：`better_response_id`, `safer_response_id`, `is_response_0_safe`, `is_response_1_safe`

---

## 数据处理

### 步骤 1：运行数据预处理脚本

```bash
python -m src.scripts.prepare_data \
    --input_dir data/raw \
    --output_dir data \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

**参数说明**：
- `--input_dir`: 原始数据目录（包含 JSONL 文件）
- `--output_dir`: 输出目录（将创建 `train/`, `val/`, `test/` 子目录）
- `--train_ratio`: 训练集比例（默认 0.8）
- `--val_ratio`: 验证集比例（默认 0.1）
- `--test_ratio`: 测试集比例（默认 0.1）

**输出文件**：
- `data/train/helpful_pairs.jsonl` - 有帮助性偏好对（训练集）
- `data/train/harmless_pairs.jsonl` - 无害性偏好对（训练集，包含 safety labels）
- `data/val/helpful_pairs.jsonl` - 有帮助性偏好对（验证集）
- `data/val/harmless_pairs.jsonl` - 无害性偏好对（验证集）
- `data/test/helpful_pairs.jsonl` - 有帮助性偏好对（测试集）
- `data/test/harmless_pairs.jsonl` - 无害性偏好对（测试集）

**重要提示**：
- 如果数据中包含 `is_response_X_safe` 字段，harmless 数据会自动提取 safety labels
- 重新运行此脚本会**覆盖**已有数据文件，建议先备份

### 步骤 2：验证数据格式

```bash
# 检查数据文件是否存在
ls -lh data/train/*.jsonl
ls -lh data/val/*.jsonl

# 查看数据样例
head -n 1 data/train/helpful_pairs.jsonl | python3 -m json.tool
head -n 1 data/train/harmless_pairs.jsonl | python3 -m json.tool

# 检查 harmless 数据是否包含 safety_labels
head -n 1 data/train/harmless_pairs.jsonl | python3 -m json.tool | grep safety_labels
```

---

## 训练 Reward Model

### 配置说明

编辑 `configs/rm_config.yaml` 以调整训练参数：

```yaml
# 关键参数
batch_size: 4              # 根据显存调整
max_length: 512           # 序列最大长度
num_epochs: 1             # 训练轮数
learning_rate: 2e-5       # 学习率
use_lora: true            # 启用LoRA（推荐）
classification_loss_weight: 1.0  # 仅用于harmless RM
```

### 训练 Helpful RM

**单GPU训练**：
```bash
python -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task helpful \
    --output_dir models/helpful_rm
```

**多GPU分布式训练**（推荐）：
```bash
# 使用8个GPU
bash scripts/train_rm_distributed.sh helpful 8

# 后台运行
bash scripts/train_rm_distributed.sh helpful 8 --background
```

**参数说明**：
- `--task`: 任务类型，`helpful` 或 `harmless`
- `--output_dir`: 模型保存目录
- `--config`: 配置文件路径

**输出**：
- 模型保存在 `models/helpful_rm/`
- 日志保存在 `logs/training/rm/helpful_*.log`

### 训练 Harmless RM (Cost Model)

**单GPU训练**：
```bash
python -m src.training.train_rm \
    --config configs/rm_config.yaml \
    --task harmless \
    --output_dir models/harmless_rm
```

**多GPU分布式训练**（推荐）：
```bash
# 使用8个GPU
bash scripts/train_rm_distributed.sh harmless 8

# 后台运行
bash scripts/train_rm_distributed.sh harmless 8 --background
```

**重要提示**：
- Harmless RM 会自动使用 classification loss（如果数据包含 safety labels）
- 确保 `classification_loss_weight: 1.0` 在配置文件中

**输出**：
- 模型保存在 `models/harmless_rm/`
- 日志保存在 `logs/training/rm/harmless_*.log`

### 验证训练结果

训练完成后，检查模型文件：

```bash
# 检查模型文件
ls -lh models/helpful_rm/
ls -lh models/harmless_rm/

# 如果使用LoRA，应该看到 adapter_model.bin
# 如果全参数微调，应该看到 pytorch_model.bin 或 model.safetensors
```

查看训练日志：

```bash
# 查看最新日志
tail -n 100 logs/training/rm/helpful_*.log | tail -n 50
tail -n 100 logs/training/rm/harmless_*.log | tail -n 50

# 查找关键指标
grep "val_loss\|val_acc" logs/training/rm/*.log
```

---

## 训练 Safe-MM-DPO

### 前置条件

在开始 DPO 训练前，确保：
1. ✅ 已完成数据处理
2. ✅ 已训练 Helpful RM（保存在 `models/helpful_rm/`）
3. ✅ 已训练 Harmless RM（保存在 `models/harmless_rm/`）

### 配置说明

编辑 `configs/dpo_config.yaml` 以调整训练参数：

```yaml
# Safe-MM-DPO 核心超参数
λ_init: 1.0              # Lambda初始值
w: 0.5                   # 动态beta参数w
k: 0.5                   # 动态beta参数k
β_ori: 0.1               # Beta初始值
cost_threshold: 0.0      # Cost threshold d

# 训练参数
learning_rate: 5e-7      # 学习率（LoRA模式可提高到1e-6）
batch_size: 4            # 批次大小
max_length: 512          # 序列最大长度
epochs: 2                # 训练轮数
gradient_accumulation_steps: 4  # 梯度累积步数

# Lambda更新
λ_lr: 0.01               # Lambda学习率

# LoRA配置
use_lora: true           # 启用LoRA（强烈推荐）
```

### 训练命令

**单GPU训练**（不推荐，显存可能不足）：
```bash
python -m src.training.train_safe_mm_dpo \
    --config configs/dpo_config.yaml \
    --output_dir models/aligned \
    --logging_dir logs/training/safe_mm_dpo
```

**多GPU分布式训练**（推荐）：
```bash
# 使用8个GPU
bash scripts/train_dpo_distributed.sh 8

# 后台运行
bash scripts/train_dpo_distributed.sh 8 --background
```

**参数说明**：
- `--config`: 配置文件路径
- `--output_dir`: 模型保存目录
- `--logging_dir`: 日志保存目录

**输出**：
- 每个 epoch 的 checkpoint 保存在 `models/aligned/epoch_1/`, `models/aligned/epoch_2/` 等
- 日志保存在 `logs/training/safe_mm_dpo/train_*.log`

### 监控训练指标

训练过程中会输出以下关键指标：

```
Epoch 1 step 10 loss_H=0.1234 loss_S=0.5678 KL=0.0123 lambda=1.2345 J_C=0.0012 delta_S_mean=0.3456
```

**指标说明**：
- `loss_H`: Helpful 损失
- `loss_S`: Safety 损失
- `KL`: KL散度（策略与参考模型的差异）
- `lambda`: 当前拉格朗日乘子值
- `J_C`: 期望成本（应该逐渐接近0）
- `delta_S_mean`: 平均安全性差异

**期望行为**：
- `J_C` 应该逐渐接近 0（满足安全性约束）
- `lambda` 应该根据 `J_C` 动态调整
- `loss_H` 和 `loss_S` 应该逐渐下降

---

## 监控和管理训练

### 查看训练状态

使用 `check_training.sh` 脚本：

```bash
bash scripts/check_training.sh
```

该脚本会显示：
- 正在运行的训练进程
- 日志文件位置
- PID 信息

### 查看实时日志

```bash
# RM训练日志
tail -f logs/training/rm/helpful_*.log
tail -f logs/training/rm/harmless_*.log

# DPO训练日志
tail -f logs/training/safe_mm_dpo/train_*.log
```

### 停止训练

如果训练在后台运行：

```bash
# 方法1：使用PID文件
PID=$(cat logs/training/safe_mm_dpo/train_*.pid)
kill $PID

# 方法2：查找进程
ps aux | grep train_safe_mm_dpo
kill <PID>

# 方法3：使用pkill
pkill -f train_safe_mm_dpo
```

### 恢复训练

当前实现不支持从 checkpoint 恢复，需要重新开始训练。如果需要恢复功能，可以：
1. 修改代码添加 `--resume_from_checkpoint` 参数
2. 或手动加载 checkpoint 并继续训练

---

## 常见问题

### Q1: 显存不足（OOM）

**解决方案**：
1. 减小 `batch_size`（如从 4 改为 2）
2. 减小 `max_length`（如从 512 改为 384）
3. 增加 `gradient_accumulation_steps`（保持有效 batch size）
4. 确保启用 LoRA（`use_lora: true`）
5. 使用更多 GPU（分布式训练）

### Q2: 训练速度慢

**解决方案**：
1. 使用多GPU分布式训练
2. 减小 `max_length`（如果数据允许）
3. 使用 LoRA 而不是全参数微调
4. 检查数据加载是否成为瓶颈

### Q3: Lambda 变化过快或过慢

**解决方案**：
1. 调整 `λ_lr`（默认 0.01）
   - 过快：减小到 0.001
   - 过慢：增大到 0.04
2. 检查 `cost_threshold` 设置
3. 监控 `J_C` 的值是否合理

### Q4: 数据格式错误

**错误信息**：`KeyError` 或 `ValueError` 关于数据字段

**解决方案**：
1. 检查原始数据是否包含必需字段
2. 重新运行数据预处理脚本
3. 验证输出数据格式：
   ```bash
   head -n 1 data/train/helpful_pairs.jsonl | python3 -m json.tool
   ```

### Q5: 模型文件找不到

**错误信息**：`FileNotFoundError` 或 `OSError` 关于模型路径

**解决方案**：
1. 检查模型路径是否正确（在配置文件中）
2. 确保 RM 模型已训练完成
3. 检查模型目录结构：
   ```bash
   ls -lh models/helpful_rm/
   ls -lh models/harmless_rm/
   ```

### Q6: 训练中断后如何继续

**当前限制**：代码不支持从 checkpoint 恢复

**临时解决方案**：
1. 保存最后一个 epoch 的 checkpoint
2. 修改代码添加恢复功能
3. 或重新开始训练（如果时间允许）

### Q7: Beta 值异常

**现象**：beta 值过大或过小

**解决方案**：
1. 代码已实现约束：`β ∈ [β_ori, (1+w)β_ori]`
2. 检查 `w` 和 `k` 参数是否合理（默认 0.5）
3. 监控 delta 的值是否正常

---

## 完整训练流程示例

### 快速开始（完整流程）

```bash
# 1. 数据处理
python -m src.scripts.prepare_data \
    --input_dir data/raw \
    --output_dir data \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1

# 2. 训练 Helpful RM（后台运行）
bash scripts/train_rm_distributed.sh helpful 8 --background

# 3. 等待 Helpful RM 训练完成后，训练 Harmless RM（后台运行）
bash scripts/train_rm_distributed.sh harmless 8 --background

# 4. 等待 Harmless RM 训练完成后，训练 DPO（后台运行）
bash scripts/train_dpo_distributed.sh 8 --background

# 5. 监控训练
bash scripts/check_training.sh
tail -f logs/training/safe_mm_dpo/train_*.log
```

### 预计时间（8x64GB GPU）

- 数据处理：5-10 分钟
- Helpful RM 训练：2-4 小时（1 epoch）
- Harmless RM 训练：2-4 小时（1 epoch）
- DPO 训练：8-16 小时（2 epochs）

**总计**：约 12-24 小时

---

## 输出文件结构

训练完成后，项目结构如下：

```
nlp_align/
├── data/
│   ├── train/
│   │   ├── helpful_pairs.jsonl
│   │   └── harmless_pairs.jsonl
│   ├── val/
│   │   ├── helpful_pairs.jsonl
│   │   └── harmless_pairs.jsonl
│   └── test/
│       ├── helpful_pairs.jsonl
│       └── harmless_pairs.jsonl
├── models/
│   ├── base/
│   │   └── Mistral-7B-v0.1/
│   ├── helpful_rm/
│   ├── harmless_rm/
│   └── aligned/
│       ├── epoch_1/
│       └── epoch_2/
└── logs/
    └── training/
        ├── rm/
        │   ├── helpful_*.log
        │   └── harmless_*.log
        └── safe_mm_dpo/
            └── train_*.log
```

---

## 模型评估

训练完成后，可以使用 BeaverTails Safety Benchmark 评估模型的安全性：

```bash
# 1. 准备安全基准数据
python scripts/prepare_safety_benchmark.py \
    --source template \
    --output_path data/benchmarks/safety_benchmark.jsonl

# 2. 运行安全评估
python -m src.evaluation.evaluate_safety \
    --model_path models/aligned/epoch_2 \
    --harmless_rm_path models/harmless_rm \
    --benchmark_path data/benchmarks/safety_benchmark.jsonl \
    --output_path results/safety_evaluation.json
```

**详细说明**：请参考 [SAFETY_EVALUATION_GUIDE.md](SAFETY_EVALUATION_GUIDE.md)

---

## 参考文档

- 配置文件：`configs/rm_config.yaml`, `configs/dpo_config.yaml`
- 训练脚本：`scripts/train_rm_distributed.sh`, `scripts/train_dpo_distributed.sh`
- 监控脚本：`scripts/check_training.sh`
- **安全评估指南**：`SAFETY_EVALUATION_GUIDE.md`

---

## 技术支持

如遇到问题，请检查：
1. 日志文件中的错误信息
2. 配置文件中的路径和参数
3. GPU 显存使用情况（`nvidia-smi`）
4. 数据格式是否正确

祝训练顺利！🎉
