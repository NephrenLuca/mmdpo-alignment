# Safe-MM-DPO 训练：LoRA + 多GPU 指南

## 概述

Safe-MM-DPO 训练现在支持：
- ✅ **LoRA参数高效微调**：大幅降低显存占用
- ✅ **多GPU分布式训练**：充分利用多块GPU

## 快速开始

### 1. 配置LoRA（推荐）

编辑 `configs/dpo_config.yaml`：

```yaml
use_lora: true
lora:
  r: 16
  alpha: 32
  dropout: 0.1

batch_size: 4            # LoRA模式下可以使用较小的batch_size
gradient_accumulation_steps: 4
```

### 2. 启动多GPU训练

```bash
bash scripts/train_dpo_distributed.sh 8
```

或使用 `torchrun` 直接启动：

```bash
torchrun --nproc_per_node=8 --master_port=29501 \
    -m src.training.train_safe_mm_dpo \
    --config configs/dpo_config.yaml \
    --output_dir models/aligned \
    --logging_dir logs/training/safe_mm_dpo
```

## 显存优化

### 全参数微调 vs LoRA微调

Safe-MM-DPO训练需要同时加载：
- **策略模型（policy_model）**：可训练
- **参考模型（ref_model）**：只前向，不训练
- **Helpful-RM**：只前向，不训练
- **Harmless-RM**：只前向，不训练

#### 全参数微调（不推荐）

| 组件 | 显存占用 |
|------|---------|
| 策略模型（bf16） | ~14GB |
| 优化器状态（fp32） | ~28GB ⚠️ |
| 梯度（bf16） | ~14GB ⚠️ |
| 参考模型（bf16） | ~14GB |
| 2个RM（bf16） | ~28GB |
| 激活值 | ~5-10GB |
| **总计** | **~103GB** ❌ 远超64GB |

#### LoRA微调（⭐推荐）

| 组件 | 显存占用 |
|------|---------|
| 策略模型（bf16） | ~14GB（只加载，大部分不训练） |
| **优化器状态（fp32）** | **~0.2GB** ✅ |
| **梯度（bf16）** | **~0.2GB** ✅ |
| 参考模型（bf16） | ~14GB |
| 2个RM（bf16） | ~28GB |
| 激活值 | ~5-10GB |
| **总计** | **~61-66GB** ✅ 接近但可能仍紧张 |

**LoRA可以节省约40GB显存！**

### 推荐配置（8×64GB GPU）

```yaml
use_lora: true
lora:
  r: 16
  alpha: 32
  dropout: 0.1
batch_size: 2-4          # 根据显存调整
max_length: 512          # 根据显存调整
gradient_accumulation_steps: 4-8
```

**预期显存占用**：~55-65GB per GPU

### 如果仍然OOM

1. **减小batch_size**：从4降到2或1
2. **增加gradient_accumulation_steps**：从4增加到8或16
3. **减小max_length**：从512降到384或256
4. **降低LoRA rank**：从16降到8

## 多GPU训练

### 工作原理

1. **数据并行**：使用 `DistributedSampler` 将数据分片到各个GPU
2. **模型并行**：只有策略模型使用DDP（可训练），其他模型在每个GPU上复制
3. **梯度同步**：DDP自动在所有GPU间同步策略模型的梯度

### 有效批次大小

**有效批次大小 = batch_size × gradient_accumulation_steps × num_gpus**

例如：
- `batch_size=4` × `gradient_accumulation_steps=4` × `8 GPUs` = **有效批次大小 128**

### 配置建议

对于8×64GB GPU：

```yaml
batch_size: 2-4
gradient_accumulation_steps: 4-8
# 有效batch_size = 2-4 × 4-8 × 8 = 64-256
```

## 训练监控

训练过程中会输出：

```
Distributed training initialized: world_size=8
LoRA enabled for policy model: 4,194,304 trainable / 7,240,192,000 total parameters (0.06%)
GPU memory allocated: 58.23 GB
GPU memory reserved: 59.48 GB
Effective batch size: 128 (batch_size=4 x accumulation=4 x 8 GPUs)
Epoch 1 step 10 loss_H=0.1234 loss_S=0.0567 KL=0.0123 lambda=1.0234
```

### 关键指标

- **loss_H**：帮助性损失（应该下降）
- **loss_S**：安全性损失（应该下降）
- **KL**：策略与参考模型的KL散度（应该保持较小）
- **lambda**：拉格朗日乘子（动态调整，平衡helpful和safety）

## 模型保存

使用LoRA训练的模型会保存为：
- **adapter权重**：只有LoRA的参数（几MB）
- 保存路径：`models/aligned/epoch_1/`, `models/aligned/epoch_2/`

加载时需要：
1. 加载原始Mistral-7B模型
2. 加载LoRA adapter权重

## 常见问题

### Q: 为什么DPO训练需要更多显存？

A: DPO训练需要同时加载：
- 策略模型（可训练）
- 参考模型（只前向）
- 2个奖励模型（只前向）

即使使用LoRA，总显存占用仍然较高。

### Q: 可以使用更小的LoRA rank吗？

A: 可以，但可能影响效果：
- `r=8`：最少显存，但效果可能略差
- `r=16`：平衡选择（推荐）⭐
- `r=32`：更多参数，可能效果更好，但显存占用更高

### Q: 参考模型和RM也需要LoRA吗？

A: 不需要：
- 参考模型和RM只在eval模式下运行
- 它们不参与梯度计算，不需要优化器状态
- 使用LoRA反而会增加复杂度

## 性能对比

| 配置 | 显存占用 | 训练速度 | 推荐度 |
|------|---------|---------|--------|
| 全参数微调 | ~103GB | 慢 | ❌ 不推荐 |
| LoRA (r=16) | ~55-65GB | 快 | ⭐⭐⭐⭐⭐ |
| LoRA (r=8) | ~50-60GB | 快 | ⭐⭐⭐⭐ |

## 下一步

训练完成后，模型保存在 `models/aligned/` 目录下，可以用于：
1. 推理测试
2. 进一步评估
3. 知识蒸馏（可选）

详细说明请参考：
- [LORA_OPTIMIZATION.md](LORA_OPTIMIZATION.md) - LoRA详细说明
- [MULTI_GPU_TRAINING.md](MULTI_GPU_TRAINING.md) - 多GPU训练说明
