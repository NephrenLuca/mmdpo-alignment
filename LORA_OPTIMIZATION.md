# LoRA 显存优化方案

## 什么是 LoRA？

LoRA (Low-Rank Adaptation) 是一种**参数高效微调（PEFT）**方法，它：

1. **冻结原始模型参数**，不参与梯度更新
2. **只训练少量低秩矩阵**（通常<1%的参数）
3. **在推理时动态合并**LoRA权重到原始模型

## 显存优势

使用LoRA训练7B模型的显存占用对比：

| 组件 | 全参数微调 | LoRA微调 | 节省 |
|------|-----------|---------|------|
| 模型参数（bf16） | ~14GB | ~14GB | 0GB（仍需加载，但不训练） |
| **优化器状态（fp32）** | **~28GB** | **~0.1-0.5GB** | **~27GB** ✅ |
| **梯度（bf16）** | **~14GB** | **~0.1-0.5GB** | **~13GB** ✅ |
| 激活值 | 取决于batch_size | 取决于batch_size | 相同 |
| **总计（batch_size=4）** | **~61GB** | **~19-23GB** | **~40GB** ✅ |

**LoRA可以节省约40GB显存！**

## 配置说明

### 推荐配置（LoRA + 8×64GB GPU）

```yaml
use_lora: true
lora:
  r: 16           # LoRA rank，推荐8-32，越大参数越多效果可能越好但显存占用也更高
  alpha: 32       # LoRA alpha，通常设为r的2倍
  dropout: 0.1    # LoRA dropout
batch_size: 4     # 可以比全参数微调使用更大的batch_size
max_length: 512   # 可以使用更长的序列
gradient_accumulation_steps: 2
```

### LoRA参数选择

- **r (rank)**：控制LoRA矩阵的秩，决定可训练参数数量
  - `r=8`：最少参数，最少显存（~0.05GB优化器状态）
  - `r=16`：平衡选择（推荐）
  - `r=32`：更多参数，可能效果更好（~0.2GB优化器状态）
  - `r=64+`：接近全参数微调，不推荐

- **alpha**：LoRA缩放因子，通常设为`r * 2`
  - `alpha=16` for `r=8`
  - `alpha=32` for `r=16`
  - `alpha=64` for `r=32`

- **target_modules**：指定应用LoRA的模块
  - 默认：`["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
  - 对于Mistral，这个默认值已经很好

## 使用方法

在 `configs/rm_config.yaml` 中设置：

```yaml
use_lora: true
lora:
  r: 16
  alpha: 32
  dropout: 0.1
```

然后正常启动训练：

```bash
bash scripts/train_rm_distributed.sh helpful 8
```

## 预期效果

使用LoRA（r=16）：
- **可训练参数**：~4-8M（约0.1%的总参数）
- **优化器状态显存**：从28GB降到~0.2GB
- **梯度显存**：从14GB降到~0.2GB
- **总显存占用**：从~61GB降到~19-23GB
- **可以使用更大的batch_size和max_length**：batch_size=4, max_length=512

## 模型保存

使用LoRA训练的模型会保存为：
- **adapter权重**：只有LoRA的参数（几MB大小）
- **原始模型**：需要单独保存或使用原始checkpoint

加载时需要同时加载原始模型和adapter权重。

## 性能影响

- **训练速度**：LoRA训练速度通常与全参数微调相近或更快（因为参数更少）
- **模型效果**：LoRA在大多数任务上能达到全参数微调90-95%的效果
- **推理速度**：可以将LoRA权重合并到原始模型，推理速度与全参数模型相同
