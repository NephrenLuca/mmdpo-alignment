# 显存优化说明

本文档说明针对 8×64GB GPU 环境所做的显存优化措施。

## 问题分析

训练 Mistral-7B Reward Model 时，即使使用半精度（bf16/fp16），每个GPU的显存占用仍然接近上限：

- **模型参数**（bf16）：~14GB
- **优化器状态**（fp32，AdamW需要momentum和variance）：~28GB
- **梯度**（bf16）：~14GB  
- **激活值**（取决于batch_size和max_length）：~10-20GB

**总计约 66-76GB**，超过了单卡 64GB 容量。

## 优化措施

### 1. 梯度累积（Gradient Accumulation）

通过梯度累积可以**减小每个step的batch size**，从而减少激活值显存占用：

```yaml
batch_size: 2                    # 每个GPU的batch size
gradient_accumulation_steps: 4   # 梯度累积步数
# 有效batch_size = 2 × 4 × 8 GPUs = 64
```

- **优势**：激活值显存降低到原来的 1/4（从batch_size=8降到batch_size=2）
- **代价**：每个训练step的时间略增（但总训练时间接近，因为总的有效batch size相同）

### 2. 减小序列长度

```yaml
max_length: 512  # 从1024降到512
```

序列长度对激活值显存是**平方关系**，减小一半可以显著降低显存占用。

### 3. 环境变量优化

启动脚本中添加了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，可以减少显存碎片，提高显存利用率。

### 4. 半精度训练

模型和梯度使用 bf16/fp16，虽然优化器状态仍然是fp32，但已是最佳实践。

## 推荐配置

### Reward Model 训练（8×64GB GPU）

#### 方案A：全参数微调（不推荐，显存紧张）

```yaml
batch_size: 1
max_length: 256
gradient_accumulation_steps: 16
use_lora: false
```

**预期显存占用**：~59-61GB（接近上限，容易OOM）

#### 方案B：LoRA微调（⭐推荐）

```yaml
use_lora: true
lora:
  r: 16
  alpha: 32
  dropout: 0.1
batch_size: 4
max_length: 512
gradient_accumulation_steps: 2
learning_rate: 2e-5  # 或1e-4
```

**有效batch_size** = 4 × 2 × 8 = 64（保持不变）

**预期显存占用**：
- 模型参数（bf16）：~14GB（只加载，不训练）
- 优化器状态（fp32）：~0.2GB（只训练LoRA参数） ✅
- 梯度（bf16）：~0.2GB ✅
- 激活值（batch_size=4, max_length=512）：~5-8GB
- **总计约 19-23GB** ✅✅✅

**LoRA可以节省约40GB显存！**

详细说明请查看 [LORA_OPTIMIZATION.md](LORA_OPTIMIZATION.md)

### Safe-MM-DPO 训练（8×64GB GPU）

```yaml
batch_size: 1-2
max_length: 512
gradient_accumulation_steps: 4-8
learning_rate: 5e-7
```

**预期显存占用**：由于需要同时加载policy、ref和2个RM，显存压力更大，建议使用更小的batch_size。

## 进一步优化（如果仍OOM）

如果仍然出现OOM，可以尝试：

1. **进一步减小batch_size**：从2降到1
2. **增加gradient_accumulation_steps**：从4增加到8或16
3. **进一步减小max_length**：从512降到384或256
4. **使用CPU offload**：将优化器状态offload到CPU（会显著降低训练速度）
5. **使用8-bit优化器**：使用 `bitsandbytes` 库的8-bit AdamW（需要额外依赖）

## 验证显存使用

训练时可以监控显存使用：

```bash
watch -n 1 nvidia-smi
```

或在Python代码中添加：

```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```
