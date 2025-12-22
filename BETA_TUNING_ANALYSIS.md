# Beta 调参实现检查报告

根据论文 **2502.10391v1 (MM-RLHF: The Next Step Forward in Multimodal LLM Alignment)** 检查代码实现。

## 论文要求

### 1. 公式（论文公式 188）

```
β(d) = β_ori(1 + w(1 - e^(-kd)))
```

其中：
- `d = r(yw) - r(yl)` 是 reward margin（奖励边际）
- `β_ori` 是初始默认缩放因子（论文中设为 0.1）
- `w` 是平衡动态组件贡献的参数（默认 0.5）
- `k` 是调整 β(d) 对 d 变化敏感度的超参数（默认 0.5）

### 2. 约束（论文第 189 行）

**重要**：论文明确要求将 β(d) 约束在 `[β_ori, (1+w)β_ori]` 范围内，以避免过于激进的更新导致训练不稳定。

### 3. 超参数调参建议（论文第 948 行）

- 默认值：`w = 0.5`, `k = 0.5`（论文实验证明效果良好）
- 调参建议：两个超参数不能同时太大或太小
- β_ori：论文中设为 0.1，因为动态调整，无需手动调参

## 代码实现检查

### ✅ 正确部分

1. **公式实现**（`src/training/train_safe_mm_dpo.py:288`）：
```python
return β_ori * (1.0 + w * (1.0 - torch.exp(-k * delta)))
```
✅ 与论文公式完全一致

2. **delta 计算**（`src/training/train_safe_mm_dpo.py:506, 516`）：
```python
delta_H = r_helpful_w - r_helpful_l
delta_S = safety_score_w - safety_score_l
```
✅ 与论文定义一致：`d = r(yw) - r(yl)`

3. **超参数默认值**（`configs/dpo_config.yaml`）：
```yaml
w: 0.5
k: 0.5
β_ori: 0.1
```
✅ 与论文推荐值一致

### ❌ 问题部分

1. **缺少 beta 约束**：
   - **问题**：代码中没有实现论文要求的约束 `β(d) ∈ [β_ori, (1+w)β_ori]`
   - **影响**：当 delta 为负数时，beta 可能小于 β_ori，违反论文约束
   - **位置**：`src/training/train_safe_mm_dpo.py:282-288`

2. **delta 为负数的处理**：
   - **问题**：如果数据有噪声，chosen 可能比 rejected 更差，导致 delta < 0
   - **当前行为**：当 delta < 0 时，`exp(-k*delta)` 会很大，导致 `1 - exp(-k*delta)` 为负
   - **结果**：beta 可能小于 β_ori，甚至为负（如果 delta 非常负）
   - **论文意图**：论文 Figure 5 显示 d 从 0 到 10，暗示 d 应该是正数

## 修复建议

### 修复 1：添加 beta 约束

```python
def dynamic_beta(β_ori: float, w: float, k: float, delta: torch.Tensor) -> torch.Tensor:
    """
    MM-DPO 动态缩放因子（论文公式 188）：
        β(δ) = β_ori * (1 + w * (1 - exp(-k * δ)))
    
    约束：β(δ) ∈ [β_ori, (1+w)β_ori]（论文第 189 行）
    
    注意：如果 delta < 0（数据噪声），使用 |delta| 或 clamp 到最小值
    """
    # 使用 delta 的绝对值，确保 beta 随 |delta| 增大而增大
    # 或者使用 max(delta, 0) 只对正 delta 应用动态缩放
    delta_abs = torch.abs(delta)  # 或 delta.clamp(min=0)
    
    beta = β_ori * (1.0 + w * (1.0 - torch.exp(-k * delta_abs)))
    
    # 应用论文约束：β(d) ∈ [β_ori, (1+w)β_ori]
    beta_min = β_ori
    beta_max = β_ori * (1.0 + w)
    beta = beta.clamp(min=beta_min, max=beta_max)
    
    return beta
```

### 修复 2：处理负 delta 的策略

**方案 A：使用绝对值**（推荐）
- 优点：简单，对噪声数据更鲁棒
- 缺点：可能对负 delta 也给予高权重

**方案 B：只对正 delta 应用动态缩放**
- 优点：符合论文意图（d 应该是正数）
- 缺点：负 delta 的样本权重固定为 β_ori

**方案 C：使用 max(delta, 0)**
- 优点：结合方案 A 和 B
- 缺点：负 delta 完全被忽略

## 当前配置检查

### `configs/dpo_config.yaml`

```yaml
w: 0.5        # ✅ 与论文默认值一致
k: 0.5        # ✅ 与论文默认值一致
β_ori: 0.1    # ✅ 与论文默认值一致
```

✅ 超参数配置正确

## 总结

| 项目 | 状态 | 说明 |
|------|------|------|
| 公式实现 | ✅ | 与论文完全一致 |
| delta 计算 | ✅ | 与论文定义一致 |
| 超参数默认值 | ✅ | 与论文推荐值一致 |
| **beta 约束** | ❌ | **缺少约束，需要修复** |
| **负 delta 处理** | ⚠️ | **需要明确策略** |

## 建议的修复

1. **立即修复**：添加 beta 约束 `[β_ori, (1+w)β_ori]`
2. **考虑修复**：明确负 delta 的处理策略（建议使用绝对值或 clamp 到 0）
3. **验证**：在训练过程中监控 beta 的值，确保在合理范围内
