# Safe RLHF 论文实现说明

根据论文 **2310.12773v1 (Safe RLHF: Safe Reinforcement Learning from Human Feedback)** 对训练代码进行了修改。

## 主要修改

### 1. Lambda 更新机制（论文公式 31）

**论文公式**：
```
ln λ_{k+1} = ln λ_k + α · λ_k · J_C(θ_k)
```

**实现**：
- 使用 log 空间更新 lambda，避免数值不稳定
- 使用指数移动平均（EMA）平滑 J_C 估计，减少噪声波动
- 平滑系数：0.95（论文未明确给出，根据经验设置）

### 2. J_C 估计（论文公式 232）

**论文定义**：
```
J_C(θ) = E_{x~D, y~π_θ}[C_ψ(y, x)] + d
```

其中：
- `C_ψ(y, x)` 是 Cost Model 的输出（值越高表示越有害）
- `d` 是 threshold（论文 Table 4 中显示为 threshold (-d)）

**实现**：
- 在 DPO 框架下，使用策略对 chosen/rejected 的相对偏好来估计期望 cost
- 策略偏好概率：`P(chosen) = softmax([logp_w, logp_l])[0]`
- 期望 cost：`E[C] = P(chosen) * cost_w + P(rejected) * cost_l`
- J_C：`J_C = E[C] + d`

**注意**：
- `harmless_rm` 输出的是 safety score（越高越安全）
- 需要转换为 cost：`cost = -safety_score`（越高越有害）

### 3. Threshold 参数

**论文 Table 4 中的值**：
- Beaver-v1: `threshold (-d) = 0` → `d = 0`
- Beaver-v2: `threshold (-d) = -3` → `d = 3`
- Beaver-v3: `threshold (-d) = -3` → `d = 3`

**配置**：
在 `configs/dpo_config.yaml` 中添加了 `cost_threshold` 参数：
```yaml
cost_threshold: 0.0  # Threshold d in J_C = E[C(y,x)] + d
```

### 4. Lambda 初始值和学习率

**论文 Table 4 中的值**：
- Beaver-v1: `λ_init = 1`, `λ_lr = 0.01`
- Beaver-v2: `λ_init = 0.5`, `λ_lr = 0.04`
- Beaver-v3: `λ_init = 1`, `λ_lr = 0.04`

**当前配置**：
```yaml
λ_init: 1.0
λ_lr: 0.01
```

## 未实现的特性

### Cost Model 的 Classification Loss

**论文公式 6**：
```
L_C(θ; D_C) = -E[log σ(C_ψ(y_w, x) - C_ψ(y_l, x))]
              -E[log σ(s_w · C_ψ(y_w, x)) + log σ(s_l · C_ψ(y_l, x))]
```

其中 `s(y) ∈ {+1, -1}` 是 safety label（+1 表示有害，-1 表示无害）。

**原因**：
- 当前数据格式只包含 `prompt`, `chosen_response`, `rejected_response`
- 没有 safety labels（`s(y)`），无法实现 classification terms
- 当前 RM 训练只使用 pairwise comparison loss

**影响**：
- Cost Model 可能无法像论文中那样明确区分安全/不安全簇
- 但 pairwise comparison 仍然可以学习相对安全性偏好

## 代码修改位置

### 1. `src/training/train_safe_mm_dpo.py`

- **DPOConfig**：添加 `cost_threshold` 参数
- **J_C 估计**：使用 Cost Model 输出和策略偏好概率
- **Lambda 更新**：使用 log 空间更新和 EMA 平滑

### 2. `configs/dpo_config.yaml`

- 添加 `cost_threshold: 0.0` 参数

### 3. RM 训练代码

- **未修改**：当前实现只使用 pairwise comparison loss
- 如果需要实现完整的 Cost Model 训练，需要：
  1. 数据格式包含 safety labels
  2. 修改 `train_rm.py` 添加 classification loss
  3. 修改 `PreferenceDataset` 读取 safety labels

## 使用建议

1. **Threshold 调参**：
   - 如果模型过于保守（拒绝回答过多），可以增大 `cost_threshold`（如 3.0）
   - 如果模型不够安全，可以减小 `cost_threshold`（如 0.0 或负值）

2. **Lambda 学习率**：
   - 如果 lambda 变化过快，可以减小 `λ_lr`（如 0.001）
   - 如果 lambda 变化过慢，可以增大 `λ_lr`（如 0.04）

3. **监控指标**：
   - `J_C`：应该逐渐接近 0（满足约束）
   - `lambda`：应该根据 J_C 动态调整
   - `delta_S_mean`：安全性差异的平均值

## 参考文献

- Dai, J., Pan, X., Sun, R., et al. (2023). Safe RLHF: Safe Reinforcement Learning from Human Feedback. arXiv:2310.12773v1
