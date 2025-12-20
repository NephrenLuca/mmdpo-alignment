# Lambda 更新机制实现说明

## 问题回顾

之前的实现导致 lambda 过快归零的原因：

```python
J_C = -delta_S.mean()  # delta_S通常>0（chosen更安全），所以J_C通常<0
lambda_param = lambda_param + λ_lr * J_C  # lambda持续减小
```

**问题**：没有考虑策略的实际偏好，只基于数据中的安全性差异。

## 正确的实现（根据 lambda_update_policy.md）

### 1. Log空间更新机制

根据论文，lambda 应该在 log 空间更新，确保始终为正：

```python
# 初始化
lambda_log = ln(λ_init)
lambda_param = λ_init

# 更新（每个step）
ln λ' = ln λ + α · λ · J_C
λ' = exp(ln λ') = λ · exp(α · λ · J_C)
```

**优势**：
- Lambda 始终为正（exp 函数保证）
- 更新更稳定（log 空间下的梯度更平滑）

### 2. J_C 的估计

在 DPO 框架下，`J_C(θ) = E_{x,y~π_θ}[C(y,x)]` 不能直接计算，我们使用以下近似：

```python
# 计算策略偏好和安全性方向
log_ratio = logp_w - logp_l  # 策略对chosen的偏好（>0表示偏好chosen）
delta_S = r_safe_w - r_safe_l  # 安全性差异（>0表示chosen更安全）

# 策略偏好与安全性方向的对齐度
alignment = sign(log_ratio) * sign(delta_S)  # +1对齐，-1不对齐

# J_C估计
J_C = -alignment * |delta_S|
```

**物理含义**：
- `alignment = +1`：策略偏好与安全性方向一致 → 策略安全 → `J_C < 0`
- `alignment = -1`：策略偏好与安全性方向不一致 → 策略危险 → `J_C > 0`

### 3. 滑动平均平滑

使用指数移动平均平滑 J_C，减少噪声：

```python
J_C_ema = 0.95 * J_C_ema + 0.05 * J_C_batch
```

### 4. 完整更新流程

```python
# 初始化
lambda_log = ln(λ_init)
lambda_param = λ_init
J_C_ema = None

# 每个训练step
log_ratio = logp_w - logp_l
delta_S = r_safe_w - r_safe_l

# 计算J_C
alignment = sign(log_ratio) * sign(delta_S)
J_C_batch = (-alignment * abs(delta_S)).mean()

# 滑动平均
J_C_ema = 0.95 * J_C_ema + 0.05 * J_C_batch

# Log空间更新
lambda_log = lambda_log + α * lambda_param * J_C_ema
lambda_param = exp(lambda_log)

# 可选：限制范围
lambda_param = clamp(lambda_param, min=1e-6, max=10.0)
lambda_log = ln(lambda_param)
```

## 预期行为

### Lambda 的变化趋势

1. **训练初期**：
   - 如果策略还不够安全（偏好与安全性不对齐），`J_C > 0`
   - Lambda 增大，加强安全性约束

2. **训练中期**：
   - 策略逐渐学习到偏好安全的回答，`J_C` 接近 0 或为负
   - Lambda 根据策略的实际安全性动态调整

3. **训练后期**：
   - 如果策略已经很安全（`J_C < 0`），Lambda 缓慢减小
   - 但不会过快归零，因为使用了 log 空间更新和滑动平均

### 与之前实现的对比

| 指标 | 旧实现 | 新实现 |
|------|--------|--------|
| **更新方式** | 线性更新 | Log空间更新 |
| **J_C估计** | 只考虑数据 | 考虑策略偏好 |
| **稳定性** | 波动大 | 使用滑动平均 |
| **Lambda趋势** | 快速归零 | 动态平衡 |

## 监控指标

训练日志现在会显示：

```
Epoch 1 step 10 loss_H=0.1234 loss_S=0.0567 KL=0.0123 lambda=0.9876 J_C=-0.0234 delta_S_mean=0.1234
```

- **lambda**：当前拉格朗日乘子值
- **J_C**：平滑后的有害性成本估计
- **delta_S_mean**：平均安全性奖励边际

通过这些指标可以观察：
1. Lambda 是否合理变化（不会过快归零）
2. J_C 的符号和大小（反映策略安全性）
3. Delta_S 的值（反映数据中的安全性差异）

## 配置建议

```yaml
λ_init: 1.0        # 初始值，可以保持不变
λ_lr: 0.01         # 更新学习率，建议使用0.01或更小（如0.005）
```

**注意**：由于使用了 log 空间更新和滑动平均，lambda 的更新会更加稳定，可以使用较大的 `λ_lr`（如 0.01）。

## 验证

训练时可以观察：

1. **Lambda 应该**：
   - 初始阶段可能增大（如果策略不够安全）
   - 然后根据策略安全性动态调整
   - 不会过快归零（在第一个 epoch 就接近 0）

2. **J_C 应该**：
   - 在训练初期可能为正或接近 0
   - 随着训练，逐渐变为负数（策略变得更安全）
   - 波动较小（由于滑动平均）

3. **Loss 应该**：
   - `loss_H` 和 `loss_S` 都应该下降
   - 总损失应该收敛
