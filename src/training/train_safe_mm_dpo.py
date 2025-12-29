"""
Safe-MM-DPO training script.

核心目标：
- 使用两个奖励模型（Helpful-RM / Harmless-RM）分别刻画“有帮助性”和“无害性”；
- 在 DPO 框架上引入 MM-DPO 的动态 β 缩放；
- 使用拉格朗日乘子 λ 动态平衡两个目标（Safe RLHF 思想）。

核心创新点（在实现中有对应注释）：
1. 双评判员：Helpful-RM & Harmless-RM 提供各自的奖励边际 δ_H, δ_S；
2. 动态缩放因子 β(δ)：高置信度样本获得更大权重（MM-DPO 思想）；
3. 拉格朗日乘子 λ：根据有害性成本 J_C 自适应调整，平衡 L_H 与 L_S（Safe RLHF 思想）。

用法（示例）：

    python -m src.training.train_safe_mm_dpo \\
        --config configs/dpo_config.yaml \\
        --output_dir models/aligned \\
        --logging_dir logs/training/safe_mm_dpo
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import yaml

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    TaskType = None

from src.models.reward_model import load_reward_model, RewardModelConfig


class PreferenceDataset(Dataset):
    """
    Dataset of preference pairs (prompt, chosen, rejected) stored in jsonl.
    """

    def __init__(self, path: Path, tokenizer, max_length: int) -> None:
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: List[Dict] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        rec = self.records[idx]
        prompt = rec["prompt"]
        chosen = rec["chosen_response"]
        rejected = rec["rejected_response"]

        # 这里简单地将 prompt + 响应拼接作为条件序列
        text_w = f"{prompt}\n{chosen}"
        text_l = f"{prompt}\n{rejected}"

        tok_w = self.tokenizer(
            text_w,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok_l = self.tokenizer(
            text_l,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "chosen_input_ids": tok_w["input_ids"].squeeze(0),
            "chosen_attention_mask": tok_w["attention_mask"].squeeze(0),
            "rejected_input_ids": tok_l["input_ids"].squeeze(0),
            "rejected_attention_mask": tok_l["attention_mask"].squeeze(0),
        }


def collate_fn(tokenizer):
    def _collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        def pad(key: str, pad_value: int) -> torch.Tensor:
            return torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch],
                batch_first=True,
                padding_value=pad_value,
            )

        return {
            "chosen_input_ids": pad("chosen_input_ids", tokenizer.pad_token_id),
            "chosen_attention_mask": pad("chosen_attention_mask", 0),
            "rejected_input_ids": pad("rejected_input_ids", tokenizer.pad_token_id),
            "rejected_attention_mask": pad("rejected_attention_mask", 0),
        }

    return _collate


@dataclass
class DPOConfig:
    λ_init: float
    w: float
    k: float
    β_ori: float
    learning_rate: float
    batch_size: int
    kl_coeff: float
    λ_lr: float
    epochs: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    warmup_steps: int
    max_length: int  # Maximum sequence length for tokenization
    cost_threshold: float = 0.0  # Threshold d in J_C = E[C(y,x)] + d (论文中的threshold (-d))

    policy_model_path: str
    ref_model_path: str
    helpful_rm_path: str
    harmless_rm_path: str

    train_helpful_path: Path
    train_harmless_path: Path
    val_helpful_path: Path
    val_harmless_path: Path

    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] | None = None


def load_dpo_config(path: Path) -> DPOConfig:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Handle LoRA config
    use_lora = cfg.get("use_lora", False)
    lora_config = cfg.get("lora", {}) if use_lora else {}

    # Get max_length from config, with a safe default
    max_length = cfg.get("max_length", 512)
    if max_length is None or max_length > 32767:  # Safe limit for tokenizer
        max_length = 512
        print(f"Warning: max_length in config is invalid or too large, using default {max_length}")

    return DPOConfig(
        λ_init=float(cfg["λ_init"]),
        w=float(cfg["w"]),
        k=float(cfg["k"]),
        β_ori=float(cfg["β_ori"]),
        learning_rate=float(cfg["learning_rate"]),
        batch_size=int(cfg["batch_size"]),
        kl_coeff=float(cfg["kl_coeff"]),
        λ_lr=float(cfg["λ_lr"]),
        epochs=int(cfg["epochs"]),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        warmup_steps=int(cfg.get("warmup_steps", 0)),
        max_length=int(max_length),
        cost_threshold=float(cfg.get("cost_threshold", 0.0)),  # threshold d in J_C = E[C] + d
        policy_model_path=cfg["policy_model_path"],
        ref_model_path=cfg["ref_model_path"],
        helpful_rm_path=cfg["helpful_rm_path"],
        harmless_rm_path=cfg["harmless_rm_path"],
        train_helpful_path=Path(cfg["train_helpful_path"]),
        train_harmless_path=Path(cfg["train_harmless_path"]),
        val_helpful_path=Path(cfg["val_helpful_path"]),
        val_harmless_path=Path(cfg["val_harmless_path"]),
        use_lora=use_lora,
        lora_r=int(lora_config.get("r", 16)),
        lora_alpha=int(lora_config.get("alpha", 32)),
        lora_dropout=float(lora_config.get("dropout", 0.1)),
        lora_target_modules=lora_config.get("target_modules"),
    )


def compute_log_probs(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个样本的平均 token log-prob（近似 log π(y|x)）。

    为简化实现，我们直接对整个序列（包括 prompt）求平均 log-prob。
    理论上 DPO 只需要响应部分，但在大多数实践中这种近似是可接受的。
    """
    # Handle DDP-wrapped models
    model_for_forward = model.module if isinstance(model, DDP) else model
    outputs = model_for_forward(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)
    log_probs = torch.log_softmax(logits, dim=-1)

    # 右移一位，与 labels 对齐
    shift_log_probs = log_probs[:, :-1, :]
    shift_input_ids = input_ids[:, 1:]
    shift_attention_mask = attention_mask[:, 1:]

    # gather log p(token)
    token_logp = shift_log_probs.gather(-1, shift_input_ids.unsqueeze(-1)).squeeze(-1)

    # 按有效 token 求平均
    token_logp = token_logp * shift_attention_mask
    lengths = shift_attention_mask.sum(dim=-1).clamp(min=1)
    seq_logp = token_logp.sum(dim=-1) / lengths
    return seq_logp


def mm_dpo_loss(
    logp_w: torch.Tensor,
    logp_l: torch.Tensor,
    logp_ref_w: torch.Tensor,
    logp_ref_l: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    MM-DPO 损失：
        L = - E[ log σ( β * ( Δθ - Δref ) ) ]
    其中 Δ = log π(y_w|x) - log π(y_l|x)
    """
    delta_theta = logp_w - logp_l
    delta_ref = logp_ref_w - logp_ref_l
    advantage = beta * (delta_theta - delta_ref)
    return -torch.nn.functional.logsigmoid(advantage).mean()


def kl_divergence(
    policy: AutoModelForCausalLM,
    ref: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    近似 KL(π_θ || π_ref)，在一个 batch 上做经验估计。
    """
    # Handle DDP-wrapped policy model
    policy_for_forward = policy.module if isinstance(policy, DDP) else policy
    with torch.no_grad():
        ref_logits = ref(input_ids=input_ids, attention_mask=attention_mask).logits
    pi_logits = policy_for_forward(input_ids=input_ids, attention_mask=attention_mask).logits

    pi_logp = torch.log_softmax(pi_logits, dim=-1)
    ref_logp = torch.log_softmax(ref_logits, dim=-1)

    kl = torch.sum(
        torch.exp(pi_logp) * (pi_logp - ref_logp),
        dim=-1,
    )  # (B, T)
    kl = (kl * attention_mask).sum(dim=-1) / attention_mask.sum(dim=-1).clamp(min=1)
    return kl.mean()


def dynamic_beta(β_ori: float, w: float, k: float, delta: torch.Tensor) -> torch.Tensor:
    """
    MM-DPO 动态缩放因子（论文 2502.10391v1 公式 188）：
        β(δ) = β_ori * (1 + w * (1 - exp(-k * δ)))
    
    高置信度（|δ| 大）的样本拥有更大的 β，从而在训练中权重更高。
    
    约束（论文第 189 行）：β(δ) ∈ [β_ori, (1+w)β_ori]
    这确保训练稳定性，避免过于激进的更新。
    
    注意：如果 delta < 0（数据噪声导致 chosen 比 rejected 更差），
    使用绝对值确保 beta 随 |delta| 增大而增大。
    """
    # 使用 delta 的绝对值，确保 beta 随 |delta| 增大而增大
    # 这处理了数据噪声导致 delta < 0 的情况
    delta_abs = torch.abs(delta)
    
    # 计算动态 beta（论文公式 188）
    beta = β_ori * (1.0 + w * (1.0 - torch.exp(-k * delta_abs)))
    
    # 应用论文约束：β(d) ∈ [β_ori, (1+w)β_ori]（论文第 189 行）
    beta_min = β_ori
    beta_max = β_ori * (1.0 + w)
    beta = beta.clamp(min=beta_min, max=beta_max)
    
    return beta


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Not in distributed mode
        return None, None, None

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_training(
    cfg: DPOConfig,
    output_dir: Path,
    logging_dir: Path,
) -> None:
    # Setup distributed training if available
    rank, world_size, ddp_device = setup_distributed()
    is_ddp = rank is not None
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if is_ddp else 0

    if is_ddp:
        device = ddp_device
        if rank == 0:
            print(f"Distributed training initialized: world_size={world_size}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Single GPU training on device: {device}")

    # ====== 加载策略模型与参考模型 ======
    tokenizer = AutoTokenizer.from_pretrained(cfg.policy_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models with memory optimizations
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    policy_model = AutoModelForCausalLM.from_pretrained(
        cfg.policy_model_path,
        dtype=dtype,  # Use dtype instead of deprecated torch_dtype
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.ref_model_path,
        dtype=dtype,  # Use dtype instead of deprecated torch_dtype
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    policy_model.to(device)
    ref_model.to(device)

    # Enable gradient checkpointing for policy model
    if hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable()
    policy_model.config.use_cache = False

    # Apply LoRA to policy model if requested
    if cfg.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError(
                "LoRA requested but peft is not installed. "
                "Install it with: pip install peft"
            )
        
        # Default target modules for Mistral/Llama architectures
        target_modules = cfg.lora_target_modules
        if target_modules is None:
            # For Mistral, typical target modules are q_proj, k_proj, v_proj, o_proj
            # Also include gate_proj, up_proj, down_proj in MLP layers
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,  # Causal language modeling task
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",  # Don't train bias parameters
        )
        policy_model = get_peft_model(policy_model, lora_config)
        
        # Print trainable parameters info
        if (not is_ddp) or rank == 0:
            trainable_params = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in policy_model.parameters())
            print(f"LoRA enabled for policy model: {trainable_params:,} trainable / {total_params:,} total parameters "
                  f"({100 * trainable_params / total_params:.2f}%)")

    # Wrap policy model with DDP (only trainable model needs DDP)
    if is_ddp:
        policy_model = DDP(policy_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    # Clear cache before training starts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if (not is_ddp) or rank == 0:
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

    # ====== 加载两个奖励模型（双评判员）======
    # Use a safe max_length for RM loading (they don't need the full sequence length)
    safe_max_length = min(cfg.max_length, 1024)  # RMs typically don't need very long sequences
    helpful_rm_cfg = RewardModelConfig(
        base_model_path=cfg.helpful_rm_path,
        tokenizer_name=cfg.helpful_rm_path,
        max_length=safe_max_length,
    )
    harmless_rm_cfg = RewardModelConfig(
        base_model_path=cfg.harmless_rm_path,
        tokenizer_name=cfg.harmless_rm_path,
        max_length=safe_max_length,
    )
    helpful_rm, _ = load_reward_model(helpful_rm_cfg, torch_dtype=dtype, use_gradient_checkpointing=False)
    harmless_rm, _ = load_reward_model(harmless_rm_cfg, torch_dtype=dtype, use_gradient_checkpointing=False)
    helpful_rm.to(device).eval()
    harmless_rm.to(device).eval()
    for p in helpful_rm.parameters():
        p.requires_grad = False
    for p in harmless_rm.parameters():
        p.requires_grad = False

    # ====== 构建数据集（helpful / harmless 共用同一结构）======
    # Use configured max_length instead of tokenizer.model_max_length (which may be too large)
    train_helpful = PreferenceDataset(cfg.train_helpful_path, tokenizer, cfg.max_length)
    train_harmless = PreferenceDataset(cfg.train_harmless_path, tokenizer, cfg.max_length)

    # 合并两个维度的训练数据（简单拼接）
    train_records = train_helpful.records + train_harmless.records
    # 复用 Dataset 逻辑：临时写一个 in-memory 版
    class CombinedDataset(PreferenceDataset):
        def __init__(self, records, tokenizer, max_length):
            self.records = records
            self.tokenizer = tokenizer
            self.max_length = max_length

    train_ds = CombinedDataset(train_records, tokenizer, cfg.max_length)

    # Use DistributedSampler for multi-GPU training
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank) if is_ddp else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn(tokenizer),
        pin_memory=True,
    )

    # Adjust effective batch size info
    effective_batch_size = cfg.batch_size * cfg.gradient_accumulation_steps * (world_size if is_ddp else 1)
    if (not is_ddp) or rank == 0:
        print(f"Effective batch size: {effective_batch_size} (batch_size={cfg.batch_size} x accumulation={cfg.gradient_accumulation_steps} x {world_size if is_ddp else 1} GPUs)")

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=cfg.learning_rate,
    )
    num_training_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 拉格朗日乘子 λ（核心：动态平衡 helpful 与 harmless 损失）
    # 使用log空间存储lambda，确保始终为正且更新稳定
    lambda_log = torch.tensor(float(cfg.λ_init), device=device).log()
    lambda_param = torch.tensor(cfg.λ_init, device=device)

    # 用于平滑J_C估计的滑动平均
    J_C_ema = None

    if (not is_ddp) or rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        policy_model.train()
        for step, batch in enumerate(train_loader, start=1):
            global_step += 1

            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            # ====== 1. 计算策略与参考模型的 log π(y|x) ======
            logp_w = compute_log_probs(policy_model, chosen_ids, chosen_mask)
            logp_l = compute_log_probs(policy_model, rejected_ids, rejected_mask)
            with torch.no_grad():
                logp_ref_w = compute_log_probs(ref_model, chosen_ids, chosen_mask)
                logp_ref_l = compute_log_probs(ref_model, rejected_ids, rejected_mask)

            # ====== 2. 计算奖励边际 δ_H, δ_S（双评判员）======
            with torch.no_grad():
                # Helpful-RM
                r_helpful_w = helpful_rm(chosen_ids, chosen_mask)
                r_helpful_l = helpful_rm(rejected_ids, rejected_mask)
                delta_H = r_helpful_w - r_helpful_l

                # Harmless-RM (Cost Model in Safe RLHF terminology)
                # 注意：harmless_rm输出的是safety score（越高越安全）
                # 在Safe RLHF中，Cost Model输出C(y,x)，值越高表示越有害
                # 因此需要转换：cost = -safety_score
                safety_score_w = harmless_rm(chosen_ids, chosen_mask)  # 安全性分数（越高越安全）
                safety_score_l = harmless_rm(rejected_ids, rejected_mask)
                cost_w = -safety_score_w  # 转换为cost（越高越有害）
                cost_l = -safety_score_l
                delta_S = safety_score_w - safety_score_l  # 仍用于MM-DPO的beta计算（safety score差异）

            # ====== 3. 动态缩放因子 β_H, β_S（MM-DPO 核心）======
            beta_H = dynamic_beta(cfg.β_ori, cfg.w, cfg.k, delta_H)
            beta_S = dynamic_beta(cfg.β_ori, cfg.w, cfg.k, delta_S)

            # ====== 4. 计算两个维度的 MM-DPO 损失 ======
            loss_H = mm_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta_H)
            loss_S = mm_dpo_loss(logp_w, logp_l, logp_ref_w, logp_ref_l, beta_S)

            # 总损失：Safe RLHF 风格的加权和
            # L_Total = 1/(1+λ) * L_H + λ/(1+λ) * L_S
            weight_helpful = 1.0 / (1.0 + lambda_param)
            weight_safety = lambda_param / (1.0 + lambda_param)
            loss_total = weight_helpful * loss_H + weight_safety * loss_S

            # 可选 KL 约束，防止策略偏离参考模型过远
            if cfg.kl_coeff > 0:
                kl = kl_divergence(policy_model, ref_model, chosen_ids, chosen_mask)
                loss_total = loss_total + cfg.kl_coeff * kl
            else:
                kl = torch.tensor(0.0, device=device)

            loss_total = loss_total / cfg.gradient_accumulation_steps
            loss_total.backward()

            if step % cfg.gradient_accumulation_steps == 0:
                # For DDP, we can directly use model.parameters()
                model_params = policy_model.module.parameters() if is_ddp else policy_model.parameters()
                torch.nn.utils.clip_grad_norm_(model_params, cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # ====== 5. 更新拉格朗日乘子 λ（控制安全性约束强度）======
                # 根据论文 2310.12773v1 (Safe RLHF) 的实现
                #
                # 理论依据（论文公式 232, 787）：
                # 1. J_C(θ) = E_{x~D, y~π_θ}[C_ψ(y, x)] + d
                #    其中 C_ψ 是 Cost Model（harmless_rm）的输出，d 是 threshold
                # 2. Lambda更新（公式 31）：ln λ_{k+1} = ln λ_k + α · λ_k · J_C(θ_k)
                # 3. 使用 moving average 来估计 J_C（论文 Appendix B.3）
                #
                # 物理含义：
                # - J_C > 0：策略有害（期望cost > -d），λ增大（加强安全性约束）
                # - J_C < 0：策略安全（期望cost < -d），λ减小（减弱约束，专注有用性）
                # - J_C = 0：刚好满足约束（期望cost = -d），λ不变
                #
                # 在DPO框架下，J_C的估计：
                # 使用策略对chosen/rejected的相对偏好来估计当前策略的期望cost
                # - 策略更偏好chosen时，使用chosen的cost作为权重
                # - 策略更偏好rejected时，使用rejected的cost作为权重
                with torch.no_grad():
                    # 根据论文2310.12773v1，J_C(θ) = E_{x~D, y~π_θ}[C_ψ(y, x)] + d
                    # 其中C_ψ是Cost Model的输出，d是threshold
                    #
                    # 在DPO框架下，我们需要估计当前策略的期望cost
                    # 使用策略对chosen/rejected的相对偏好来估计策略会选择哪个回答
                    
                    # 计算策略对chosen和rejected的相对偏好概率
                    # 使用softmax归一化，得到策略选择chosen和rejected的概率
                    log_probs = torch.stack([logp_w, logp_l], dim=-1)  # (batch, 2)
                    probs = torch.softmax(log_probs, dim=-1)  # (batch, 2)
                    prob_chosen = probs[:, 0]  # 策略选择chosen的概率
                    prob_rejected = probs[:, 1]  # 策略选择rejected的概率
                    
                    # Cost Model的输出（论文中的C_ψ(y, x)）
                    # harmless_rm输出的是safety score（越高越安全），需要转换为cost（越高越有害）
                    # cost = -safety_score
                    # 注意：cost_w和cost_l在上面已经计算（在计算delta_S之前）
                    
                    # 估计当前策略的期望cost：J_C(θ) = E_{y~π_θ}[C(y,x)] + d
                    # 使用策略偏好作为权重，估计策略会选择哪个回答及其对应的cost
                    expected_cost = (prob_chosen * cost_w + prob_rejected * cost_l).mean()
                    J_C_batch = expected_cost + cfg.cost_threshold  # J_C = E[C] + d
                    
                    # 使用指数移动平均平滑J_C，减少噪声波动（论文Appendix B.3）
                    if J_C_ema is None:
                        J_C_ema = J_C_batch.item()
                    else:
                        # 使用较大的平滑系数（0.95）确保稳定性
                        # 论文中提到使用moving average，但没有明确给出系数，这里使用0.95
                        J_C_ema = 0.95 * J_C_ema + 0.05 * J_C_batch.item()
                    
                    # 在log空间更新lambda（论文公式 31）
                    # ln λ' = ln λ + α · λ · J_C
                    alpha = cfg.λ_lr
                    lambda_log = lambda_log + alpha * lambda_param * J_C_ema
                    
                    # 指数变换回原始空间：λ' = exp(ln λ')
                    lambda_param_new = lambda_log.exp()
                    
                    # 可选：添加上下限以防止lambda过大或过小
                    lambda_param = lambda_param_new.clamp(min=1e-6, max=10.0)
                    # 同步更新log空间的值（考虑clamp的影响）
                    lambda_log = lambda_param.log()

                if (not is_ddp or rank == 0) and global_step % 10 == 0:
                    print(
                        f"Epoch {epoch} step {global_step} "
                        f"loss_H={loss_H.item():.4f} loss_S={loss_S.item():.4f} "
                        f"KL={kl.item():.4f} lambda={lambda_param.item():.4f} "
                        f"J_C={J_C_ema:.4f} delta_S_mean={delta_S.mean().item():.4f}"
                    )

        # 一个 epoch 结束后保存 checkpoint (only on rank 0)
        if (not is_ddp) or rank == 0:
        save_dir = output_dir / f"epoch_{epoch}"
        save_dir.mkdir(parents=True, exist_ok=True)
            # Unwrap DDP model for saving
            model_to_save = policy_model.module if is_ddp else policy_model
            model_to_save.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved epoch {epoch} checkpoint to {save_dir}")

    cleanup_distributed()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train policy with Safe-MM-DPO.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to dpo_config.yaml.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save aligned policy checkpoints.",
    )
    parser.add_argument(
        "--logging_dir",
        type=Path,
        required=False,
        default=Path("logs/training/safe_mm_dpo"),
        help="Directory for logs (currently used only for structure).",
    )
    args = parser.parse_args()

    cfg = load_dpo_config(args.config)
    run_training(cfg, args.output_dir, args.logging_dir)


if __name__ == "__main__":
    main()


