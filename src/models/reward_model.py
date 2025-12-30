"""
Reward model definition.

According to dev.md, we need a scalar-valued head on top of a base LM
to score (prompt, response) pairs. For simplicity, we reuse
`AutoModelForSequenceClassification` with `num_labels=1`, which works
with most decoder-only LMs (including Mistral-7B) via the
`AutoModelForSequenceClassification` API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    TaskType = None


@dataclass
class RewardModelConfig:
    base_model_path: str
    tokenizer_name: Optional[str] = None
    max_length: int = 1024
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[list[str]] = None


class RewardModel(nn.Module):
    """
    Thin wrapper around AutoModelForSequenceClassification (num_labels=1).
    """

    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns a scalar score for each sequence in the batch.
        Shape: (batch,)
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)  # (batch,)
        return logits


def load_reward_model(
    cfg: RewardModelConfig,
    dtype: Optional[torch.dtype] = None,
    use_gradient_checkpointing: bool = True,
) -> tuple[RewardModel, PreTrainedTokenizerBase]:
    """
    Load base model and tokenizer and wrap as a RewardModel.

    - base_model_path: path or HF id for the base LM
    - tokenizer_name: optional, if None we reuse base_model_path
    - dtype: optional dtype for model weights (e.g., torch.bfloat16)
    - use_gradient_checkpointing: enable gradient checkpointing to save memory
    - use_lora: if True, use LoRA for parameter-efficient fine-tuning
    """
    tokenizer_name = cfg.tokenizer_name or cfg.base_model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        # For decoder-only LMs (e.g. Mistral) we usually don't have a pad token
        # defined by default. Reuse EOS as PAD so that:
        #   - our DataLoader can pad sequences (see train_rm.collate_fn)
        #   - HuggingFace internals (SequenceSummary etc.) know pad_token_id
        tokenizer.pad_token = tokenizer.eos_token

    # Auto-select dtype for memory efficiency if not specified
    if dtype is None and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    # Make sure the model itself also has pad_token_id set, otherwise some
    # transformer utilities will raise when batch_size > 1.
    load_kwargs = {
        "num_labels": 1,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if dtype is not None:
        load_kwargs["dtype"] = dtype

    base_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model_path,
        **load_kwargs,
    )
    # Double‑check on the loaded config as well.
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Enable gradient checkpointing for memory efficiency
    if use_gradient_checkpointing:
        base_model.config.use_cache = False
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()

    # Apply LoRA if requested
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
            task_type=TaskType.SEQ_CLS,  # Sequence classification task
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",  # Don't train bias parameters
        )
        base_model = get_peft_model(base_model, lora_config)
        
        # Print trainable parameters info
        trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in base_model.parameters())
        print(f"LoRA enabled: {trainable_params:,} trainable / {total_params:,} total parameters "
              f"({100 * trainable_params / total_params:.2f}%)")

    model = RewardModel(base_model)
    return model, tokenizer


