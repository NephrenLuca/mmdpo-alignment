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

    # Try to load the model from the specified path
    # If it fails (e.g., missing model_type in config), try loading from base model
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            cfg.base_model_path,
            **load_kwargs,
        )
    except (ValueError, OSError) as e:
        # If loading fails, it might be because:
        # 1. The saved model is missing model_type in config.json
        # 2. The path points to a LoRA adapter without base model
        # Try to load from base model path if different
        import os
        from pathlib import Path
        
        base_path = Path(cfg.base_model_path)
        # Check if this is a saved reward model (has config.json but might be missing model_type)
        config_path = base_path / "config.json"
        if config_path.exists():
            # Try to fix the config by loading and re-saving with model_type
            import json
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                # If model_type is missing or incorrect, try to infer from architecture
                needs_fix = False
                if "model_type" not in config:
                    needs_fix = True
                elif config.get("model_type") == "align":  # Common error
                    needs_fix = True
                
                if needs_fix and "architectures" in config:
                    arch = config["architectures"][0] if config["architectures"] else None
                    # Map common architectures to model_type
                    if arch and "Mistral" in arch:
                        config["model_type"] = "mistral"
                    elif arch and "Llama" in arch:
                        config["model_type"] = "llama"
                    elif arch and "GPT" in arch:
                        config["model_type"] = "gpt2"
                    else:
                        # Default to mistral for Mistral-based models
                        config["model_type"] = "mistral"
                    
                    # Backup original config
                    backup_path = config_path.with_suffix('.json.bak')
                    with open(backup_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    
                    # Save the fixed config
                    with open(config_path, 'w') as f:
                        json.dump(config, f, indent=2)
                    print(f"✓ Auto-fixed model_type in {cfg.base_model_path}: {config['model_type']}")
                    
                    # Retry loading
                    base_model = AutoModelForSequenceClassification.from_pretrained(
                        cfg.base_model_path,
                        **load_kwargs,
                    )
                else:
                    raise e
            except Exception:
                # If fixing config fails, re-raise original error
                raise ValueError(
                    f"Failed to load reward model from {cfg.base_model_path}. "
                    f"Original error: {e}. "
                    f"Please ensure the model was saved correctly with model_type in config.json."
                ) from e
        else:
            raise ValueError(
                f"Model path {cfg.base_model_path} does not exist or is not a valid model directory. "
                f"Original error: {e}"
            ) from e
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


