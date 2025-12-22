"""
Simple preprocessing utilities for BeaverTails-style preference data.

This file is intentionally lightweight. The main entrypoint for users
is the CLI script in src/scripts/prepare_data.py, which uses the helpers
defined here to:
- load raw json/jsonl files;
- construct (prompt, chosen_response, rejected_response) pairs;
- split into train/val/test;
- write to jsonl files consumed by training scripts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple


PreferenceRecord = Dict[str, Any]


@dataclass
class SplitConfig:
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def validate(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total} "
                f"({self.train_ratio}, {self.val_ratio}, {self.test_ratio})"
            )


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load a jsonl file into memory."""
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def iter_raw_files(input_dir: Path, pattern: str = "*.jsonl") -> Iterable[Path]:
    """Yield all raw data files under input_dir matching the given pattern."""
    return sorted(input_dir.glob(pattern))


def build_preference_pairs(
    records: Iterable[Dict[str, Any]],
    dimension: str,
) -> List[PreferenceRecord]:
    """
    Convert raw BeaverTails / PKU-SafeRLHF-style records into preference pairs.

    We support两种常见格式：

    1）已预先构造好的偏好对（通用 Beavertails 形式）：
        - 'prompt'
        - 'response_chosen'
        - 'response_rejected'

    2）PKU-SafeRLHF 形式（如 Alpaca-7B / Alpaca2-7B / Alpaca3-8B）：
        - 'prompt'
        - 'response_0', 'response_1'
        - 'better_response_id'   （helpful 维度）
        - 'safer_response_id'    （harmless 维度）

    对于（2），我们根据 `dimension` 选择使用哪一个 id 字段：
        - dimension == "helpful"  -> better_response_id
        - dimension == "harmless" -> safer_response_id
    """
    out: List[PreferenceRecord] = []
    for rec in records:
        prompt = rec.get("prompt")
        if not prompt:
            continue

        # 情况 1：通用 Beavertails 格式（已经给出 chosen/rejected）
        if "response_chosen" in rec and "response_rejected" in rec:
            chosen = rec.get("response_chosen")
            rejected = rec.get("response_rejected")
            if chosen is None or rejected is None:
                continue

        # 情况 2：PKU-SafeRLHF 格式，利用 *_response_id 进行构造
        elif "response_0" in rec and "response_1" in rec:
            if dimension == "helpful":
                id_key = "better_response_id"
            elif dimension == "harmless":
                id_key = "safer_response_id"
            else:
                # 未知维度，直接跳过
                continue

            idx = rec.get(id_key)
            if idx not in (0, 1):
                # 缺少偏好标签或格式不对，跳过
                continue

            chosen = rec.get(f"response_{idx}")
            rejected = rec.get(f"response_{1 - idx}")
            if chosen is None or rejected is None:
                continue

            # 对于harmless维度，提取safety labels（用于Cost Model的classification loss）
            # 根据论文，s(y) = +1 表示有害，s(y) = -1 表示无害
            # PKU-SafeRLHF中：is_response_X_safe = True 表示无害，False 表示有害
            # 因此：s(y) = -1 if is_safe else +1
            safety_labels = None
            if dimension == "harmless":
                is_safe_chosen = rec.get(f"is_response_{idx}_safe")
                is_safe_rejected = rec.get(f"is_response_{1 - idx}_safe")
                
                # 如果存在safety labels，转换为论文格式：+1表示有害，-1表示无害
                if is_safe_chosen is not None and is_safe_rejected is not None:
                    # is_safe=True -> s=-1 (无害), is_safe=False -> s=+1 (有害)
                    safety_labels = {
                        "chosen": -1 if is_safe_chosen else +1,
                        "rejected": -1 if is_safe_rejected else +1,
                    }

        # 其它未知格式：暂时跳过
        else:
            continue

        pair_record = {
            "prompt": prompt,
            "chosen_response": chosen,
            "rejected_response": rejected,
            "dimension": dimension,
        }
        
        # 如果是harmless维度且存在safety labels，添加到记录中
        if dimension == "harmless" and safety_labels is not None:
            pair_record["safety_labels"] = safety_labels

        out.append(pair_record)
    return out


def split_dataset(
    data: List[PreferenceRecord],
    split_cfg: SplitConfig,
    seed: int = 42,
) -> Tuple[List[PreferenceRecord], List[PreferenceRecord], List[PreferenceRecord]]:
    """Shuffle and split data into train/val/test."""
    split_cfg.validate()
    rng = random.Random(seed)
    rng.shuffle(data)

    n = len(data)
    n_train = int(n * split_cfg.train_ratio)
    n_val = int(n * split_cfg.val_ratio)

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]
    return train, val, test


def write_jsonl(path: Path, records: Iterable[PreferenceRecord]) -> None:
    """Write records to jsonl file, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")



