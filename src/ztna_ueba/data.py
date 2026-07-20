"""Dataset and batching utilities for sampled normalized events."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import FieldTokenizer, TokenizedRequest, collate_requests


NORMAL_LABELS = {"0", "false", "normal", "benign", "allow", "allowed", "success"}
ATTACK_LABELS = {"1", "true", "abnormal", "attack", "malicious", "deny", "denied"}


def extract_label(record: dict[str, object]) -> float:
    for field in ("is_attack", "label", "target", "ground_truth"):
        if field not in record or pd.isna(record[field]):
            continue
        value = str(record[field]).strip().lower()
        if value in NORMAL_LABELS:
            return 0.0
        if value in ATTACK_LABELS:
            return 1.0
    raise ValueError("record has no supported binary attack label")


class TokenizedFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: FieldTokenizer):
        self.samples: list[tuple[TokenizedRequest, float, str]] = []
        seen_fields: set[str] = set()
        for record in frame.to_dict(orient="records"):
            label = extract_label(record)
            tokenized = tokenizer.tokenize_request(record)
            seen_fields.update(tokenized.events[0].field_names)
            self.samples.append((tokenized, label, str(record.get("dataset", "unknown"))))
        self.seen_fields = seen_fields

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def make_collate() -> Callable:
    def collate(samples):
        requests, labels, datasets = zip(*samples, strict=True)
        batch = collate_requests(requests)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        batch["datasets"] = list(datasets)
        return batch

    return collate


def move_tensors(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
