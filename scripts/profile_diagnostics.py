"""Compare baseline features and learned field contributions across dataset slices."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ztna_ueba.baseline import BaselineRegistry
from ztna_ueba.data import TokenizedFrameDataset, make_collate, move_tensors
from ztna_ueba.model import HierarchicalFieldAttention, ModelConfig
from ztna_ueba.tokenizer import FieldTokenizer, TokenizerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_runtime(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer_values = dict(checkpoint["tokenizer_config"])
    tokenizer_values["excluded_fields"] = frozenset(
        tokenizer_values["excluded_fields"]
    )
    tokenizer_values["profile_fields"] = tuple(tokenizer_values["profile_fields"])
    registry = BaselineRegistry.from_dict(checkpoint["baseline_registry"])
    tokenizer = FieldTokenizer(TokenizerConfig(**tokenizer_values), registry)
    model = HierarchicalFieldAttention(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return tokenizer, model.to(device).eval()


@torch.no_grad()
def summarize_slice(
    frame: pd.DataFrame,
    tokenizer: FieldTokenizer,
    model: HierarchicalFieldAttention,
    device: torch.device,
    batch_size: int,
) -> dict:
    loader = DataLoader(
        TokenizedFrameDataset(frame, tokenizer),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(),
    )
    field_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    risk_scores = []
    for batch in loader:
        batch = move_tensors(batch, device)
        output = model(batch)
        risk_scores.extend(output["event_anomaly_score"][:, 0].cpu().tolist())
        for batch_index, events in enumerate(batch["field_names"]):
            for event_index, field_names in enumerate(events):
                for field_index, field_name in enumerate(field_names):
                    features = batch["numeric_features"][
                        batch_index, event_index, field_index
                    ]
                    total = field_totals[field_name]
                    total["count"] += 1
                    total["presence_surprise"] += float(features[10].cpu())
                    total["categorical_rarity"] += float(features[11].cpu())
                    total["numeric_deviation"] += float(features[12].cpu())
                    total["type_mismatch"] += float(features[13].cpu())
                    total["attention"] += float(
                        output["field_weights"][batch_index, event_index, field_index].cpu()
                    )
                    total["contribution"] += float(
                        output["field_contributions"][
                            batch_index, event_index, field_index
                        ].cpu()
                    )
    fields = {}
    for field_name, totals in field_totals.items():
        count = totals.pop("count")
        fields[field_name] = {
            key: value / count for key, value in sorted(totals.items())
        }
        fields[field_name]["count"] = int(count)
    series = pd.Series(risk_scores, dtype="float64")
    return {
        "rows": len(frame),
        "risk_score": {
            "mean": float(series.mean()),
            "q50": float(series.quantile(0.50)),
            "q95": float(series.quantile(0.95)),
            "q99": float(series.quantile(0.99)),
        },
        "fields": dict(sorted(fields.items())),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    tokenizer, model = load_runtime(args.checkpoint, device)
    frame = pd.read_csv(args.data, compression="infer", dtype="string", low_memory=False)
    frame = frame[frame["dataset"] == args.dataset]
    report = {"dataset": args.dataset, "slices": {}}
    for split in ("train", "validation", "test"):
        for label in ("normal", "abnormal"):
            selected = frame[
                (frame["split"] == split)
                & (frame["label"].str.strip().str.lower() == label)
            ]
            if selected.empty:
                continue
            report["slices"][f"{split}:{label}"] = summarize_slice(
                selected, tokenizer, model, device, args.batch_size
            )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
