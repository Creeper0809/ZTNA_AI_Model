"""Audit attention concentration and field contributions on an evaluation split."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ztna_ueba.baseline import BaselineRegistry
from ztna_ueba.calibration import NormalScoreCalibrator
from ztna_ueba.data import TokenizedFrameDataset, make_collate, move_tensors
from ztna_ueba.model import HierarchicalFieldAttention, ModelConfig
from ztna_ueba.tokenizer import FieldTokenizer, TokenizerConfig
from ztna_ueba.train import binary_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer_values = dict(checkpoint["tokenizer_config"])
    tokenizer_values["excluded_fields"] = frozenset(tokenizer_values["excluded_fields"])
    if "profile_fields" in tokenizer_values:
        tokenizer_values["profile_fields"] = tuple(tokenizer_values["profile_fields"])
    baseline_values = checkpoint.get("baseline_registry")
    baseline_registry = (
        BaselineRegistry.from_dict(baseline_values) if baseline_values else None
    )
    calibrator_values = checkpoint.get("score_calibrator")
    score_calibrator = (
        NormalScoreCalibrator.from_dict(calibrator_values)
        if calibrator_values
        else None
    )
    tokenizer = FieldTokenizer(
        TokenizerConfig(**tokenizer_values), baseline_registry=baseline_registry
    )
    model = HierarchicalFieldAttention(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    frame = pd.read_csv(args.data, compression="infer", dtype="string", low_memory=False)
    holdout = checkpoint.get("training", {}).get("holdout_dataset")
    selected = (
        frame[(frame["dataset"] == holdout) & (frame["split"] == "test")]
        if holdout
        else frame[frame["split"] == "test"]
    )
    dataset = TokenizedFrameDataset(selected, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=make_collate(),
    )
    field_stats = defaultdict(lambda: {"occurrences": 0, "weight": 0.0, "abs": 0.0, "signed": 0.0})
    concentration = []
    trust_by_label = defaultdict(list)
    weight_sum_errors = []
    all_labels = []
    all_probabilities = []
    all_raw_probabilities = []
    all_ready = []
    all_datasets = []
    profile_readiness = defaultdict(lambda: {"rows": 0, "ready": 0})
    with torch.no_grad():
        for batch in loader:
            batch = move_tensors(batch, device)
            output = model(batch)
            calibrated = output["risk_probability"]
            ready = torch.ones_like(calibrated, dtype=torch.bool)
            if score_calibrator is not None:
                calibrated, ready = score_calibrator.normalize_batch(
                    batch["profile_keys"],
                    output["event_anomaly_score"],
                    output["event_weights"],
                    batch["event_mask"],
                )
            effective_probability = torch.where(
                ready, calibrated, output["risk_probability"]
            )
            joint = output["event_weights"].unsqueeze(-1) * output["field_weights"]
            contributions = output["field_contributions"]
            valid = batch["field_mask"]
            sums = (joint * valid).sum(dim=(1, 2))
            weight_sum_errors.extend(torch.abs(sums - 1.0).cpu().tolist())
            masked_joint = joint.masked_fill(~valid, 0.0)
            concentration.extend(masked_joint.amax(dim=(1, 2)).cpu().tolist())
            labels = batch["labels"].cpu().tolist()
            trust = (100.0 * (1.0 - effective_probability)).cpu().tolist()
            all_labels.extend(labels)
            all_probabilities.extend(effective_probability.cpu().tolist())
            all_raw_probabilities.extend(output["risk_probability"].cpu().tolist())
            all_ready.extend(ready.cpu().tolist())
            all_datasets.extend(batch["datasets"])
            for label, score in zip(labels, trust, strict=True):
                trust_by_label[str(int(label))].append(score)
            for request_profiles, request_ready in zip(
                batch["profile_keys"], ready.cpu().tolist(), strict=True
            ):
                for profile_key in request_profiles:
                    profile_readiness[profile_key]["rows"] += 1
                    profile_readiness[profile_key]["ready"] += int(request_ready)
            for row_index, request_names in enumerate(batch["field_names"]):
                for event_index, names in enumerate(request_names):
                    for field_index, name in enumerate(names):
                        weight = float(joint[row_index, event_index, field_index].cpu())
                        contribution = float(contributions[row_index, event_index, field_index].cpu())
                        stats = field_stats[name]
                        stats["occurrences"] += 1
                        stats["weight"] += weight
                        stats["abs"] += abs(contribution)
                        stats["signed"] += contribution
    fields = []
    for name, stats in field_stats.items():
        count = stats["occurrences"]
        fields.append(
            {
                "field": name,
                "occurrences": count,
                "mean_weight_when_present": stats["weight"] / count,
                "mean_abs_logit_contribution": stats["abs"] / count,
                "mean_signed_logit_contribution": stats["signed"] / count,
            }
        )
    fields.sort(key=lambda row: row["mean_abs_logit_contribution"], reverse=True)
    labels_array = np.asarray(all_labels, dtype=np.int64)
    probabilities_array = np.asarray(all_probabilities, dtype=np.float64)
    raw_probabilities_array = np.asarray(all_raw_probabilities, dtype=np.float64)
    ready_array = np.asarray(all_ready, dtype=bool)
    dataset_array = np.asarray(all_datasets)
    threshold = float(checkpoint["threshold"])
    per_dataset = {
        dataset_name: binary_metrics(
            labels_array[dataset_array == dataset_name],
            probabilities_array[dataset_array == dataset_name],
            threshold,
        )
        for dataset_name in sorted(set(all_datasets))
    }
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "rows": len(dataset),
        "holdout_dataset": holdout,
        "max_weight_sum_error": max(weight_sum_errors, default=0.0),
        "top_weight_concentration": {
            "p50": float(np.percentile(concentration, 50)),
            "p90": float(np.percentile(concentration, 90)),
            "p99": float(np.percentile(concentration, 99)),
        },
        "calibrated_metrics": binary_metrics(
            labels_array, probabilities_array, threshold
        ),
        "raw_metrics": binary_metrics(
            labels_array, raw_probabilities_array, threshold
        ),
        "ready_only_metrics": (
            binary_metrics(
                labels_array[ready_array],
                probabilities_array[ready_array],
                threshold,
            )
            if ready_array.any()
            else None
        ),
        "calibration_ready_rows": int(ready_array.sum()),
        "shadow_rows": int((~ready_array).sum()),
        "metrics_by_dataset": per_dataset,
        "profile_readiness": {
            profile_key: {
                **counts,
                "ready_rate": counts["ready"] / max(1, counts["rows"]),
            }
            for profile_key, counts in sorted(profile_readiness.items())
        },
        "trust_score_by_label": {
            label: {
                "count": len(values),
                "mean": float(np.mean(values)),
                "p10": float(np.percentile(values, 10)),
                "p90": float(np.percentile(values, 90)),
            }
            for label, values in trust_by_label.items()
        },
        "top_fields": fields[:20],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
