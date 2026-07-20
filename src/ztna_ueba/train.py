"""Training entry point for the schema-independent Trust Score model."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
from torch import nn
from torch.utils.data import DataLoader

from .baseline import BASELINE_FEATURE_DIM, BaselineRegistry
from .calibration import NormalScoreCalibrator
from .data import TokenizedFrameDataset, make_collate, move_tensors
from .model import HierarchicalFieldAttention, ModelConfig
from .tokenizer import FieldTokenizer, TokenizerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--field-dropout", type=float, default=0.10)
    parser.add_argument("--attention-entropy-weight", type=float, default=0.03)
    parser.add_argument("--holdout-dataset")
    parser.add_argument("--portable", action="store_true")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def apply_field_dropout(batch: dict[str, object], probability: float) -> None:
    if probability <= 0:
        return
    mask = batch["field_mask"]
    keep = mask & (torch.rand(mask.shape, device=mask.device) >= probability)
    batch_size, event_count, _ = keep.shape
    for batch_index in range(batch_size):
        for event_index in range(event_count):
            if not batch["event_mask"][batch_index, event_index]:
                continue
            if not keep[batch_index, event_index].any():
                first = torch.nonzero(mask[batch_index, event_index], as_tuple=False)[0, 0]
                keep[batch_index, event_index, first] = True
    batch["field_mask"] = keep


def normalized_attention_entropy(
    output: dict[str, torch.Tensor], field_mask: torch.Tensor
) -> torch.Tensor:
    joint = output["event_weights"].unsqueeze(-1) * output["field_weights"]
    joint = joint * field_mask.to(joint.dtype)
    entropy = -torch.sum(
        torch.where(
            joint > 0,
            joint * torch.log(joint.clamp_min(1e-12)),
            torch.zeros_like(joint),
        ),
        dim=(1, 2),
    )
    counts = field_mask.sum(dim=(1, 2)).to(joint.dtype)
    normalizer = torch.log(counts.clamp_min(2.0))
    return (entropy / normalizer).mean()


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    normal_count = true_negative + false_positive
    attack_count = true_positive + false_negative
    result = {
        "rows": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "threshold": float(threshold),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "false_positive_rate": false_positive / max(1, normal_count),
        "specificity": true_negative / max(1, normal_count),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["average_precision"] = float(average_precision_score(labels, probabilities))
    return result


def best_f1_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    scores = [f1_score(labels, probabilities >= value, zero_division=0) for value in candidates]
    return float(candidates[int(np.argmax(scores))])


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    loss_function,
    calibrator: NormalScoreCalibrator | None = None,
) -> tuple[float, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    losses = []
    labels = []
    probabilities = []
    datasets = []
    for batch in loader:
        batch = move_tensors(batch, device)
        output = model(batch)
        loss = loss_function(output["risk_logit"], batch["labels"])
        losses.append(float(loss.detach().cpu()))
        labels.extend(batch["labels"].detach().cpu().numpy().tolist())
        current_probabilities = output["risk_probability"]
        if calibrator is not None:
            calibrated, ready = calibrator.normalize_batch(
                batch["profile_keys"],
                output["event_anomaly_score"],
                output["event_weights"],
                batch["event_mask"],
            )
            current_probabilities = torch.where(ready, calibrated, current_probabilities)
        probabilities.extend(current_probabilities.detach().cpu().numpy().tolist())
        datasets.extend(batch["datasets"])
    return float(np.mean(losses)), np.asarray(labels), np.asarray(probabilities), datasets


@torch.no_grad()
def collect_normal_event_scores(model, loader, device) -> tuple[list[str], list[float]]:
    model.eval()
    profile_keys: list[str] = []
    scores: list[float] = []
    for batch in loader:
        batch = move_tensors(batch, device)
        output = model(batch)
        for batch_index, request_profiles in enumerate(batch["profile_keys"]):
            for event_index, profile_key in enumerate(request_profiles):
                profile_keys.append(profile_key)
                scores.append(
                    float(output["event_anomaly_score"][batch_index, event_index].cpu())
                )
    return profile_keys, scores


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    frame = pd.read_csv(args.data, compression="infer", dtype="string", low_memory=False)
    required_splits = {"train", "validation", "test"}
    if "split" not in frame or not required_splits.issubset(set(frame["split"].dropna())):
        raise ValueError("sample must contain train, validation, and test splits")
    if args.holdout_dataset:
        held_out = frame["dataset"] == args.holdout_dataset
        if not held_out.any():
            raise ValueError(f"holdout dataset not found: {args.holdout_dataset}")
        split_frames = {
            "train": frame[(~held_out) & (frame["split"] == "train")],
            "validation": frame[(~held_out) & (frame["split"] == "validation")],
            "test": frame[held_out & (frame["split"] == "test")],
        }
    else:
        split_frames = {
            split: frame[frame["split"] == split] for split in required_splits
        }
    for split, split_frame in split_frames.items():
        labels = {
            str(value).strip().lower()
            for value in split_frame["label"].dropna()
        }
        if not {"normal", "abnormal"}.issubset(labels):
            raise ValueError(f"split {split} does not contain both binary classes")

    tokenizer_config = TokenizerConfig(portable_mode=args.portable)
    baseline_registry = None
    baseline_frame = frame[
        (frame["split"] == "train")
        & (frame["label"].str.strip().str.lower() == "normal")
    ]
    if args.portable:
        baseline_registry = BaselineRegistry.fit(
            baseline_frame.to_dict(orient="records"),
            excluded_fields=tokenizer_config.excluded_fields,
            profile_fields=tokenizer_config.profile_fields,
        )
    tokenizer = FieldTokenizer(tokenizer_config, baseline_registry=baseline_registry)
    datasets = {
        split: TokenizedFrameDataset(split_frame, tokenizer)
        for split, split_frame in split_frames.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=split == "train",
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=make_collate(),
        )
        for split, dataset in datasets.items()
    }
    calibration_loader = None
    if args.portable:
        calibration_dataset = TokenizedFrameDataset(baseline_frame, tokenizer)
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=make_collate(),
        )
    model_config = ModelConfig(
        numeric_feature_dim=8 + BASELINE_FEATURE_DIM if args.portable else 8,
        monotonic_baseline=args.portable,
    )
    model = HierarchicalFieldAttention(model_config).to(device)
    train_labels = np.asarray([sample[1] for sample in datasets["train"].samples])
    positive = max(1.0, float(train_labels.sum()))
    negative = max(1.0, float(len(train_labels) - positive))
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / positive, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_state = None
    best_score = -float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_bce_losses = []
        train_entropies = []
        for batch in loaders["train"]:
            batch = move_tensors(batch, device)
            apply_field_dropout(batch, args.field_dropout)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(batch)
                bce_loss = loss_function(output["risk_logit"], batch["labels"])
                attention_entropy = normalized_attention_entropy(output, batch["field_mask"])
                loss = bce_loss - args.attention_entropy_weight * attention_entropy
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            train_bce_losses.append(float(bce_loss.detach().cpu()))
            train_entropies.append(float(attention_entropy.detach().cpu()))

        val_loss, val_y, val_p, _ = evaluate(
            model, loaders["validation"], device, loss_function
        )
        threshold = best_f1_threshold(val_y, val_p)
        metrics = binary_metrics(val_y, val_p, threshold)
        score = metrics.get("average_precision", metrics["f1"])
        epoch_result = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_bce_loss": float(np.mean(train_bce_losses)),
            "train_normalized_attention_entropy": float(np.mean(train_entropies)),
            "validation_loss": val_loss,
            "validation": metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    score_calibrator = None
    if calibration_loader is not None:
        calibration_profiles, calibration_scores = collect_normal_event_scores(
            model, calibration_loader, device
        )
        score_calibrator = NormalScoreCalibrator.fit(
            calibration_profiles, calibration_scores
        )
    val_loss, val_y, val_p_raw, _ = evaluate(
        model, loaders["validation"], device, loss_function
    )
    _, _, val_p, _ = evaluate(
        model,
        loaders["validation"],
        device,
        loss_function,
        calibrator=score_calibrator,
    )
    threshold = 0.5 if score_calibrator is not None else best_f1_threshold(val_y, val_p)
    test_loss, test_y, test_p_raw, test_origins = evaluate(
        model, loaders["test"], device, loss_function
    )
    _, _, test_p, _ = evaluate(
        model,
        loaders["test"],
        device,
        loss_function,
        calibrator=score_calibrator,
    )
    test_metrics = binary_metrics(test_y, test_p, threshold)
    per_dataset = {}
    origins = np.asarray(test_origins)
    for origin in sorted(set(test_origins)):
        selected = origins == origin
        per_dataset[origin] = binary_metrics(test_y[selected], test_p[selected], threshold)

    seen_fields = sorted(datasets["train"].seen_fields)
    training_frame = split_frames["train"]
    seen_datasets = sorted(training_frame["dataset"].dropna().astype(str).unique().tolist())
    seen_source_types = sorted(
        training_frame["source_type"].dropna().astype(str).unique().tolist()
    )
    checkpoint = {
        "state_dict": best_state,
        "model_config": asdict(model_config),
        "tokenizer_config": {
            **asdict(tokenizer_config),
            "excluded_fields": sorted(tokenizer_config.excluded_fields),
        },
        "threshold": threshold,
        "seen_fields": seen_fields,
        "seen_datasets": seen_datasets,
        "seen_source_types": seen_source_types,
        "baseline_registry": (
            baseline_registry.to_dict() if baseline_registry is not None else None
        ),
        "score_calibrator": (
            score_calibrator.to_dict() if score_calibrator is not None else None
        ),
        "training": {
            "seed": args.seed,
            "device": str(device),
            "torch_version": torch.__version__,
            "data": str(Path(args.data).resolve()),
            "holdout_dataset": args.holdout_dataset,
            "attention_entropy_weight": args.attention_entropy_weight,
            "portable": args.portable,
            "baseline_calibration_rows": int(len(baseline_frame)),
            "score_calibration_profiles": (
                len(score_calibrator.profiles) if score_calibrator is not None else 0
            ),
        },
    }
    torch.save(checkpoint, output_dir / "model.pt")
    result = {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "validation_loss": val_loss,
        "validation": binary_metrics(val_y, val_p, threshold),
        "validation_raw": binary_metrics(val_y, val_p_raw, threshold),
        "test_loss": test_loss,
        "test": test_metrics,
        "test_raw": binary_metrics(test_y, test_p_raw, threshold),
        "test_by_dataset": per_dataset,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("TRAINING_DONE " + json.dumps(result["test"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
