"""Head-only fine-tuning for a portable ZTNA-UEBA checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .baseline import BaselineRegistry
from .calibration import NormalScoreCalibrator
from .data import TokenizedFrameDataset, extract_label, make_collate, move_tensors
from .model import HierarchicalFieldAttention, ModelConfig
from .tokenizer import FieldTokenizer, TokenizerConfig
from .train import (
    apply_field_dropout,
    binary_metrics,
    choose_device,
    collect_normal_event_scores,
    evaluate,
    normalized_attention_entropy,
    seed_everything,
)


REQUIRED_SPLITS = ("train", "validation", "test")
TRAINABLE_PREFIXES = (
    "field_weight_head.",
    "field_evidence_head.",
    "event_weight_head.",
)
TRAINABLE_EXACT = {"global_bias"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune only the field/event weighting and evidence heads using "
            "reviewed normal/abnormal logs."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replay-data")
    parser.add_argument("--dataset")
    parser.add_argument("--source-type")
    parser.add_argument("--event-type")
    parser.add_argument("--confirmed-labeled", action="store_true")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--field-dropout", type=float, default=0.10)
    parser.add_argument("--attention-entropy-weight", type=float, default=0.03)
    parser.add_argument("--minimum-validation-ap-gain", type=float, default=0.0)
    parser.add_argument("--max-validation-fpr-increase", type=float, default=0.02)
    parser.add_argument("--max-replay-f1-drop", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_frame(path: Path) -> pd.DataFrame:
    lower_name = path.name.lower()
    if lower_name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, compression="infer", dtype="string", low_memory=False)
    if lower_name.endswith(".jsonl"):
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame.from_records(records)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "events" in payload:
        payload = payload["events"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise ValueError("fine-tuning data must be CSV, JSONL, a JSON object, or a JSON list")
    return pd.DataFrame.from_records(payload)


def prepare_labeled_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    dataset: str | None = None,
    source_type: str | None = None,
    event_type: str | None = None,
    minimum_rows: int = 20,
) -> pd.DataFrame:
    if len(frame) < minimum_rows:
        raise ValueError(f"{name} must contain at least {minimum_rows} labeled records")
    prepared = frame.copy()
    overrides = {
        "dataset": dataset,
        "source_type": source_type,
        "event_type": event_type,
    }
    for field, value in overrides.items():
        if value:
            prepared[field] = value
    missing_metadata = [
        field
        for field in overrides
        if field not in prepared
        or prepared[field].fillna("").astype(str).str.strip().eq("").any()
    ]
    if missing_metadata:
        raise ValueError(
            f"{name} requires profile metadata: " + ", ".join(missing_metadata)
        )
    if "split" not in prepared:
        raise ValueError(f"{name} must contain a split column")
    prepared["split"] = prepared["split"].fillna("").astype(str).str.strip().str.lower()
    try:
        labels = [
            "abnormal" if extract_label(record) == 1.0 else "normal"
            for record in prepared.to_dict(orient="records")
        ]
    except ValueError as error:
        raise ValueError(f"{name} contains an unsupported or missing label") from error
    prepared["label"] = labels
    available_splits = set(prepared["split"])
    missing_splits = set(REQUIRED_SPLITS) - available_splits
    if missing_splits:
        raise ValueError(f"{name} is missing splits: {sorted(missing_splits)}")
    for split in REQUIRED_SPLITS:
        split_labels = set(prepared.loc[prepared["split"] == split, "label"])
        if split_labels != {"normal", "abnormal"}:
            raise ValueError(
                f"{name} split {split} must contain normal and abnormal labels"
            )
    return prepared


def split_frame(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split: frame[frame["split"] == split].reset_index(drop=True)
        for split in REQUIRED_SPLITS
    }


def tokenizer_config_from_checkpoint(checkpoint: dict) -> TokenizerConfig:
    tokenizer_values = dict(checkpoint["tokenizer_config"])
    tokenizer_values["excluded_fields"] = frozenset(tokenizer_values["excluded_fields"])
    tokenizer_values["profile_fields"] = tuple(tokenizer_values["profile_fields"])
    return TokenizerConfig(**tokenizer_values)


def merged_registry_for_new_logs(
    checkpoint: dict,
    tokenizer_config: TokenizerConfig,
    new_train_frame: pd.DataFrame,
) -> BaselineRegistry:
    existing_values = checkpoint.get("baseline_registry")
    if not existing_values:
        raise ValueError("fine-tuning requires a checkpoint with a normal baseline")
    existing = BaselineRegistry.from_dict(existing_values)
    normal_records = new_train_frame[
        new_train_frame["label"] == "normal"
    ].to_dict(orient="records")
    incoming = BaselineRegistry.fit(
        normal_records,
        excluded_fields=tokenizer_config.excluded_fields,
        profile_fields=tokenizer_config.profile_fields,
    )
    return existing.merged(incoming)


def make_loader(
    frame: pd.DataFrame,
    tokenizer: FieldTokenizer,
    *,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
) -> tuple[TokenizedFrameDataset, DataLoader]:
    dataset = TokenizedFrameDataset(frame, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=make_collate(),
    )
    return dataset, loader


def is_fine_tunable_parameter(name: str) -> bool:
    return name in TRAINABLE_EXACT or name.startswith(TRAINABLE_PREFIXES)


def configure_head_only_fine_tuning(
    model: HierarchicalFieldAttention,
) -> list[str]:
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = is_fine_tunable_parameter(name)
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable:
        raise RuntimeError("no fine-tuning parameters were selected")
    return trainable


def fine_tune_batch(
    model: HierarchicalFieldAttention,
    batch: dict[str, object],
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    *,
    device: torch.device,
    field_dropout: float,
    attention_entropy_weight: float,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, float, float]:
    batch = move_tensors(batch, device)
    apply_field_dropout(batch, field_dropout)
    optimizer.zero_grad(set_to_none=True)
    active_scaler = scaler or torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda"
    )
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        output = model(batch)
        bce_loss = loss_function(output["risk_logit"], batch["labels"])
        attention_entropy = normalized_attention_entropy(
            output, batch["field_mask"]
        )
        loss = bce_loss - attention_entropy_weight * attention_entropy
    active_scaler.scale(loss).backward()
    active_scaler.unscale_(optimizer)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
    active_scaler.step(optimizer)
    active_scaler.update()
    return (
        float(loss.detach().cpu()),
        float(bce_loss.detach().cpu()),
        float(attention_entropy.detach().cpu()),
    )


def metrics_for_loader(
    model: HierarchicalFieldAttention,
    loader: DataLoader,
    device: torch.device,
    loss_function: nn.Module,
) -> dict:
    _, labels, probabilities, _ = evaluate(
        model, loader, device, loss_function
    )
    return binary_metrics(labels, probabilities, threshold=0.5)


def copy_state(model: HierarchicalFieldAttention) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def changed_parameter_names(
    before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]
) -> list[str]:
    return [
        name for name in before if not torch.equal(before[name], after[name])
    ]


def main() -> None:
    args = parse_args()
    if not args.confirmed_labeled:
        raise ValueError(
            "--confirmed-labeled is required; fine-tuning changes model parameters"
        )
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("output must differ from the input checkpoint")

    seed_everything(args.seed)
    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tokenizer_config = tokenizer_config_from_checkpoint(checkpoint)
    if not tokenizer_config.portable_mode:
        raise ValueError("head-only fine-tuning requires a portable checkpoint")

    new_frame = prepare_labeled_frame(
        load_frame(Path(args.data)),
        name="fine-tuning data",
        dataset=args.dataset,
        source_type=args.source_type,
        event_type=args.event_type,
    )
    replay_frame = None
    if args.replay_data:
        replay_frame = prepare_labeled_frame(
            load_frame(Path(args.replay_data)),
            name="replay data",
        )
    new_splits = split_frame(new_frame)
    replay_splits = split_frame(replay_frame) if replay_frame is not None else None
    combined_splits = {
        split: pd.concat(
            [
                new_splits[split],
                *(
                    [replay_splits[split]]
                    if replay_splits is not None
                    else []
                ),
            ],
            ignore_index=True,
        )
        for split in REQUIRED_SPLITS
    }

    baseline_registry = merged_registry_for_new_logs(
        checkpoint, tokenizer_config, new_splits["train"]
    )
    tokenizer = FieldTokenizer(
        tokenizer_config, baseline_registry=baseline_registry
    )
    combined_datasets = {}
    combined_loaders = {}
    new_loaders = {}
    replay_loaders = {}
    for split in REQUIRED_SPLITS:
        dataset, loader = make_loader(
            combined_splits[split],
            tokenizer,
            batch_size=args.batch_size,
            device=device,
            shuffle=split == "train",
        )
        combined_datasets[split] = dataset
        combined_loaders[split] = loader
        _, new_loaders[split] = make_loader(
            new_splits[split],
            tokenizer,
            batch_size=args.batch_size,
            device=device,
            shuffle=False,
        )
        if replay_splits is not None:
            _, replay_loaders[split] = make_loader(
                replay_splits[split],
                tokenizer,
                batch_size=args.batch_size,
                device=device,
                shuffle=False,
            )

    model = HierarchicalFieldAttention(
        ModelConfig(**checkpoint["model_config"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    base_state = copy_state(model)
    train_labels = np.asarray(
        [sample[1] for sample in combined_datasets["train"].samples]
    )
    positive = max(1.0, float(train_labels.sum()))
    negative = max(1.0, float(len(train_labels) - positive))
    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / positive, device=device)
    )
    base_metrics = {
        "new_validation": metrics_for_loader(
            model, new_loaders["validation"], device, loss_function
        ),
        "new_test": metrics_for_loader(
            model, new_loaders["test"], device, loss_function
        ),
    }
    if replay_loaders:
        base_metrics["replay_validation"] = metrics_for_loader(
            model, replay_loaders["validation"], device, loss_function
        )
        base_metrics["replay_test"] = metrics_for_loader(
            model, replay_loaders["test"], device, loss_function
        )

    trainable_names = configure_head_only_fine_tuning(model)
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -float("inf")
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.eval()
        train_losses = []
        train_bce_losses = []
        train_entropies = []
        for batch in combined_loaders["train"]:
            loss, bce_loss, attention_entropy = fine_tune_batch(
                model,
                batch,
                optimizer,
                loss_function,
                device=device,
                field_dropout=args.field_dropout,
                attention_entropy_weight=args.attention_entropy_weight,
                scaler=scaler,
            )
            train_losses.append(loss)
            train_bce_losses.append(bce_loss)
            train_entropies.append(attention_entropy)
        _, validation_labels, validation_probabilities, _ = evaluate(
            model,
            combined_loaders["validation"],
            device,
            loss_function,
        )
        validation_metrics = binary_metrics(
            validation_labels, validation_probabilities, threshold=0.5
        )
        score = validation_metrics["average_precision"]
        epoch_result = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_bce_loss": float(np.mean(train_bce_losses)),
            "train_normalized_attention_entropy": float(np.mean(train_entropies)),
            "validation": validation_metrics,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False), flush=True)
        if score > best_score:
            best_score = score
            best_state = copy_state(model)
    if best_state is None:
        raise RuntimeError("fine-tuning did not produce a candidate checkpoint")
    model.load_state_dict(best_state)
    model.eval()

    candidate_metrics = {
        "new_validation": metrics_for_loader(
            model, new_loaders["validation"], device, loss_function
        ),
        "new_test": metrics_for_loader(
            model, new_loaders["test"], device, loss_function
        ),
    }
    if replay_loaders:
        candidate_metrics["replay_validation"] = metrics_for_loader(
            model, replay_loaders["validation"], device, loss_function
        )
        candidate_metrics["replay_test"] = metrics_for_loader(
            model, replay_loaders["test"], device, loss_function
        )

    calibration_frame = combined_splits["train"][
        combined_splits["train"]["label"] == "normal"
    ]
    _, calibration_loader = make_loader(
        calibration_frame,
        tokenizer,
        batch_size=args.batch_size,
        device=device,
        shuffle=False,
    )
    calibration_profiles, calibration_scores = collect_normal_event_scores(
        model, calibration_loader, device
    )
    incoming_calibrator = NormalScoreCalibrator.fit(
        calibration_profiles, calibration_scores
    )
    existing_values = checkpoint.get("score_calibrator")
    existing_calibrator = (
        NormalScoreCalibrator.from_dict(existing_values)
        if existing_values
        else NormalScoreCalibrator()
    )
    merged_calibrator = NormalScoreCalibrator(
        profiles={
            **existing_calibrator.profiles,
            **incoming_calibrator.profiles,
        },
        minimum_records=existing_calibrator.minimum_records,
        slope=existing_calibrator.slope,
    )

    updated_names = changed_parameter_names(base_state, best_state)
    unexpected_updates = [
        name for name in updated_names if not is_fine_tunable_parameter(name)
    ]
    if unexpected_updates:
        raise RuntimeError(
            "frozen parameters changed during fine-tuning: "
            + ", ".join(unexpected_updates)
        )
    validation_ap_gain = (
        candidate_metrics["new_validation"]["average_precision"]
        - base_metrics["new_validation"]["average_precision"]
    )
    validation_fpr_increase = (
        candidate_metrics["new_validation"]["false_positive_rate"]
        - base_metrics["new_validation"]["false_positive_rate"]
    )
    replay_f1_drop = 0.0
    if replay_loaders:
        replay_f1_drop = (
            base_metrics["replay_validation"]["f1"]
            - candidate_metrics["replay_validation"]["f1"]
        )
    automatic_gate_passed = (
        bool(updated_names)
        and validation_ap_gain >= args.minimum_validation_ap_gain
        and validation_fpr_increase <= args.max_validation_fpr_increase
        and replay_f1_drop <= args.max_replay_f1_drop
    )
    gate = {
        "passed": automatic_gate_passed,
        "validation_ap_gain": validation_ap_gain,
        "minimum_validation_ap_gain": args.minimum_validation_ap_gain,
        "validation_fpr_increase": validation_fpr_increase,
        "max_validation_fpr_increase": args.max_validation_fpr_increase,
        "replay_f1_drop": replay_f1_drop if replay_loaders else None,
        "max_replay_f1_drop": args.max_replay_f1_drop if replay_loaders else None,
        "deployment_status": (
            "requires_admin_approval"
            if automatic_gate_passed
            else "do_not_deploy"
        ),
    }
    fine_tuning_record = {
        "parent_checkpoint": str(checkpoint_path.resolve()),
        "data": str(Path(args.data).resolve()),
        "replay_data": (
            str(Path(args.replay_data).resolve()) if args.replay_data else None
        ),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "trainable_parameters": trainable_names,
        "updated_parameters": updated_names,
        "base_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "gate": gate,
        "history": history,
    }
    checkpoint["state_dict"] = best_state
    checkpoint["baseline_registry"] = baseline_registry.to_dict()
    checkpoint["score_calibrator"] = merged_calibrator.to_dict()
    checkpoint["threshold"] = 0.5
    checkpoint["seen_fields"] = sorted(
        set(checkpoint.get("seen_fields", []))
        | combined_datasets["train"].seen_fields
    )
    checkpoint["seen_datasets"] = sorted(
        set(checkpoint.get("seen_datasets", []))
        | set(combined_splits["train"]["dataset"].astype(str))
    )
    checkpoint["seen_source_types"] = sorted(
        set(checkpoint.get("seen_source_types", []))
        | set(combined_splits["train"]["source_type"].astype(str))
    )
    checkpoint.setdefault("fine_tuning_history", []).append(fine_tuning_record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    report = {
        "candidate_checkpoint": str(output_path.resolve()),
        "deployment_status": gate["deployment_status"],
        "automatic_gate": gate,
        "trainable_parameters": trainable_names,
        "updated_parameters": updated_names,
        "base_metrics": base_metrics,
        "candidate_metrics": candidate_metrics,
        "calibration_profiles_updated": sorted(incoming_calibrator.profiles),
    }
    report_path = output_path.with_suffix(".fine_tuning.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
