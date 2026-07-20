"""Run one Trust Score prediction and emit field-level explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from .baseline import BaselineRegistry
from .calibration import NormalScoreCalibrator
from .explain import explain_request
from .model import HierarchicalFieldAttention, ModelConfig
from .policy import map_risk_to_policy
from .tokenizer import FieldTokenizer, TokenizerConfig, collate_requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", help="JSON file; stdin is used when omitted")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
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
    tokenizer_config = TokenizerConfig(**tokenizer_values)
    tokenizer = FieldTokenizer(tokenizer_config, baseline_registry=baseline_registry)
    model = HierarchicalFieldAttention(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    payload_text = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    payload = json.loads(payload_text)
    records = payload["events"] if isinstance(payload, dict) and "events" in payload else payload
    record_list = [records] if isinstance(records, dict) else list(records)
    tokenized = tokenizer.tokenize_request(records)
    batch = collate_requests([tokenized])
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        output = model(tensor_batch)
    calibrated_risk = output["risk_probability"]
    score_calibration_ready = not tokenizer_config.portable_mode
    if score_calibrator is not None:
        calibrated_risk, calibration_ready = score_calibrator.normalize_batch(
            batch["profile_keys"],
            output["event_anomaly_score"],
            output["event_weights"],
            tensor_batch["event_mask"],
        )
        score_calibration_ready = bool(calibration_ready[0].cpu())
    field_names = batch["field_names"][0]
    explanation = explain_request(
        field_names,
        output["event_weights"][0].cpu(),
        output["field_weights"][0].cpu(),
        output["field_contributions"][0].cpu(),
        top_k=args.top_k,
    )
    seen = set(checkpoint.get("seen_fields", []))
    current = {name for event in field_names for name in event}
    novelty_ratio = len(current - seen) / max(1, len(current))
    seen_datasets = set(checkpoint.get("seen_datasets", []))
    seen_source_types = set(checkpoint.get("seen_source_types", []))
    current_datasets = {
        str(record["dataset"])
        for record in record_list
        if record.get("dataset") not in {None, ""}
    }
    current_source_types = {
        str(record["source_type"])
        for record in record_list
        if record.get("source_type") not in {None, ""}
    }
    unseen_datasets = sorted(current_datasets - seen_datasets)
    unseen_source_types = sorted(current_source_types - seen_source_types)
    source_metadata_missing = not current_datasets and not current_source_types
    baseline_coverages = (
        [
            baseline_registry.coverage(record, tokenizer_config.excluded_fields)
            for record in record_list
        ]
        if baseline_registry is not None
        else []
    )
    minimum_baseline_coverage = (
        min(item["coverage"] for item in baseline_coverages)
        if baseline_coverages
        else 0.0
    )
    baseline_ready = bool(baseline_coverages) and all(
        item["profile_known"] and item["coverage"] >= 0.80
        for item in baseline_coverages
    )
    attention_concentration = float(output["attention_concentration"][0].cpu())
    confidence_margin = (
        abs(2.0 * float(calibrated_risk[0].cpu()) - 1.0)
        if score_calibration_ready
        else abs(2.0 * float(output["risk_probability"][0].cpu()) - 1.0)
    )
    raw_confidence = 0.5 * attention_concentration + 0.5 * confidence_margin
    ood_factor = 0.5 if unseen_datasets or unseen_source_types or source_metadata_missing else 1.0
    if tokenizer_config.portable_mode and (
        not baseline_ready or not score_calibration_ready
    ):
        ood_factor *= 0.25
    confidence = raw_confidence * (1.0 - 0.5 * novelty_ratio) * ood_factor
    risk_probability = (
        float(calibrated_risk[0].cpu())
        if score_calibration_ready
        else None
    )
    shadow_mode_required = (
        confidence < 0.40
        or novelty_ratio > 0.30
        or bool(unseen_datasets)
        or bool(unseen_source_types)
        or source_metadata_missing
        or (
            tokenizer_config.portable_mode
            and (not baseline_ready or not score_calibration_ready)
        )
    )
    result = {
        "trust_score": (
            100.0 * (1.0 - risk_probability)
            if risk_probability is not None
            else None
        ),
        "risk_score": risk_probability,
        "raw_model_risk_probability": float(output["risk_probability"][0].cpu()),
        "confidence": confidence,
        "novel_field_ratio": novelty_ratio,
        "unseen_datasets": unseen_datasets,
        "unseen_source_types": unseen_source_types,
        "source_metadata_missing": source_metadata_missing,
        "baseline_ready": baseline_ready,
        "score_calibration_ready": score_calibration_ready,
        "minimum_baseline_coverage": minimum_baseline_coverage,
        "baseline_coverages": baseline_coverages,
        "decision_threshold": float(checkpoint["threshold"]),
        "shadow_mode_required": shadow_mode_required,
        "policy": map_risk_to_policy(
            risk_probability
            if risk_probability is not None
            else float(output["risk_probability"][0].cpu()),
            shadow_mode_required,
        ),
        "source_weights": output["event_weights"][0, : len(field_names)].cpu().tolist(),
        **explanation,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
