"""Search reproducible bidirectional disagreements between rule and AI policies."""

from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ztna_ueba.baseline import BaselineRegistry
from ztna_ueba.calibration import NormalScoreCalibrator
from ztna_ueba.model import HierarchicalFieldAttention, ModelConfig
from ztna_ueba.policy import map_risk_to_policy
from ztna_ueba.rule_baseline import evaluate_paper_rule_baseline
from ztna_ueba.tokenizer import FieldTokenizer, TokenizerConfig, collate_requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def candidate_events() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for hour, device_health, auth_attempts, latency_ms, geo_zone in product(
        [0, 3, 6, 9, 12, 18, 21, 23],
        ["healthy", "degraded", "compromised"],
        [1, 4, 6, 12, 20, 30],
        [100, 450, 600, 1200, 2500],
        ["seoul", "busan", "incheon"],
    ):
        candidates.append(
            {
                "dataset": "company_demo",
                "source_type": "vpn",
                "event_type": "session",
                "event_time": f"2026-07-20T{hour:02d}:00:00Z",
                "actor_alias": "u25",
                "device_health": device_health,
                "auth_attempts": auth_attempts,
                "latency_ms": latency_ms,
                "geo_zone": geo_zone,
            }
        )
    base_candidates = list(candidates)
    rule_only_variants: tuple[dict[str, object], ...] = (
        {"failed_login_attempts": 50},
        {"download_mb": 50_000},
        {"ip_reputation": "malicious"},
        {"network_traffic_status": "abnormal"},
        {"network_traffic_status": "malicious"},
    )
    for record in base_candidates:
        for extra_fields in rule_only_variants:
            candidates.append({**record, **extra_fields})
    return candidates


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    tokenizer_values = dict(checkpoint["tokenizer_config"])
    tokenizer_values["excluded_fields"] = frozenset(tokenizer_values["excluded_fields"])
    if "profile_fields" in tokenizer_values:
        tokenizer_values["profile_fields"] = tuple(tokenizer_values["profile_fields"])
    tokenizer_config = TokenizerConfig(**tokenizer_values)

    baseline_values = checkpoint.get("baseline_registry")
    baseline_registry = BaselineRegistry.from_dict(baseline_values) if baseline_values else None
    calibrator_values = checkpoint.get("score_calibrator")
    score_calibrator = (
        NormalScoreCalibrator.from_dict(calibrator_values) if calibrator_values else None
    )

    tokenizer = FieldTokenizer(tokenizer_config, baseline_registry=baseline_registry)
    model = HierarchicalFieldAttention(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    candidates = candidate_events()
    tokenized = [tokenizer.tokenize_request(record) for record in candidates]
    batch = collate_requests(tokenized)
    tensor_batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.no_grad():
        output = model(tensor_batch)

    calibrated_risk = output["risk_probability"]
    calibration_ready = torch.full(
        (len(candidates),), not tokenizer_config.portable_mode, dtype=torch.bool
    )
    if score_calibrator is not None:
        calibrated_risk, calibration_ready = score_calibrator.normalize_batch(
            batch["profile_keys"],
            output["event_anomaly_score"],
            output["event_weights"],
            tensor_batch["event_mask"],
        )

    seen_fields = set(checkpoint.get("seen_fields", []))
    seen_datasets = set(checkpoint.get("seen_datasets", []))
    seen_source_types = set(checkpoint.get("seen_source_types", []))
    rows: list[dict[str, object]] = []

    for index, record in enumerate(candidates):
        field_names = batch["field_names"][index]
        current_fields = {name for event in field_names for name in event}
        novelty_ratio = len(current_fields - seen_fields) / max(1, len(current_fields))
        unseen_datasets = sorted({str(record["dataset"])} - seen_datasets)
        unseen_source_types = sorted({str(record["source_type"])} - seen_source_types)

        coverage = (
            baseline_registry.coverage(record, tokenizer_config.excluded_fields)
            if baseline_registry is not None
            else None
        )
        baseline_ready = bool(
            coverage is not None
            and coverage["profile_known"]
            and coverage["coverage"] >= 0.80
        )
        calibration_is_ready = bool(calibration_ready[index].cpu())
        risk = (
            float(calibrated_risk[index].cpu())
            if calibration_is_ready
            else float(output["risk_probability"][index].cpu())
        )
        attention_concentration = float(output["attention_concentration"][index].cpu())
        confidence_margin = abs(2.0 * risk - 1.0)
        raw_confidence = 0.5 * attention_concentration + 0.5 * confidence_margin
        ood_factor = 0.5 if unseen_datasets or unseen_source_types else 1.0
        if tokenizer_config.portable_mode and (
            not baseline_ready or not calibration_is_ready
        ):
            ood_factor *= 0.25
        confidence = raw_confidence * (1.0 - 0.5 * novelty_ratio) * ood_factor
        shadow_mode_required = (
            confidence < 0.40
            or novelty_ratio > 0.30
            or bool(unseen_datasets)
            or bool(unseen_source_types)
            or (
                tokenizer_config.portable_mode
                and (not baseline_ready or not calibration_is_ready)
            )
        )
        proposed_policy = map_risk_to_policy(risk, shadow_mode_required)
        rule = evaluate_paper_rule_baseline(record)
        rows.append(
            {
                "event": record,
                "rule_trust_score": rule["trust_score"],
                "rule_stage": rule["policy"]["stage"],
                "proposed_trust_score": 100.0 * (1.0 - risk),
                "proposed_risk_score": risk,
                "proposed_stage": proposed_policy["stage"],
                "confidence": confidence,
                "shadow_mode_required": shadow_mode_required,
            }
        )

    rule_allow_model_nonallow = [
        row
        for row in rows
        if row["rule_stage"] == "allow" and row["proposed_stage"] != "allow"
    ]
    rule_nonallow_model_allow = [
        row
        for row in rows
        if row["rule_stage"] != "allow" and row["proposed_stage"] == "allow"
    ]
    rule_block_model_allow = [
        row
        for row in rows
        if row["rule_stage"] == "block" and row["proposed_stage"] == "allow"
    ]
    rule_allow_model_nonallow.sort(
        key=lambda row: (-row["proposed_risk_score"], -row["rule_trust_score"])
    )
    rule_nonallow_model_allow.sort(
        key=lambda row: (row["proposed_risk_score"], row["rule_trust_score"])
    )
    rule_block_model_allow.sort(
        key=lambda row: (row["proposed_risk_score"], row["rule_trust_score"])
    )
    model_allow = [row for row in rows if row["proposed_stage"] == "allow"]
    model_allow.sort(key=lambda row: (row["rule_trust_score"], row["proposed_risk_score"]))

    result = {
        "candidate_count": len(candidates),
        "match_counts": {
            "rule_allow_model_nonallow": len(rule_allow_model_nonallow),
            "rule_nonallow_model_allow": len(rule_nonallow_model_allow),
            "rule_block_model_allow": len(rule_block_model_allow),
        },
        "rule_allow_model_nonallow": rule_allow_model_nonallow[: args.top_k],
        "rule_nonallow_model_allow": rule_nonallow_model_allow[: args.top_k],
        "rule_block_model_allow": rule_block_model_allow[: args.top_k],
        "model_allow_with_lowest_rule_score": model_allow[: args.top_k],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
