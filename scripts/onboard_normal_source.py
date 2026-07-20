"""Onboard an arbitrary log schema using operator-approved normal-only records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from ztna_ueba.baseline import BaselineRegistry
from ztna_ueba.calibration import NormalScoreCalibrator
from ztna_ueba.data import NORMAL_LABELS, move_tensors
from ztna_ueba.model import HierarchicalFieldAttention, ModelConfig
from ztna_ueba.tokenizer import FieldTokenizer, TokenizerConfig, collate_requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--normal-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--source-type")
    parser.add_argument("--event-type")
    parser.add_argument("--confirmed-normal", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, object]]:
    lower_name = path.name.lower()
    if lower_name.endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, compression="infer", dtype="string").to_dict(
            orient="records"
        )
    if lower_name.endswith(".jsonl"):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "events" in payload:
        payload = payload["events"]
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("normal data must be CSV, JSONL, a JSON object, or a JSON list")
    return payload


def validate_normal_confirmation(records: list[dict[str, object]], confirmed: bool) -> None:
    if not confirmed:
        raise ValueError("--confirmed-normal is required; onboarding data changes the trusted baseline")
    for record in records:
        for field in ("label", "is_attack", "target", "ground_truth"):
            value = record.get(field)
            if value is None or str(value).strip().lower() in {"", "<na>", "nan"}:
                continue
            if str(value).strip().lower() not in NORMAL_LABELS:
                raise ValueError(f"non-normal label found in onboarding data: {field}={value}")


def apply_metadata(records: list[dict[str, object]], args: argparse.Namespace) -> None:
    overrides = {
        "dataset": args.dataset,
        "source_type": args.source_type,
        "event_type": args.event_type,
    }
    for record in records:
        for field, value in overrides.items():
            if value:
                record[field] = value
        missing = [field for field in overrides if not str(record.get(field, "")).strip()]
        if missing:
            raise ValueError(
                "profile metadata is required; provide columns or CLI values for "
                + ", ".join(missing)
            )


@torch.no_grad()
def collect_scores(
    records: list[dict[str, object]],
    tokenizer: FieldTokenizer,
    model: HierarchicalFieldAttention,
    device: torch.device,
    batch_size: int,
) -> tuple[list[str], list[float], set[str]]:
    profiles: list[str] = []
    scores: list[float] = []
    fields: set[str] = set()
    for start in range(0, len(records), batch_size):
        requests = [
            tokenizer.tokenize_request(record)
            for record in records[start : start + batch_size]
        ]
        batch = move_tensors(collate_requests(requests), device)
        output = model(batch)
        for row_index, request_profiles in enumerate(batch["profile_keys"]):
            for event_index, profile_key in enumerate(request_profiles):
                profiles.append(profile_key)
                scores.append(float(output["event_anomaly_score"][row_index, event_index].cpu()))
        for request_names in batch["field_names"]:
            for event_names in request_names:
                fields.update(event_names)
    return profiles, scores, fields


def main() -> None:
    args = parse_args()
    input_path = Path(args.normal_data)
    output_path = Path(args.output)
    if Path(args.checkpoint).resolve() == output_path.resolve():
        raise ValueError("output must differ from the input checkpoint")
    records = load_records(input_path)
    if len(records) < 20:
        raise ValueError("at least 20 approved normal records are required")
    validate_normal_confirmation(records, args.confirmed_normal)
    apply_metadata(records, args)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer_values = dict(checkpoint["tokenizer_config"])
    tokenizer_values["excluded_fields"] = frozenset(tokenizer_values["excluded_fields"])
    tokenizer_values["profile_fields"] = tuple(tokenizer_values["profile_fields"])
    tokenizer_config = TokenizerConfig(**tokenizer_values)
    if not tokenizer_config.portable_mode:
        raise ValueError("normal-only onboarding requires a portable checkpoint")

    existing_registry = BaselineRegistry.from_dict(checkpoint["baseline_registry"])
    incoming_registry = BaselineRegistry.fit(
        records,
        excluded_fields=tokenizer_config.excluded_fields,
        profile_fields=tokenizer_config.profile_fields,
    )
    merged_registry = BaselineRegistry(
        profiles={**existing_registry.profiles, **incoming_registry.profiles},
        profile_fields=existing_registry.profile_fields,
    )
    tokenizer = FieldTokenizer(tokenizer_config, merged_registry)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    model = HierarchicalFieldAttention(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    profiles, scores, fields = collect_scores(
        records, tokenizer, model, device, args.batch_size
    )
    incoming_calibrator = NormalScoreCalibrator.fit(profiles, scores)
    if not incoming_calibrator.profiles:
        raise ValueError("no profile has enough records for score calibration")
    existing_values = checkpoint.get("score_calibrator")
    existing_calibrator = (
        NormalScoreCalibrator.from_dict(existing_values)
        if existing_values
        else NormalScoreCalibrator()
    )
    merged_calibrator = NormalScoreCalibrator(
        profiles={**existing_calibrator.profiles, **incoming_calibrator.profiles},
        minimum_records=existing_calibrator.minimum_records,
        slope=existing_calibrator.slope,
    )

    checkpoint["baseline_registry"] = merged_registry.to_dict()
    checkpoint["score_calibrator"] = merged_calibrator.to_dict()
    checkpoint["seen_fields"] = sorted(set(checkpoint.get("seen_fields", [])) | fields)
    checkpoint["seen_datasets"] = sorted(
        set(checkpoint.get("seen_datasets", []))
        | {str(record["dataset"]) for record in records}
    )
    checkpoint["seen_source_types"] = sorted(
        set(checkpoint.get("seen_source_types", []))
        | {str(record["source_type"]) for record in records}
    )
    onboarding = checkpoint.setdefault("onboarding_history", [])
    onboarding.append(
        {
            "normal_data": str(input_path.resolve()),
            "records": len(records),
            "profiles": sorted(incoming_calibrator.profiles),
            "confirmed_normal": True,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    report = {
        "checkpoint": str(output_path.resolve()),
        "normal_records": len(records),
        "baseline_profiles_updated": sorted(incoming_registry.profiles),
        "calibration_profiles_ready": sorted(incoming_calibrator.profiles),
        "stored_raw_values": False,
    }
    report_path = output_path.with_suffix(".onboarding.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
