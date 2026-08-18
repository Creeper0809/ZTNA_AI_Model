"""Build reproducible normal-only profiles for the complete BNDT examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

COMPLETE_NDR_CONTEXT: dict[str, Any] = {
    "auth_attempts": 1,
    "failed_login_attempts": 0,
    "geo_zone": "seoul",
    "download_mb": 100,
    "ip_reputation": "highly_trusted",
    "vpn_approval_status": "approved",
    "network_traffic_status": "normal",
    "latency_ms": 100,
    "device_health": "healthy",
    "threat_history": 20,
    "security_policy_violation_history": 20,
    "account_compromise_history": 20,
    "policy_violation_frequency": 20,
    "response_to_past_threats": 20,
}

COMPLETE_BACKUP_CONTEXT: dict[str, Any] = {
    "ip_reputation": "highly_trusted",
    "vpn_approval_status": "approved",
    "threat_history": 20,
    "security_policy_violation_history": 20,
    "account_compromise_history": 20,
    "policy_violation_frequency": 20,
    "response_to_past_threats": 20,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-sample",
        default=str(ROOT / "artifacts" / "training_sample.csv.gz"),
    )
    parser.add_argument(
        "--backup-normal",
        default=str(ROOT / "examples" / "approved_backup_normal.jsonl"),
    )
    parser.add_argument(
        "--ndr-output",
        default=str(ROOT / "examples" / "complete_bndt_ndr_normal.jsonl"),
    )
    parser.add_argument(
        "--backup-output",
        default=str(ROOT / "examples" / "complete_bndt_backup_normal.jsonl"),
    )
    parser.add_argument("--ndr-records", type=int, default=100)
    return parser.parse_args()


def clean_sample_record(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = {name: value for name, value in record.items() if pd.notna(value)}
    for name in ("src_port", "dst_port", "_sample_hash"):
        if name in cleaned:
            cleaned[name] = int(cleaned[name])
    return cleaned


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.training_sample)
    selected = frame[
        (frame["dataset"] == "unsw_nb15")
        & (frame["source_type"] == "ndr")
        & (frame["label"] == "normal")
        & (frame["split"] == "train")
    ].head(args.ndr_records)
    if len(selected) < 20:
        raise ValueError("at least 20 UNSW-NB15 train-normal NDR records are required")

    ndr_records = []
    for raw_record in selected.to_dict(orient="records"):
        record = clean_sample_record(raw_record)
        record.update(COMPLETE_NDR_CONTEXT)
        ndr_records.append(record)

    backup_records = []
    for line in Path(args.backup_normal).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record.update(COMPLETE_BACKUP_CONTEXT)
        backup_records.append(record)
    if len(backup_records) < 20:
        raise ValueError("at least 20 approved backup records are required")

    ndr_output = Path(args.ndr_output)
    backup_output = Path(args.backup_output)
    write_jsonl(ndr_output, ndr_records)
    write_jsonl(backup_output, backup_records)
    print(
        json.dumps(
            {
                "ndr_output": str(ndr_output.resolve()),
                "ndr_records": len(ndr_records),
                "backup_output": str(backup_output.resolve()),
                "backup_records": len(backup_records),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
