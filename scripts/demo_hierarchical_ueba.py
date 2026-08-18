"""Demonstrate entity-specific UEBA baselines and deterministic fallback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ztna_ueba.baseline import BaselineRegistry
from ztna_ueba.tokenizer import DEFAULT_EXCLUDED_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="artifacts/hierarchical_ueba_poc/demo_output.json",
    )
    return parser.parse_args()


def approved_normals() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for actor, attempts in (("analyst-a", 1), ("analyst-b", 10)):
        records.extend(
            {
                "dataset": "company_demo",
                "source_type": "auth",
                "event_type": "login",
                "actor_alias": actor,
                "device_id_hash": f"device-{actor}",
                "department": "security",
                "auth_attempts": attempts,
                "latency_ms": 50,
            }
            for _ in range(25)
        )
    records.extend(
        {
            "dataset": "company_demo",
            "source_type": "vpn",
            "event_type": "session",
            "actor_alias": f"guest-{index}",
            "device_id_hash": "shared-finance-device",
            "department": "finance",
            "auth_attempts": 2,
            "latency_ms": 100 + index % 3,
        }
        for index in range(25)
    )
    return records


def selection_summary(
    registry: BaselineRegistry,
    event: dict[str, object],
    field_name: str,
) -> dict[str, object]:
    features, selection = registry.field_features(
        event, field_name, event[field_name]
    )
    return {
        "scope": selection.scope,
        "scope_label": selection.scope_label,
        "normal_records": selection.records,
        "field_observations": selection.field_observations,
        "fallback_used": selection.fallback_used,
        "numeric_baseline_deviation": features[4],
        "selection_reason": selection.selection_reason,
    }


def main() -> None:
    args = parse_args()
    normals = approved_normals()
    registry = BaselineRegistry.fit(normals, DEFAULT_EXCLUDED_FIELDS)

    actor_a = {**normals[0], "auth_attempts": 10}
    actor_b = {**normals[25], "auth_attempts": 10}
    device_event = {
        **normals[-1],
        "actor_alias": "new-finance-user",
    }
    peer_event = {
        **device_event,
        "device_id_hash": "new-finance-device",
    }
    profile_event = {
        **peer_event,
        "department": "new-department",
    }
    serialized = json.dumps(registry.to_dict(), sort_keys=True)
    output = {
        "claim": (
            "같은 로그 스키마라도 사용자별 정상 행동이 다르면 서로 다른 기준선을 "
            "사용하고, 표본이 부족하면 단말·동료 집단·로그 유형 순으로 대체한다."
        ),
        "same_schema_different_actor": {
            "actor_a_observed_attempts": 10,
            "actor_a": selection_summary(
                registry, actor_a, "auth_attempts"
            ),
            "actor_b_observed_attempts": 10,
            "actor_b": selection_summary(
                registry, actor_b, "auth_attempts"
            ),
        },
        "fallback_chain": {
            "new_actor_known_device": selection_summary(
                registry, device_event, "latency_ms"
            ),
            "new_actor_new_device_known_peer": selection_summary(
                registry, peer_event, "latency_ms"
            ),
            "new_actor_device_and_peer": selection_summary(
                registry, profile_event, "latency_ms"
            ),
        },
        "privacy": {
            "raw_actor_absent_from_registry": "analyst-a" not in serialized,
            "raw_device_absent_from_registry": "shared-finance-device" not in serialized,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
