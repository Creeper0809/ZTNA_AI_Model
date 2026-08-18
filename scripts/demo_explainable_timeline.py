"""Run an end-to-end readable-log and actor-timeline PoC with the real model."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from ztna_ueba.inference import TrustPredictor
from ztna_ueba.service import ExplainableTrustService
from ztna_ueba.timeline import TimelineStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/complete_bndt_ueba/model.pt")
    parser.add_argument(
        "--output", default="artifacts/explainable_timeline_poc/demo_output.json"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


def build_demo_events() -> list[dict]:
    backup = json.loads(Path("examples/approved_backup_event.json").read_text(encoding="utf-8"))
    attack = json.loads(
        Path("examples/rule_allow_model_deny_event.json").read_text(encoding="utf-8")
    )
    backup.update(
        {
            "event_id": "demo-001-approved-backup",
            "event_time": "2026-08-03T03:00:00Z",
            "actor_alias": "svc_backup",
        }
    )
    first_attack = deepcopy(attack)
    first_attack.update(
        {
            "event_id": "demo-002-suspicious-flow",
            "event_time": "2026-08-03T03:07:00Z",
            "actor_alias": "svc_backup",
        }
    )
    second_attack = deepcopy(attack)
    second_attack.update(
        {
            "event_id": "demo-003-followup-flow",
            "event_time": "2026-08-03T03:12:00Z",
            "actor_alias": "svc_backup",
            "dst_ip": "149.171.126.18",
            "dst_port": 44322,
        }
    )
    return [backup, first_attack, second_attack]


def main() -> None:
    args = parse_args()
    predictor = TrustPredictor(args.checkpoint, device=args.device)
    store = TimelineStore()
    service = ExplainableTrustService(predictor, store)
    try:
        assessments = [service.assess(event) for event in build_demo_events()]
        suspicious = service.actor_timeline("svc_backup", suspicious_only=True)
        complete = service.actor_timeline("svc_backup", suspicious_only=False)
        output = {
            "poc_claim": (
                "사용자·서비스 계정, 단말, 동료 집단, 로그 유형 순으로 정상 기준선을 선택하고, "
                "기준선 편차와 AI 칼럼 가중치를 사람이 읽는 판정 근거로 반환하며, "
                "여러 칼럼 조합의 재추론으로 필드별 영향과 조합 효과를 검증한다."
            ),
            "assessments": assessments,
            "suspicious_timeline": suspicious,
            "complete_timeline": complete,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for assessment in assessments:
            print(assessment["readable_log"]["full_text"])
        print(suspicious["summary"]["narrative"])
        print(f"PoC output: {output_path}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
