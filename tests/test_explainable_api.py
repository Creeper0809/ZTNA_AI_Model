import json
from threading import Thread
from urllib.request import Request, urlopen

from ztna_ueba.api import create_server
from ztna_ueba.baseline import BaselineRegistry
from ztna_ueba.service import ExplainableTrustService
from ztna_ueba.timeline import TimelineStore
from ztna_ueba.tokenizer import DEFAULT_EXCLUDED_FIELDS


class FakePredictor:
    def __init__(self):
        normals = [
            {
                "dataset": "company",
                "source_type": "vpn",
                "event_type": "session",
                "event_time": f"2026-07-{day:02d}T09:00:00Z",
                "actor_alias": f"u{day:02d}",
                "auth_attempts": 1,
            }
            for day in range(1, 25)
        ]
        self.baseline_registry = BaselineRegistry.fit(normals, DEFAULT_EXCLUDED_FIELDS)

    def predict(self, payload, *, top_k=10):
        risk = 0.95 if int(payload.get("auth_attempts", 0)) >= 10 else 0.05
        stage = "deny" if risk >= 0.9 else "allow"
        return {
            "model_version": "fake-v1",
            "trust_score": 100 * (1 - risk),
            "risk_score": risk,
            "raw_model_risk_probability": risk,
            "monotonic_baseline": True,
            "portable_mode": True,
            "model_dimension": 128,
            "field_context_layers": 2,
            "confidence": 0.9,
            "baseline_ready": True,
            "score_calibration_ready": True,
            "shadow_mode_required": False,
            "policy": {"stage": stage, "action": stage},
            "top_fields": [
                {
                    "event_index": 0,
                    "field": "auth_attempts",
                    "event_field_count": 1,
                    "weight": 0.8,
                    "event_weight": 1.0,
                    "field_weight_within_event": 0.8,
                    "field_importance_logit": 2.0,
                    "field_weight_rank": 1,
                    "request_field_count": 1,
                    "risk_logit_contribution": 1.5 if risk > 0.5 else -0.2,
                }
            ],
        }


def test_http_assess_and_actor_timeline_round_trip():
    store = TimelineStore()
    service = ExplainableTrustService(FakePredictor(), store)
    server = create_server(service, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    event = {
        "event_id": "evt-http",
        "dataset": "company",
        "source_type": "vpn",
        "event_type": "session",
        "event_time": "2026-08-03T03:00:00Z",
        "actor_alias": "u99",
        "auth_attempts": 17,
        "label": "attack",
    }
    try:
        request = Request(
            f"{base}/v1/assess",
            data=json.dumps({"event": event}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            assessment = json.load(response)
        with urlopen(f"{base}/v1/actors/u99/timeline", timeout=5) as response:
            timeline = json.load(response)
        with urlopen(f"{base}/v1/events/evt-http", timeout=5) as response:
            raw_event = json.load(response)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        store.close()

    assert assessment["decision"]["policy"]["stage"] == "deny"
    assert "정상 중앙값" in assessment["readable_log"]["reason"]
    assert assessment["readable_log"]["event_summary"] == assessment["readable_log"]["event"]
    assert assessment["readable_log"]["judgment_reason"] == assessment["readable_log"]["reason"]
    assert assessment["readable_log"]["policy_reason"] == assessment["readable_log"]["decision"]
    assert "평소 행동 기준" in assessment["readable_log"]["usual_behavior"]
    assert "현재 인증 시도 횟수 값은 17" in assessment["readable_log"]["current_behavior"]
    assert assessment["readable_log"]["ueba_judgment"].startswith("UEBA는")
    assert "재검사하자" not in assessment["readable_log"]["judgment_reason"]
    assert assessment["evidence"]["influence_check"]["method"] == (
        "counterfactual_coalition_v3"
    )
    coalition = assessment["evidence"]["influence_check"]["coalition_analysis"]
    assert coalition["method"] == "exact_subset_shapley_v2"
    assert assessment["evidence"]["influence_check"]["explanation_confidence"]
    influence_reason = assessment["readable_log"]["model_influence_reason"]
    assert "재검사하자" in influence_reason
    assert "%에서" in influence_reason
    assert "%p 낮아졌다" in influence_reason
    scope_reason = assessment["readable_log"]["analysis_scope_reason"]
    assert "후보를 모으고" in scope_reason
    assert "부분 조합을 검사했다" in scope_reason
    quality_reason = assessment["readable_log"]["explanation_confidence_reason"]
    assert "설명 품질은" in quality_reason
    assert "후보 점수 포착률" in quality_reason
    assert "점으로" not in quality_reason
    assert coalition["field_marginal_contributions"][0]["field_label"]
    assert assessment["evidence"]["influence_check"]["primary_evidence"][
        "type"
    ] == "single_field"
    assert timeline["summary"]["returned_events"] == 1
    assert timeline["timeline"][0]["event_id"] == "evt-http"
    assert "label" not in raw_event["event"]
