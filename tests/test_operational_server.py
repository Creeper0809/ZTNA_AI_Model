from datetime import datetime, timezone
import time

from fastapi.testclient import TestClient

from ztna_ueba.baseline import BaselineRegistry
from ztna_ueba.server import create_app
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
                "actor_alias": "employee-01",
                "auth_attempts": 1,
            }
            for day in range(1, 25)
        ]
        self.baseline_registry = BaselineRegistry.fit(normals, DEFAULT_EXCLUDED_FIELDS)

    def predict(self, payload, *, top_k=10):
        attempts = int(payload.get("auth_attempts", 0))
        risk = 0.95 if attempts >= 10 else 0.05
        stage = "deny" if risk >= 0.9 else "allow"
        return {
            "model_version": "fake-v1",
            "trust_score": 100 * (1 - risk),
            "risk_score": risk,
            "raw_model_risk_probability": risk,
            "monotonic_baseline": True,
            "portable_mode": True,
            "model_dimension": 32,
            "field_context_layers": 1,
            "confidence": 0.91,
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


def _event() -> dict:
    return {
        "event_id": "evt-operational",
        "dataset": "company",
        "source_type": "vpn",
        "event_type": "session",
        "event_time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "actor_alias": "employee-01",
        "auth_attempts": 17,
        "label": "attack",
    }


def test_operational_api_scores_then_completes_explanation_and_review():
    store = TimelineStore()
    service = ExplainableTrustService(FakePredictor(), store)
    app = create_app(service, api_key="test-key")

    with TestClient(app) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/api/v1/overview").status_code == 401

        headers = {"X-API-Key": "test-key"}
        response = client.post(
            "/api/v1/assess",
            headers=headers,
            json={"event": _event(), "explanation_mode": "auto"},
        )
        assert response.status_code == 201
        payload = response.json()
        assert payload["decision"]["policy"]["stage"] == "deny"
        assert payload["explanation_status"] == "pending"
        assert response.headers["X-Request-ID"]

        detail = None
        for _ in range(100):
            detail_response = client.get(
                "/api/v1/events/evt-operational", headers=headers
            )
            assert detail_response.status_code == 200
            detail = detail_response.json()
            if detail["explanation_status"] == "completed":
                break
            time.sleep(0.01)
        assert detail is not None
        assert detail["explanation_status"] == "completed"
        assert detail["evidence"]["influence_check"]["coalition_analysis"]
        assert "label" not in detail["raw_event"]

        overview = client.get("/api/v1/overview", headers=headers).json()
        assert overview["total_events"] == 1
        assert overview["suspicious_events"] == 1

        events = client.get(
            "/api/v1/events?suspicious_only=true", headers=headers
        ).json()
        assert events["total"] == 1
        assert events["items"][0]["event_id"] == "evt-operational"

        review = client.patch(
            "/api/v1/events/evt-operational/review",
            headers=headers,
            json={"status": "investigating", "note": "관제 확인 중"},
        )
        assert review.status_code == 200
        updated = client.get(
            "/api/v1/events/evt-operational", headers=headers
        ).json()
        assert updated["review_status"] == "investigating"
        assert updated["review_note"] == "관제 확인 중"

        assert client.get("/").status_code == 200

    store.close()


def test_allow_event_skips_expensive_explanation_in_auto_mode():
    store = TimelineStore()
    service = ExplainableTrustService(FakePredictor(), store)
    app = create_app(service)
    event = {**_event(), "event_id": "evt-allow", "auth_attempts": 1}

    with TestClient(app) as client:
        response = client.post("/api/v1/assess", json={"event": event})
        assert response.status_code == 201
        payload = response.json()
        assert payload["decision"]["policy"]["stage"] == "allow"
        assert payload["explanation_status"] == "skipped"
        detail = client.get("/api/v1/events/evt-allow").json()
        assert detail["explanation_status"] == "skipped"
        assert detail["evidence"]["influence_check"] == {}

        batch = client.post(
            "/api/v1/assess/batch",
            json={
                "explanation_mode": "none",
                "events": [
                    {**event, "event_id": "evt-batch-1"},
                    {**event, "event_id": "evt-batch-2"},
                ],
            },
        )
        assert batch.status_code == 201
        assert batch.json()["accepted"] == 2

    store.close()


def test_sanitized_snapshot_keeps_credential_metadata_but_removes_secrets():
    store = TimelineStore()
    service = ExplainableTrustService(FakePredictor(), store)
    app = create_app(service)
    event = {
        **_event(),
        "event_id": "evt-credential",
        "credential_id_hash": "credential_abc123",
        "credential_type": "passkey",
        "password": "must-not-be-stored",
        "access_token": "must-not-be-stored-either",
    }

    with TestClient(app) as client:
        client.post(
            "/api/v1/assess",
            json={"event": event, "explanation_mode": "none"},
        )
        raw = client.get("/api/v1/events/evt-credential").json()["raw_event"]
        assert raw["credential_id_hash"] == "credential_abc123"
        assert raw["credential_type"] == "passkey"
        assert raw["password"] == "[REDACTED]"
        assert raw["access_token"] == "[REDACTED]"

    store.close()


def test_server_restart_recovers_pending_explanation():
    store = TimelineStore()
    service = ExplainableTrustService(FakePredictor(), store)
    service.assess_fast(_event(), explanation_status="pending")
    app = create_app(service)

    with TestClient(app) as client:
        detail = None
        for _ in range(100):
            detail = client.get("/api/v1/events/evt-operational").json()
            if detail["explanation_status"] == "completed":
                break
            time.sleep(0.01)
        assert detail is not None
        assert detail["explanation_status"] == "completed"

    store.close()


def test_event_id_rejects_path_unsafe_characters():
    store = TimelineStore()
    service = ExplainableTrustService(FakePredictor(), store)
    app = create_app(service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/assess",
            json={"event": {**_event(), "event_id": "unsafe/id"}},
        )
        assert response.status_code == 422

    store.close()
