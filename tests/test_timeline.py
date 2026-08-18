from ztna_ueba.operator_explain import build_operator_explanation
from ztna_ueba.timeline import TimelineStore


def _prediction(risk: float, stage: str, field: str = "auth_attempts") -> dict:
    return {
        "model_version": "fake-v1",
        "trust_score": 100 * (1 - risk),
        "risk_score": risk,
        "raw_model_risk_probability": risk,
        "confidence": 0.9,
        "policy": {"stage": stage, "action": stage},
        "top_fields": [
            {
                "event_index": 0,
                "field": field,
                "weight": 0.7,
                "risk_logit_contribution": risk,
            }
        ],
    }


def test_actor_timeline_orders_suspicious_events_and_explains_links():
    store = TimelineStore()
    events = [
        {
            "event_id": "evt-normal",
            "dataset": "company",
            "source_type": "vpn",
            "event_type": "session",
            "event_time": "2026-08-03T08:00:00Z",
            "actor_alias": "u99",
            "device_id_hash": "device-1",
            "session_id": "session-1",
            "auth_attempts": 1,
            "label": "normal",
        },
        {
            "event_id": "evt-auth",
            "dataset": "company",
            "source_type": "iam",
            "event_type": "authentication",
            "event_time": "2026-08-03T09:00:00Z",
            "actor_alias": "u99",
            "device_id_hash": "device-1",
            "session_id": "session-1",
            "auth_attempts": 12,
            "label": "attack",
        },
        {
            "event_id": "evt-ndr",
            "dataset": "company",
            "source_type": "ndr",
            "event_type": "network_flow",
            "event_time": "2026-08-03T09:05:00Z",
            "actor_alias": "u99",
            "device_id_hash": "device-1",
            "session_id": "session-1",
            "dst_ip": "203.0.113.10",
            "dst_port": 4444,
            "label": "attack",
        },
    ]
    predictions = [
        _prediction(0.05, "allow"),
        _prediction(0.82, "restrict"),
        _prediction(0.97, "deny", field="dst_port"),
    ]
    for event, prediction in zip(events, predictions, strict=True):
        explanation = build_operator_explanation(event, prediction, None)
        store.record(event, prediction, explanation)

    result = store.actor_timeline("u99")

    assert [item["event_id"] for item in result["timeline"]] == ["evt-auth", "evt-ndr"]
    assert result["summary"]["returned_events"] == 2
    assert result["summary"]["source_count"] == 2
    assert result["summary"]["maximum_risk"] == 0.97
    links = result["timeline"][1]["linked_from_previous"]
    assert links[0]["type"] == "same_actor"
    assert {item.get("field") for item in links} >= {"device_id_hash", "session_id"}
    assert store.raw_event("evt-auth").get("label") is None

    latest = store.actor_timeline("u99", suspicious_only=False, limit=2)
    assert [item["event_id"] for item in latest["timeline"]] == ["evt-auth", "evt-ndr"]
    store.close()
