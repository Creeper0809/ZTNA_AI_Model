from ztna_ueba.baseline import BaselineRegistry
from ztna_ueba.operator_explain import build_operator_explanation, sanitize_event
from ztna_ueba.tokenizer import DEFAULT_EXCLUDED_FIELDS


def _normal_records():
    return [
        {
            "dataset": "company",
            "source_type": "vpn",
            "event_type": "session",
            "event_time": f"2026-07-{day:02d}T09:00:00Z",
            "actor_alias": f"u{day:02d}",
            "auth_attempts": 1 if day % 3 else 2,
            "geo_zone": "seoul",
            "device_health": "healthy",
        }
        for day in range(1, 25)
    ]


def test_readable_explanation_uses_real_baseline_and_model_contribution():
    registry = BaselineRegistry.fit(_normal_records(), DEFAULT_EXCLUDED_FIELDS)
    event = {
        "dataset": "company",
        "source_type": "vpn",
        "event_type": "session",
        "event_time": "2026-07-25T03:00:00Z",
        "actor_alias": "u99",
        "auth_attempts": 17,
        "geo_zone": "unknown-region",
        "device_health": "compromised",
        "label": "attack",
        "attack_category": "credential_access",
    }
    prediction = {
        "model_version": "test",
        "trust_score": 0.0,
        "risk_score": 1.0,
        "raw_model_risk_probability": 0.99,
        "monotonic_baseline": True,
        "portable_mode": True,
        "model_dimension": 128,
        "field_context_layers": 2,
        "score_calibration_ready": True,
        "confidence": 0.95,
        "policy": {"stage": "deny", "action": "deny_or_isolate"},
        "top_fields": [
            {
                "event_index": 0,
                "field": "auth_attempts",
                "event_field_count": 2,
                "weight": 0.42,
                "event_weight": 1.0,
                "field_weight_within_event": 0.42,
                "field_importance_logit": 2.4,
                "field_weight_rank": 1,
                "request_field_count": 2,
                "risk_logit_contribution": 1.8,
            },
            {
                "event_index": 0,
                "field": "event_time",
                "event_field_count": 2,
                "weight": 0.25,
                "event_weight": 1.0,
                "field_weight_within_event": 0.25,
                "field_importance_logit": 1.9,
                "field_weight_rank": 2,
                "request_field_count": 2,
                "risk_logit_contribution": 0.8,
            },
        ],
    }

    explanation = build_operator_explanation(event, prediction, registry)
    first = explanation["risk_increasing_evidence"][0]
    second = explanation["risk_increasing_evidence"][1]

    assert first["reason_code"] == "NUMERIC_BASELINE_DEVIATION"
    assert first["observed_value"] == 17
    assert first["baseline"]["kind"] == "numeric"
    assert first["baseline"]["field_observations"] == 24
    assert first["baseline"]["robust_z"] > 1
    assert first["field_weight"] == 0.42
    assert first["model_risk_contribution"] == 1.8
    assert "정상 중앙값" in first["message"]
    assert "% 증가" in first["message"]
    assert "128차원" not in first["message"]
    assert "자기어텐션" not in first["message"]
    assert "AI 가중치는 42.00%(1위)였다" in first["message"]
    assert "기준선 이탈도는" in second["message"]
    assert "%" in second["message"]
    assert "단말는" not in explanation["readable_log"]["reason"]
    assert "정상 중앙값" in explanation["readable_log"]["reason"]
    assert "재검사하자" not in explanation["readable_log"]["reason"]
    assert "평소 행동 기준" in explanation["readable_log"]["usual_behavior"]
    assert "현재 인증 시도 횟수 값은 17" in explanation["readable_log"]["current_behavior"]
    assert explanation["readable_log"]["ueba_judgment"].startswith("UEBA는")
    assert "중요도 원점수" not in first["message"]
    assert "위험 기여도" not in first["message"]
    assert "차단" in explanation["readable_log"]["headline"]
    assert "정상 기준을 벗어난 의심 상황" in explanation["readable_log"]["decision"]
    assert "위험 점수 1.000, Trust Score 0.00" in explanation["readable_log"]["decision"]
    assert "보정 전 위험 확률" not in explanation["readable_log"]["decision"]
    assert explanation["readable_log"]["event_summary"] == explanation["readable_log"]["event"]
    assert explanation["readable_log"]["judgment_reason"] == explanation["readable_log"]["reason"]
    assert explanation["readable_log"]["policy_reason"] == explanation["readable_log"]["decision"]
    assert "u99" not in explanation["readable_log"]["full_text"]
    assert "auth_attempts" not in explanation["readable_log"]["full_text"]
    assert "단말와" not in explanation["readable_log"]["full_text"]
    assert explanation["audit"]["uses_generative_ai"] is False
    assert "label" not in sanitize_event(event)
    assert "attack_category" not in sanitize_event(event)


def test_secret_values_are_redacted_from_audit_snapshot():
    event = {"user": "u01", "access_token": "secret-value", "label": "normal"}
    sanitized = sanitize_event(event)
    assert sanitized == {"user": "u01", "access_token": "[REDACTED]"}


def test_service_account_backup_is_written_as_plain_korean():
    event = {
        "dataset": "company",
        "source_type": "vpn",
        "event_type": "scheduled_backup",
        "event_time": "2026-08-03T03:00:00Z",
        "actor_alias": "svc_backup",
        "job_type": "nightly_backup",
        "action": "backup",
    }
    prediction = {
        "trust_score": 99.9,
        "risk_score": 0.001,
        "raw_model_risk_probability": 0.001,
        "confidence": 0.9,
        "policy": {"stage": "allow", "action": "allow"},
        "top_fields": [],
    }
    explanation = build_operator_explanation(event, prediction, None)
    text = explanation["readable_log"]["full_text"]
    assert "백업 서비스 계정이 야간 정기 백업 작업을 수행했다" in text
    assert "svc_backup" not in text
    assert "nightly_backup" not in text
    assert "backup 행위" not in text
    assert "식별값는" not in text
    assert "작았다" not in text


def test_service_account_type_is_inferred_without_exposing_user_identifier():
    event = {
        "source_type": "ndr",
        "event_type": "network_flow",
        "user_id_hash": "user_889bd92091747447",
        "account_type": "service_account",
        "action": "http",
        "dst_ip": "149.171.126.17",
    }
    prediction = {
        "trust_score": 50.0,
        "risk_score": 0.5,
        "policy": {"stage": "monitor", "action": "observe_only"},
        "top_fields": [],
    }

    explanation = build_operator_explanation(event, prediction, None)

    assert explanation["readable_log"]["event_summary"].startswith("서비스 계정에서")
    assert "user_889bd92091747447" not in explanation["readable_log"]["full_text"]


def test_shadow_policy_reason_names_incomplete_operational_validation():
    event = {
        "source_type": "ndr",
        "event_type": "network_flow",
        "user_id_hash": "user_889bd92091747447",
    }
    prediction = {
        "trust_score": 0.0,
        "risk_score": 1.0,
        "policy": {"stage": "shadow", "action": "observe_only"},
        "top_fields": [],
    }

    explanation = build_operator_explanation(event, prediction, None)

    assert "정상 기준선·점수 보정·새 로그 검증" in explanation["readable_log"]["policy_reason"]
    assert "관찰 정책" in explanation["readable_log"]["policy_reason"]


def test_network_evidence_is_written_as_behavior_change_not_protocol_suspicion():
    normals = [
        {
            "dataset": "company",
            "source_type": "ndr",
            "event_type": "network_flow",
            "actor_alias": f"u{index:02d}",
            "src_ip": f"10.0.0.{index}",
            "device_id": f"device-{index}",
            "dst_ip": f"192.0.2.{index}",
        }
        for index in range(1, 25)
    ]
    registry = BaselineRegistry.fit(normals, DEFAULT_EXCLUDED_FIELDS)
    event = {
        "dataset": "company",
        "source_type": "ndr",
        "event_type": "network_flow",
        "actor_alias": "svc_new",
        "src_ip": "203.0.113.10",
        "device_id": "device-new",
        "dst_ip": "198.51.100.20",
        "action": "http",
    }
    prediction = {
        "trust_score": 0.0,
        "risk_score": 1.0,
        "policy": {"stage": "deny", "action": "deny_or_isolate"},
        "top_fields": [
            {
                "event_index": 0,
                "field": field,
                "weight": weight,
                "field_weight_rank": rank,
                "risk_logit_contribution": contribution,
            }
            for rank, (field, weight, contribution) in enumerate(
                (
                    ("src_ip", 0.40, 2.4),
                    ("device_id", 0.30, 1.8),
                    ("dst_ip", 0.20, 1.2),
                ),
                start=1,
            )
        ],
    }

    explanation = build_operator_explanation(event, prediction, registry)
    reason = explanation["readable_log"]["judgment_reason"]

    assert "같은 유형의 정상 로그 24건" in reason
    assert "기존 값 안에서 반복됐고" in explanation["readable_log"]["usual_behavior"]
    assert "이번 요청의 값은 모두 처음 나타났다" in explanation["readable_log"]["usual_behavior"]
    assert "비교 기준에 없던 단말과 출발지 IP" in explanation["readable_log"]["current_behavior"]
    assert "접속 기록이 없던 목적지 IP" in explanation["readable_log"]["current_behavior"]
    assert "동시에 새로운 값으로 바뀐 것" in explanation["readable_log"]["ueba_judgment"]
    assert "평소에 없던" not in explanation["readable_log"]["ueba_judgment"]
    assert "재검사하자" not in reason
    assert "HTTP 통신" not in reason


def test_operator_evidence_names_actor_baseline_and_selection_reason():
    normals = [
        {
            "dataset": "company",
            "source_type": "vpn",
            "event_type": "scheduled_backup",
            "actor_alias": "svc_backup",
            "workload_role": "backup_agent",
            "download_mb": 50000 + index,
        }
        for index in range(25)
    ]
    registry = BaselineRegistry.fit(normals, DEFAULT_EXCLUDED_FIELDS)
    event = {**normals[0], "download_mb": 90000}
    prediction = {
        "trust_score": 10.0,
        "risk_score": 0.9,
        "raw_model_risk_probability": 0.9,
        "monotonic_baseline": True,
        "portable_mode": True,
        "policy": {"stage": "restrict", "action": "restrict"},
        "ueba_baseline": {
            "mode": "hierarchical_field_level",
            "field_scope_counts": {"actor": 3},
        },
        "top_fields": [
            {
                "event_index": 0,
                "field": "download_mb",
                "weight": 0.5,
                "event_weight": 1.0,
                "field_weight_within_event": 0.5,
                "field_weight_rank": 1,
                "request_field_count": 3,
                "risk_logit_contribution": 1.0,
            }
        ],
    }

    explanation = build_operator_explanation(event, prediction, registry)
    evidence = explanation["risk_increasing_evidence"][0]

    assert evidence["baseline"]["baseline_scope"] == "actor"
    assert evidence["baseline"]["profile_records"] == 25
    assert "사용자·서비스 계정 기준선" in evidence["message"]
    assert "정상 로그 25건" in evidence["baseline"]["baseline_selection_reason"]
    assert evidence["selection_basis"]["metric"] == (
        "absolute_model_risk_contribution"
    )
    assert "1위" in evidence["selection_basis"]["reason"]
    assert "평소 백업 서비스 계정의 정상 로그 25건" in explanation["readable_log"]["usual_behavior"]
    assert "다운로드 용량의 정상 중앙값" in explanation["readable_log"]["usual_behavior"]
    assert "현재 다운로드 용량 값은 90,000" in explanation["readable_log"]["current_behavior"]
    assert "평소 행동 범위를 크게 벗어난 점" in explanation["readable_log"]["ueba_judgment"]
