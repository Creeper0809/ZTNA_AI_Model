import pytest

from ztna_ueba.rule_baseline import (
    evaluate_paper_rule_baseline,
    map_paper_trust_to_policy,
)


def test_paper_policy_boundaries():
    assert map_paper_trust_to_policy(80.0)["stage"] == "allow"
    assert map_paper_trust_to_policy(79.999)["stage"] == "mfa"
    assert map_paper_trust_to_policy(60.0)["stage"] == "mfa"
    assert map_paper_trust_to_policy(59.999)["stage"] == "block"


def test_normal_vpn_event_is_allowed():
    result = evaluate_paper_rule_baseline(
        {
            "dataset": "company_demo",
            "source_type": "vpn",
            "event_type": "session",
            "event_time": "2026-07-20T09:00:00Z",
            "device_health": "healthy",
            "auth_attempts": 1,
            "latency_ms": 101,
            "geo_zone": "seoul",
        }
    )
    assert result["factor_scores"] == {"B": 100.0, "N": 100.0, "D": 100.0}
    assert result["trust_score"] == 100.0
    assert result["policy"]["stage"] == "allow"
    assert "assumed_normal" not in result["coverage"]["evidence_source_counts"]


def test_composite_event_uses_only_observed_metrics():
    result = evaluate_paper_rule_baseline(
        {
            "dataset": "company_demo",
            "source_type": "vpn",
            "event_type": "session",
            "event_time": "2026-07-20T03:00:00Z",
            "device_health": "degraded",
            "auth_attempts": 4,
            "latency_ms": 450,
            "geo_zone": "busan",
        }
    )
    assert result["factor_scores"] == pytest.approx(
        {"B": 58.333333, "N": 62.5, "D": 50.0}
    )
    assert result["trust_score"] == pytest.approx(57.870370)
    assert result["policy"]["stage"] == "block"
    assert result["coverage"]["total_metrics"] == 10
    assert result["coverage"]["ratio"] == 1.0


def test_rule_allow_model_deny_example_has_complete_ndr_rule_coverage():
    result = evaluate_paper_rule_baseline(
        {
            "dataset": "unsw_nb15",
            "source_type": "ndr",
            "event_type": "network_flow",
            "protocol": "tcp",
            "service": "http",
            "state": "FIN",
            "dst_port": 80,
            "event_time": "2015-02-18T10:40:29Z",
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
    )
    assert result["factor_scores"] == {"B": 100.0, "N": 100.0, "D": 100.0, "T": 100.0}
    assert result["trust_score"] == 100.0
    assert result["policy"]["stage"] == "allow"
    assert result["coverage"]["observed_or_project_mapped_metrics"] == 20
    assert result["coverage"]["ratio"] == 1.0


def test_approved_backup_remains_mfa_without_normal_assumptions():
    result = evaluate_paper_rule_baseline(
        {
            "dataset": "company_ops",
            "source_type": "vpn",
            "event_type": "scheduled_backup",
            "event_time": "2026-07-20T03:00:00Z",
            "actor_alias": "svc_backup",
            "device_health": "healthy",
            "auth_attempts": 30,
            "failed_login_attempts": 0,
            "latency_ms": 100,
            "geo_zone": "seoul",
            "network_traffic_status": "normal",
            "download_mb": 50_000,
            "ip_reputation": "highly_trusted",
            "vpn_approval_status": "approved",
            "threat_history": 20,
            "security_policy_violation_history": 20,
            "account_compromise_history": 20,
            "policy_violation_frequency": 20,
            "response_to_past_threats": 20,
        }
    )
    assert result["factor_scores"] == {"B": 40.0, "N": 100.0, "D": 100.0, "T": 100.0}
    assert result["trust_score"] == pytest.approx(76.0)
    assert result["policy"]["stage"] == "mfa"
    assert result["coverage"]["observed_or_project_mapped_metrics"] == 20
    assert result["coverage"]["ratio"] == 1.0


def test_missing_ndr_state_is_penalized_not_assumed_normal():
    result = evaluate_paper_rule_baseline(
        {
            "source_type": "ndr",
            "event_type": "network_flow",
            "protocol": "17",
            "action": "connect",
            "dst_port": 43322,
        }
    )
    assert result["factor_scores"] == {"N": 25.0}
    assert result["policy"]["stage"] == "block"
    assert result["coverage"]["ratio"] == 0.5
    assert result["coverage"]["evidence_source_counts"]["missing_required"] == 1
