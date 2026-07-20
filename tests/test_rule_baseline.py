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
    assert result["factor_scores"] == {"B": 100.0, "N": 100.0, "D": 100.0, "T": 100.0}
    assert result["trust_score"] == 100.0
    assert result["policy"]["stage"] == "allow"


def test_composite_event_accumulates_risk_and_requires_mfa():
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
    assert result["factor_scores"] == {"B": 75.0, "N": 85.0, "D": 50.0, "T": 100.0}
    assert result["factor_contributions"] == {"B": 30.0, "N": 25.5, "D": 10.0, "T": 10.0}
    assert result["trust_score"] == 75.5
    assert result["risk_score"] == 0.245
    assert result["policy"]["stage"] == "mfa"
    assert result["coverage"]["total_metrics"] == 20
    assert result["coverage"]["ratio"] == 0.50


def test_rule_allow_model_deny_example_has_rule_trust_98_5():
    result = evaluate_paper_rule_baseline(
        {
            "dataset": "company_demo",
            "source_type": "vpn",
            "event_type": "session",
            "event_time": "2026-07-20T09:00:00Z",
            "device_health": "healthy",
            "auth_attempts": 4,
            "latency_ms": 450,
            "geo_zone": "seoul",
        }
    )
    assert result["factor_scores"] == {"B": 100.0, "N": 95.0, "D": 100.0, "T": 100.0}
    assert result["trust_score"] == 98.5
    assert result["policy"]["stage"] == "allow"


def test_rule_mfa_model_allow_example_has_rule_trust_79_5():
    result = evaluate_paper_rule_baseline(
        {
            "dataset": "company_demo",
            "source_type": "vpn",
            "event_type": "session",
            "event_time": "2026-07-20T03:00:00Z",
            "device_health": "healthy",
            "auth_attempts": 30,
            "latency_ms": 100,
            "geo_zone": "seoul",
            "network_traffic_status": "abnormal",
        }
    )
    assert result["factor_scores"] == {"B": 60.0, "N": 85.0, "D": 100.0, "T": 100.0}
    assert result["trust_score"] == 79.5
    assert result["policy"]["stage"] == "mfa"
