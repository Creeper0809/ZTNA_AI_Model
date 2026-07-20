"""Deterministic Trust Score baseline adapted from Jeong and Yang (2025).

The paper defines twenty 0/5/10/15/20 sub-metrics, four fixed factor
weights (B/N/D/T), and three access bands.  This module preserves that
structure and makes every project-specific proxy or missing-field assumption
explicit in the returned explanation.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Iterable


PAPER_FACTOR_WEIGHTS = {"B": 0.4, "N": 0.3, "D": 0.2, "T": 0.1}
PAPER_SCORE_LEVELS = (0.0, 5.0, 10.0, 15.0, 20.0)


def map_paper_trust_to_policy(trust_score: float) -> dict[str, Any]:
    """Map the paper's 0--100 Trust Score to its three policy bands."""

    score = max(0.0, min(100.0, float(trust_score)))
    if score >= 80.0:
        stage, action = "allow", "allow"
    elif score >= 60.0:
        stage, action = "mfa", "require_additional_authentication"
    else:
        stage, action = "block", "block"
    return {
        "stage": stage,
        "action": action,
        "thresholds": {"allow": ">=80", "mfa": "60-79", "block": "<60"},
    }


def _metric(
    factor: str,
    name: str,
    score: float,
    observed: Any,
    basis: str,
    evidence_source: str,
) -> dict[str, Any]:
    if float(score) not in PAPER_SCORE_LEVELS:
        raise ValueError(f"{name} score must be one of {PAPER_SCORE_LEVELS}: {score}")
    return {
        "factor": factor,
        "metric": name,
        "score": float(score),
        "observed": observed,
        "basis": basis,
        "evidence_source": evidence_source,
    }


def _score_login_frequency(value: int) -> float:
    if value <= 5:
        return 20.0
    if value <= 10:
        return 15.0
    if value <= 15:
        return 10.0
    if value < 30:
        return 5.0
    return 0.0


def _score_failed_logins(value: int) -> float:
    if value <= 2:
        return 20.0
    if value <= 5:
        return 15.0
    if value <= 10:
        return 10.0
    if value < 50:
        return 5.0
    return 0.0


def _event_hour(value: str) -> int:
    timestamp = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(timestamp).hour


def _score_login_hour(hour: int) -> float:
    if 9 <= hour < 18:
        return 20.0
    if 18 <= hour < 22:
        return 15.0
    if hour >= 22 or hour < 2:
        return 10.0
    if 2 <= hour < 5:
        return 0.0
    return 10.0


def _score_download_mb(value: float) -> float:
    if value <= 500:
        return 20.0
    if value <= 5_000:
        return 15.0
    if value <= 10_000:
        return 10.0
    if value < 50_000:
        return 5.0
    return 0.0


def _score_latency_proxy(value: float) -> float:
    """Project proxy for the paper's qualitative network-stability levels."""

    if value <= 150:
        return 20.0
    if value <= 500:
        return 15.0
    if value <= 1_000:
        return 10.0
    if value <= 2_000:
        return 5.0
    return 0.0


def _categorical_score(value: Any, mapping: dict[str, float], default: float = 10.0) -> float:
    return mapping.get(str(value).strip().lower(), default)


def _direct_or_assumed(
    event: dict[str, Any],
    factor: str,
    field: str,
    name: str,
    scorer,
    normal_observed: str,
    basis_prefix: str,
) -> dict[str, Any]:
    if field not in event or event[field] in {None, ""}:
        return _metric(
            factor,
            name,
            20.0,
            normal_observed,
            f"{basis_prefix}; field absent, favorable normal assumption",
            "assumed_normal",
        )
    value = event[field]
    return _metric(factor, name, scorer(value), value, basis_prefix, "direct")


def evaluate_paper_rule_baseline(
    event: dict[str, Any],
    *,
    familiar_geo_zones: Iterable[str] = ("seoul",),
    domestic_geo_zones: Iterable[str] = (
        "seoul",
        "busan",
        "incheon",
        "daegu",
        "daejeon",
        "gwangju",
        "ulsan",
        "jeju",
    ),
) -> dict[str, Any]:
    """Evaluate one event with the paper-derived deterministic scorecard.

    Missing sub-metrics receive a favorable 20/20 instead of being silently
    treated as risky.  Aggregate ``device_health`` is an explicit project
    mapping because the demo event does not expose the paper's five posture
    fields independently.
    """

    familiar = {str(value).lower() for value in familiar_geo_zones}
    domestic = {str(value).lower() for value in domestic_geo_zones}
    geo_zone = str(event.get("geo_zone", "unknown")).lower()
    is_familiar_geo = geo_zone in familiar
    is_domestic_geo = geo_zone in domestic

    auth_attempts = int(event.get("auth_attempts", 1))
    metrics: list[dict[str, Any]] = [
        _metric(
            "B",
            "login_frequency",
            _score_login_frequency(auth_attempts),
            auth_attempts,
            "paper Table 2 login-frequency bands",
            "direct",
        ),
        _direct_or_assumed(
            event,
            "B",
            "failed_login_attempts",
            "failed_login_attempts",
            lambda value: _score_failed_logins(int(value)),
            "not observed; assumed 0",
            "paper Table 2 failed-login bands",
        ),
    ]

    if event.get("event_time"):
        hour = _event_hour(str(event["event_time"]))
        metrics.append(
            _metric(
                "B",
                "off_hours_login",
                _score_login_hour(hour),
                f"{hour:02d}:00",
                "paper Table 2 off-hours bands",
                "direct",
            )
        )
    else:
        metrics.append(
            _metric(
                "B",
                "off_hours_login",
                20.0,
                "not observed; assumed business hours",
                "paper Table 2; field absent, favorable normal assumption",
                "assumed_normal",
            )
        )

    behavior_location_score = 20.0 if is_familiar_geo else 15.0 if is_domestic_geo else 10.0
    metrics.extend(
        [
            _metric(
                "B",
                "new_location_login",
                behavior_location_score,
                geo_zone,
                "paper Table 2: familiar / different domestic region / new country",
                "proxy",
            ),
            _direct_or_assumed(
                event,
                "B",
                "download_mb",
                "large_download",
                lambda value: _score_download_mb(float(value)),
                "not observed; assumed <=500 MB",
                "paper Table 2 download-volume bands",
            ),
        ]
    )

    ip_mapping = {
        "highly_trusted": 20.0,
        "typical": 15.0,
        "warning": 10.0,
        "blacklisted": 5.0,
        "malicious": 0.0,
    }
    traffic_mapping = {
        "normal": 20.0,
        "minor": 15.0,
        "increased": 10.0,
        "abnormal": 5.0,
        "malicious": 0.0,
    }
    metrics.extend(
        [
            _direct_or_assumed(
                event,
                "N",
                "ip_reputation",
                "ip_reputation",
                lambda value: _categorical_score(value, ip_mapping),
                "not observed; assumed highly trusted",
                "paper Table 3 IP-reputation levels",
            ),
            _metric(
                "N",
                "vpn_usage",
                20.0,
                event.get("source_type", "not observed"),
                "company_demo VPN treated as enterprise-approved; explicit favorable assumption",
                "assumed_normal",
            ),
            _direct_or_assumed(
                event,
                "N",
                "network_traffic_status",
                "anomalous_network_traffic",
                lambda value: _categorical_score(value, traffic_mapping),
                "not observed; assumed normal",
                "paper Table 3 anomalous-traffic levels",
            ),
            _metric(
                "N",
                "access_location_reliability",
                20.0 if is_familiar_geo else 10.0 if is_domestic_geo else 5.0,
                geo_zone,
                "paper Table 3: registered / new-low-risk / new-medium-risk location",
                "proxy",
            ),
        ]
    )

    if "latency_ms" in event and event["latency_ms"] not in {None, ""}:
        latency_ms = float(event["latency_ms"])
        metrics.append(
            _metric(
                "N",
                "network_stability",
                _score_latency_proxy(latency_ms),
                latency_ms,
                "project latency proxy for paper Table 3 connection-stability levels",
                "proxy",
            )
        )
    else:
        metrics.append(
            _metric(
                "N",
                "network_stability",
                20.0,
                "not observed; assumed stable",
                "paper Table 3; field absent, favorable normal assumption",
                "assumed_normal",
            )
        )

    device_health = event.get("device_health")
    if device_health in {None, ""}:
        device_score = 20.0
        device_source = "assumed_normal"
        device_observed = "not observed; assumed healthy"
    else:
        device_score = _categorical_score(
            device_health,
            {"healthy": 20.0, "degraded": 10.0, "compromised": 0.0},
            default=10.0,
        )
        device_source = "inferred_aggregate"
        device_observed = device_health
    for name in (
        "security_patch_status",
        "antivirus_status",
        "device_authentication",
        "device_integrity",
        "device_encryption",
    ):
        metrics.append(
            _metric(
                "D",
                name,
                device_score,
                device_observed,
                "aggregate device_health projected to paper Table 4 band",
                device_source,
            )
        )

    threat_metrics = (
        "threat_history",
        "security_policy_violation_history",
        "account_compromise_history",
        "policy_violation_frequency",
        "response_to_past_threats",
    )
    for name in threat_metrics:
        field_value = event.get(name)
        if isinstance(field_value, (int, float)) and float(field_value) in PAPER_SCORE_LEVELS:
            metrics.append(
                _metric("T", name, float(field_value), field_value, "explicit paper Table 5 score", "direct")
            )
        else:
            metrics.append(
                _metric(
                    "T",
                    name,
                    20.0,
                    "not observed; assumed no adverse history",
                    "paper Table 5; field absent, favorable normal assumption",
                    "assumed_normal",
                )
            )

    factor_scores = {
        factor: sum(row["score"] for row in metrics if row["factor"] == factor)
        for factor in PAPER_FACTOR_WEIGHTS
    }
    factor_contributions = {
        factor: PAPER_FACTOR_WEIGHTS[factor] * factor_scores[factor]
        for factor in PAPER_FACTOR_WEIGHTS
    }
    trust_score = sum(factor_contributions.values())
    source_counts = Counter(row["evidence_source"] for row in metrics)
    observed_count = len(metrics) - source_counts["assumed_normal"]

    return {
        "method": "Jeong and Yang (2025) 20-sub-metric deterministic Trust Score baseline",
        "paper_factor_weights": dict(PAPER_FACTOR_WEIGHTS),
        "factor_scores": factor_scores,
        "factor_contributions": factor_contributions,
        "trust_score": trust_score,
        "risk_score": 1.0 - trust_score / 100.0,
        "policy": map_paper_trust_to_policy(trust_score),
        "coverage": {
            "observed_or_project_mapped_metrics": observed_count,
            "total_metrics": len(metrics),
            "ratio": observed_count / len(metrics),
            "evidence_source_counts": dict(sorted(source_counts.items())),
        },
        "metrics": metrics,
        "limitations": [
            "The paper defines the scorecard; current VPN fields do not directly expose all twenty sub-metrics.",
            "Missing metrics receive favorable normal scores and aggregate device_health is a documented project mapping.",
            "The latency-to-stability mapping is a project proxy, not a threshold specified by the paper.",
        ],
    }
