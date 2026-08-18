"""Coverage-aware deterministic Trust Score baseline.

Jeong and Yang (2025) define twenty 0/5/10/15/20 sub-metrics, four fixed
factor weights (B/N/D/T), and three access bands.  Their paper assumes that
the required PIP values are available.  This project adaptation never fills
an absent value with a favourable normal score: it evaluates only metrics
applicable to the source profile, penalises missing required profile fields,
and renormalises the active factor weights.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Iterable


PAPER_FACTOR_WEIGHTS = {"B": 0.4, "N": 0.3, "D": 0.2, "T": 0.1}
PAPER_SCORE_LEVELS = (0.0, 5.0, 10.0, 15.0, 20.0)
FALLBACK_TRUST_SCORE = 70.0


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


def _direct_metric(
    event: dict[str, Any],
    factor: str,
    field: str,
    name: str,
    scorer,
    basis_prefix: str,
) -> dict[str, Any] | None:
    if field not in event or event[field] in {None, ""}:
        return None
    value = event[field]
    return _metric(factor, name, scorer(value), value, basis_prefix, "direct")


def _missing_required(factor: str, name: str, fields: str, basis: str) -> dict[str, Any]:
    return _metric(
        factor,
        name,
        0.0,
        f"missing required field(s): {fields}",
        basis,
        "missing_required",
    )


def _score_ndr_traffic_conformance(event: dict[str, Any]) -> float:
    """Coarse fixed signature for the paper's anomalous-traffic sub-metric."""

    protocol = str(event.get("protocol", "")).strip().lower()
    service = str(event.get("service", event.get("action", ""))).strip().lower()
    destination = event.get("dst_port")
    try:
        destination_port = int(float(destination))
    except (TypeError, ValueError):
        destination_port = None

    standard_protocols = {"6", "17", "tcp", "udp"}
    registered_ports = {
        "dns": {53},
        "ftp": {21},
        "http": {80},
        "https": {443},
        "pop3": {110},
        "smtp": {25},
        "ssh": {22},
    }
    if protocol not in standard_protocols:
        return 10.0
    if service in registered_ports and destination_port in registered_ports[service]:
        return 20.0
    if service in {"", "-", "nan", "none"} or destination_port is None:
        return 15.0
    return 10.0


def _score_ndr_connection_state(value: Any) -> float:
    state = str(value).strip().upper()
    if state in {"CON", "ESTABLISHED", "FIN", "SF"}:
        return 20.0
    if state in {"INT", "REQ"}:
        return 15.0
    if state in {"RST", "RSTO", "RSTR"}:
        return 10.0
    if state in {"CLO", "ECO", "PAR", "URN"}:
        return 5.0
    return 10.0


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
    """Evaluate one event without favourable missing-value imputation.

    Source-specific adapters expose only the paper sub-metrics that the log can
    support.  A missing required adapter field scores zero; structurally
    inapplicable factors are excluded and the remaining factor weights are
    renormalised.  If no metric can be mapped at all, the midpoint of the MFA
    band is returned as an explicit fail-safe policy fallback.
    """

    familiar = {str(value).lower() for value in familiar_geo_zones}
    domestic = {str(value).lower() for value in domestic_geo_zones}
    geo_value = event.get("geo_zone")
    geo_zone = str(geo_value).lower() if geo_value not in {None, ""} else None
    is_familiar_geo = bool(geo_zone in familiar) if geo_zone is not None else False
    is_domestic_geo = bool(geo_zone in domestic) if geo_zone is not None else False
    source_type = str(event.get("source_type", "")).strip().lower()
    event_type = str(event.get("event_type", "")).strip().lower()
    is_ndr = source_type == "ndr" or event_type == "network_flow"
    has_behavior_snapshot = any(
        event.get(field) not in {None, ""}
        for field in (
            "auth_attempts",
            "failed_login_attempts",
            "geo_zone",
            "download_mb",
        )
    )
    is_access_event = (
        source_type in {"auth", "iam", "vpn", "ztna"}
        or event_type in {"authentication", "scheduled_backup", "session"}
        or has_behavior_snapshot
    )

    metrics: list[dict[str, Any]] = []

    if is_ndr and event.get("network_traffic_status") in {None, ""}:
        protocol_present = event.get("protocol") not in {None, ""}
        if protocol_present:
            metrics.append(
                _metric(
                    "N",
                    "anomalous_network_traffic",
                    _score_ndr_traffic_conformance(event),
                    {
                        "protocol": event.get("protocol"),
                        "service": event.get("service", event.get("action")),
                        "dst_port": event.get("dst_port"),
                    },
                    "paper Table 3 qualitative traffic level; project fixed protocol/service/port proxy",
                    "proxy",
                )
            )
        else:
            metrics.append(
                _missing_required(
                    "N",
                    "anomalous_network_traffic",
                    "protocol",
                    "NDR profile requires protocol for the fixed traffic proxy",
                )
            )
        if event.get("latency_ms") in {None, ""}:
            if event.get("state") not in {None, ""}:
                metrics.append(
                    _metric(
                        "N",
                        "network_stability",
                        _score_ndr_connection_state(event["state"]),
                        event["state"],
                        "paper Table 3 qualitative connection stability; project fixed state proxy",
                        "proxy",
                    )
                )
            else:
                metrics.append(
                    _missing_required(
                        "N",
                        "network_stability",
                        "state",
                        "NDR profile requires connection state for the fixed stability proxy",
                    )
                )

    if is_access_event and event.get("auth_attempts") not in {None, ""}:
        auth_attempts = int(event["auth_attempts"])
        metrics.append(
            _metric(
                "B",
                "login_frequency",
                _score_login_frequency(auth_attempts),
                auth_attempts,
                "paper Table 2 login-frequency bands",
                "direct",
            )
        )

    failed_login_metric = _direct_metric(
        event,
        "B",
        "failed_login_attempts",
        "failed_login_attempts",
        lambda value: _score_failed_logins(int(value)),
        "paper Table 2 failed-login bands",
    )
    if is_access_event and failed_login_metric is not None:
        metrics.append(failed_login_metric)

    if is_access_event and event.get("event_time"):
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
    if is_access_event and geo_zone is not None:
        behavior_location_score = 20.0 if is_familiar_geo else 15.0 if is_domestic_geo else 10.0
        metrics.append(
            _metric(
                "B",
                "new_location_login",
                behavior_location_score,
                geo_zone,
                "paper Table 2: familiar / different domestic region / new country",
                "proxy",
            )
        )
    download_metric = _direct_metric(
        event,
        "B",
        "download_mb",
        "large_download",
        lambda value: _score_download_mb(float(value)),
        "paper Table 2 download-volume bands",
    )
    if is_access_event and download_metric is not None:
        metrics.append(download_metric)

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
    ip_metric = _direct_metric(
        event,
        "N",
        "ip_reputation",
        "ip_reputation",
        lambda value: _categorical_score(value, ip_mapping),
        "paper Table 3 IP-reputation levels",
    )
    if ip_metric is not None:
        metrics.append(ip_metric)
    vpn_metric = _direct_metric(
        event,
        "N",
        "vpn_approval_status",
        "vpn_usage",
        lambda value: _categorical_score(
            value,
            {"approved": 20.0, "managed": 20.0, "unknown": 10.0, "unapproved": 0.0},
        ),
        "paper Table 3 VPN-use levels; explicit approval status only",
    )
    if vpn_metric is not None:
        metrics.append(vpn_metric)
    traffic_metric = _direct_metric(
        event,
        "N",
        "network_traffic_status",
        "anomalous_network_traffic",
        lambda value: _categorical_score(value, traffic_mapping),
        "paper Table 3 anomalous-traffic levels",
    )
    if traffic_metric is not None:
        metrics.append(traffic_metric)
    if geo_zone is not None:
        metrics.append(
            _metric(
                "N",
                "access_location_reliability",
                20.0 if is_familiar_geo else 10.0 if is_domestic_geo else 5.0,
                geo_zone,
                "paper Table 3: registered / new-low-risk / new-medium-risk location",
                "proxy",
            )
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
    device_health = event.get("device_health")
    if device_health not in {None, ""}:
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
    active_factors = {
        factor: [row for row in metrics if row["factor"] == factor]
        for factor in PAPER_FACTOR_WEIGHTS
    }
    active_factors = {factor: rows for factor, rows in active_factors.items() if rows}
    factor_scores = {
        factor: 100.0 * sum(row["score"] for row in rows) / (20.0 * len(rows))
        for factor, rows in active_factors.items()
    }
    active_weight = sum(PAPER_FACTOR_WEIGHTS[factor] for factor in active_factors)
    fallback_used = not active_factors
    if fallback_used:
        normalized_factor_weights: dict[str, float] = {}
        factor_contributions: dict[str, float] = {}
        trust_score = FALLBACK_TRUST_SCORE
    else:
        normalized_factor_weights = {
            factor: PAPER_FACTOR_WEIGHTS[factor] / active_weight for factor in active_factors
        }
        factor_contributions = {
            factor: normalized_factor_weights[factor] * factor_scores[factor]
            for factor in active_factors
        }
        trust_score = sum(factor_contributions.values())
    source_counts = Counter(row["evidence_source"] for row in metrics)
    observed_count = len(metrics) - source_counts["missing_required"]

    return {
        "method": "coverage-aware deterministic rule baseline adapted from Jeong and Yang (2025)",
        "paper_factor_weights": dict(PAPER_FACTOR_WEIGHTS),
        "normalized_active_factor_weights": normalized_factor_weights,
        "factor_scores": factor_scores,
        "factor_contributions": factor_contributions,
        "trust_score": trust_score,
        "risk_score": 1.0 - trust_score / 100.0,
        "policy": map_paper_trust_to_policy(trust_score),
        "coverage": {
            "observed_or_project_mapped_metrics": observed_count,
            "total_metrics": len(metrics),
            "paper_total_metrics": 20,
            "ratio": observed_count / len(metrics) if metrics else 0.0,
            "evidence_source_counts": dict(sorted(source_counts.items())),
        },
        "metrics": metrics,
        "fallback_used": fallback_used,
        "limitations": [
            "The paper assumes complete PIP inputs and does not define missing-value handling.",
            "This adaptation excludes structurally inapplicable metrics, scores required profile omissions as zero, and renormalises active factor weights.",
            "NDR protocol/service/port, connection-state, latency, location, and aggregate device-health mappings are documented project proxies.",
            "No absent metric receives a favourable normal score.",
        ],
    }
