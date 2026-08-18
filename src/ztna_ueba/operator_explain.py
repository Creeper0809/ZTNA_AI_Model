"""Build deterministic, human-readable evidence from model contributions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from typing import Mapping, Sequence

from .baseline import BaselineRegistry, parse_number, parse_time_bucket, value_fingerprint


GROUND_TRUTH_FIELDS = frozenset(
    {
        "label",
        "is_attack",
        "attack_category",
        "raw_label",
        "result",
        "source_file",
        "split",
        "_sample_hash",
        "_group_id",
        "target",
        "ground_truth",
        "y",
    }
)

ACTOR_FIELDS = (
    "actor_id",
    "actor_alias",
    "user_id_hash",
    "user_id",
    "user",
    "username",
    "subject",
    "account",
    "principal",
    "tenant_actor",
)

TARGET_FIELDS = (
    "resource",
    "resource_id",
    "application",
    "dst_host",
    "destination_host",
    "dst_ip",
    "service",
    "job_type",
    "workload_role",
    "asset",
    "object",
)

ACTION_FIELDS = ("action", "operation", "activity", "method", "event_type")

LINK_FIELDS = (
    "session_id",
    "session_id_hash",
    "device_id",
    "device_id_hash",
    "host",
    "hostname",
    "src_ip",
    "src_ip_hash",
    "dst_ip",
    "resource",
    "resource_id",
    "application",
)

FIELD_LABELS_KO = {
    "actor_alias": "사용자",
    "user_id_hash": "사용자 식별값",
    "account_type": "계정 유형",
    "credential_id_hash": "인증 수단 식별값",
    "credential_type": "인증 수단 종류",
    "identity_provider": "인증 제공자",
    "authentication_method": "인증 방식",
    "mfa_result": "MFA 결과",
    "authentication_result": "인증 결과",
    "session_id_hash": "세션 식별값",
    "auth_attempts": "인증 시도 횟수",
    "failed_login_attempts": "실패한 로그인 횟수",
    "event_time": "발생 시간",
    "device_health": "단말 상태",
    "device_id": "단말",
    "device_id_hash": "단말",
    "src_ip": "출발지 IP",
    "src_ip_hash": "출발지 IP",
    "dst_ip": "목적지 IP",
    "src_port": "출발지 포트",
    "dst_port": "목적지 포트",
    "protocol": "통신 프로토콜",
    "service": "서비스",
    "latency_ms": "통신 지연",
    "download_mb": "다운로드 용량",
    "geo_zone": "접속 지역",
    "network_traffic_status": "네트워크 상태",
    "maintenance_window": "승인 작업 시간",
    "job_type": "업무 종류",
    "threat_history": "과거 위협 이력",
}

POLICY_LABELS_KO = {
    "shadow": "관찰",
    "allow": "허용",
    "monitor": "허용 후 관찰",
    "step_up": "추가 인증",
    "restrict": "접근 제한",
    "deny": "차단",
}

SOURCE_LABELS_KO = {
    "vpn": "VPN",
    "iam": "인증 시스템",
    "auth": "인증 시스템",
    "ndr": "네트워크 탐지",
    "edr": "단말 보안",
    "soc": "보안 관제",
}

EVENT_LABELS_KO = {
    "session": "접속 세션",
    "authentication": "인증",
    "network_flow": "네트워크 통신",
    "scheduled_backup": "정기 백업",
    "file_access": "파일 접근",
    "process_execution": "프로세스 실행",
}

VALUE_LABELS_KO = {
    "nightly_backup": "야간 정기 백업",
    "backup": "백업",
    "backup_agent": "백업 서비스",
    "healthy": "정상",
    "degraded": "성능 저하",
    "compromised": "침해 의심",
    "approved": "승인됨",
    "normal": "정상",
    "abnormal": "비정상",
    "unknown-region": "확인되지 않은 지역",
    "seoul": "서울",
    "busan": "부산",
    "http": "HTTP",
    "https": "HTTPS",
    "tcp": "TCP",
    "udp": "UDP",
    "fin": "연결 종료",
}

SENSITIVE_FIELD_TOKENS = (
    "password",
    "secret",
    "access_token",
    "refresh_token",
    "session_token",
    "credential_value",
    "private_key",
)


def _is_sensitive_field(field_name: str) -> bool:
    normalized = field_name.lower()
    return any(token in normalized for token in SENSITIVE_FIELD_TOKENS)


def _normalized_record(record: Mapping[str, object]) -> dict[str, object]:
    return {str(key).strip().lower(): value for key, value in record.items()}


def sanitize_event(record: Mapping[str, object]) -> dict[str, object]:
    """Remove labels and secrets while retaining an auditable event snapshot."""

    sanitized: dict[str, object] = {}
    for raw_name, value in record.items():
        name = str(raw_name).strip()
        normalized = name.lower()
        if normalized in GROUND_TRUTH_FIELDS:
            continue
        if _is_sensitive_field(normalized):
            sanitized[name] = "[REDACTED]"
        else:
            sanitized[name] = value
    return sanitized


def extract_actor_id(record: Mapping[str, object]) -> str:
    normalized = _normalized_record(record)
    for field in ACTOR_FIELDS:
        value = normalized.get(field)
        if value not in {None, ""}:
            return str(value)
    return "unknown-actor"


def stable_event_id(record: Mapping[str, object]) -> str:
    normalized = _normalized_record(record)
    for field in ("event_id", "request_id", "trace_id"):
        value = normalized.get(field)
        if value not in {None, ""}:
            return str(value)
    canonical = json.dumps(
        sanitize_event(record), ensure_ascii=False, sort_keys=True, default=str
    ).encode("utf-8")
    digest = hashlib.blake2b(canonical, digest_size=10, person=b"ztna-event").hexdigest()
    return f"evt-{digest}"


def normalize_event_time(value: object) -> str:
    text = str(value or "").strip()
    if text:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _display_value(field_name: str, value: object) -> object:
    normalized = field_name.lower()
    if _is_sensitive_field(normalized):
        return "[REDACTED]"
    if value is None:
        return None
    text = str(value)
    if "hash" in normalized and len(text) > 18:
        return f"{text[:9]}…{text[-5:]}"
    return value


def _field_label(field_name: str) -> str:
    normalized = field_name.lower()
    if normalized in FIELD_LABELS_KO:
        return FIELD_LABELS_KO[normalized]
    keyword_labels = (
        ("auth", "인증 정보"),
        ("login", "로그인 정보"),
        ("device", "단말 정보"),
        ("host", "호스트 정보"),
        ("time", "발생 시간"),
        ("src", "출발지 정보"),
        ("dst", "목적지 정보"),
        ("port", "통신 포트"),
        ("ip", "네트워크 주소"),
        ("user", "사용자 정보"),
        ("resource", "접근 자원"),
        ("download", "다운로드 정보"),
        ("upload", "업로드 정보"),
    )
    for keyword, label in keyword_labels:
        if keyword in normalized:
            return label
    return "기타 로그 항목"


def _human_value(value: object) -> object:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    return VALUE_LABELS_KO.get(normalized, value)


def _actor_display(actor_id: str, record: Mapping[str, object]) -> str:
    normalized = _normalized_record(record)
    context = " ".join(
        str(normalized.get(field) or "").lower()
        for field in (
            "account_type",
            "workload_role",
            "job_type",
            "action",
            "event_type",
        )
    )
    actor = actor_id.lower()
    if "backup" in actor or "backup" in context:
        return "백업 서비스 계정"
    if actor.startswith(("svc_", "service_")) or "service" in context:
        return "서비스 계정"
    if actor_id == "unknown-actor":
        return "식별되지 않은 사용자"
    return "해당 사용자"


def _is_external_ip(value: object) -> bool | None:
    try:
        address = ipaddress.ip_address(str(value))
    except ValueError:
        return None
    return not (address.is_private or address.is_loopback or address.is_link_local)


def _target_display(record: Mapping[str, object]) -> str:
    normalized = _normalized_record(record)
    job_type = str(normalized.get("job_type") or "").lower()
    if job_type == "nightly_backup":
        return "야간 정기 백업 작업"
    service = str(normalized.get("service") or normalized.get("action") or "").lower()
    destination = normalized.get("dst_ip")
    if service in {"http", "https"}:
        external = _is_external_ip(destination)
        return "외부 웹 서버" if external is not False else "내부 웹 서버"
    if destination not in {None, ""}:
        external = _is_external_ip(destination)
        return "외부 네트워크 대상" if external is not False else "내부 네트워크 대상"
    if any(normalized.get(field) not in {None, ""} for field in ("resource", "resource_id", "asset")):
        return "보호 자원"
    return "업무 대상"


def _action_display(record: Mapping[str, object]) -> str:
    normalized = _normalized_record(record)
    raw_action = next(
        (str(normalized[name]).lower() for name in ACTION_FIELDS if normalized.get(name) not in {None, ""}),
        "",
    )
    if raw_action == "backup":
        return "백업 작업"
    if raw_action in {"http", "https"}:
        return f"{raw_action.upper()} 통신"
    if raw_action in {"login", "logon", "authenticate", "authentication"}:
        return "로그인 인증"
    return str(_human_value(raw_action)) if raw_action else "접근 활동"


def _subject_phrase(text: str) -> str:
    last = text[-1] if text else ""
    code = ord(last) - 0xAC00
    has_final_consonant = 0 <= code <= 0xD7A3 - 0xAC00 and code % 28 != 0
    return f"{text}{'이' if has_final_consonant else '가'}"


def _and_phrase(text: str) -> str:
    last = text[-1] if text else ""
    code = ord(last) - 0xAC00
    has_final_consonant = 0 <= code <= 0xD7A3 - 0xAC00 and code % 28 != 0
    return f"{text}{'과' if has_final_consonant else '와'}"


def _topic_phrase(text: str) -> str:
    last = text[-1] if text else ""
    code = ord(last) - 0xAC00
    has_final_consonant = 0 <= code <= 0xD7A3 - 0xAC00 and code % 28 != 0
    return f"{text}{'은' if has_final_consonant else '는'}"


def _object_phrase(text: str) -> str:
    last = text[-1] if text else ""
    code = ord(last) - 0xAC00
    has_final_consonant = 0 <= code <= 0xD7A3 - 0xAC00 and code % 28 != 0
    return f"{text}{'을' if has_final_consonant else '를'}"


def _baseline_detail(
    registry: BaselineRegistry | None,
    record: Mapping[str, object],
    field_name: str,
    value: object,
) -> dict:
    if registry is None:
        return {
            "known": False,
            "kind": "unavailable",
            "profile_key": None,
            "baseline_key": None,
            "baseline_scope": "unavailable",
            "baseline_scope_label": "사용 가능한 기준선 없음",
            "baseline_selector_field": None,
            "baseline_fallback_used": False,
            "baseline_selection_reason": "정상 표본이 없어 기준선을 선택하지 못했다.",
            "profile_records": 0,
            "field_observations": 0,
            "deviation": None,
            "rarity": None,
            "type_mismatch": None,
            "support": None,
        }

    features, selection = registry.field_features(record, field_name, value)
    profile_key = selection.profile_key
    profile = registry.selected_profile(selection)
    field = None if profile is None else profile.fields.get(field_name.lower())
    detail = {
        "known": field is not None,
        "kind": field.kind if field is not None else "new_field",
        "profile_key": profile_key,
        "baseline_key": selection.baseline_key,
        "baseline_scope": selection.scope,
        "baseline_scope_label": selection.scope_label,
        "baseline_selector_field": selection.selector_field,
        "baseline_fallback_used": selection.fallback_used,
        "baseline_selection_reason": selection.selection_reason,
        "baseline_candidates": [dict(candidate) for candidate in selection.candidates],
        "profile_records": 0 if profile is None else profile.records,
        "field_observations": 0 if field is None else field.observed,
        "presence_surprise": features[2],
        "rarity": features[3],
        "deviation": features[4],
        "type_mismatch": features[5],
        "support": features[6],
        "ready": features[7],
    }
    if field is None:
        return detail

    if field.kind == "numeric":
        number = parse_number(value)
        detail.update(
            {
                "median": field.median,
                "variation_scale": field.scale,
                "robust_z": (
                    abs(number - field.median) / max(field.scale, 1e-6)
                    if number is not None
                    else None
                ),
            }
        )
    else:
        counts = field.category_counts or {}
        if field.kind == "datetime":
            bucket = parse_time_bucket(value)
            count = counts.get(f"hour-of-week:{bucket}", 0) if bucket is not None else 0
        else:
            count = counts.get(value_fingerprint(value), 0)
        detail.update(
            {
                "normal_occurrences": count,
                "normal_frequency": count / max(1, field.observed),
                "unique_values": field.unique_count,
            }
        )
    return detail


def _reason_code(baseline: Mapping[str, object]) -> str:
    if not baseline.get("known"):
        return "NEW_OR_UNBASELINED_FIELD"
    if float(baseline.get("type_mismatch") or 0.0) >= 0.5:
        return "TYPE_MISMATCH"
    if baseline.get("kind") == "numeric" and float(baseline.get("deviation") or 0.0) >= 0.20:
        return "NUMERIC_BASELINE_DEVIATION"
    if baseline.get("kind") == "datetime" and float(baseline.get("rarity") or 0.0) >= 0.20:
        return "RARE_TIME_WINDOW"
    if baseline.get("kind") == "categorical" and float(baseline.get("rarity") or 0.0) >= 0.20:
        return "RARE_CATEGORY"
    return "FIELD_RELATION_EVIDENCE"


def _baseline_anomaly_score(baseline: Mapping[str, object]) -> float:
    """Return the baseline anomaly term used by the monotonic scoring head."""

    if not baseline.get("known"):
        return 1.0 if int(baseline.get("profile_records") or 0) > 0 else 0.0
    return max(
        float(baseline.get("rarity") or 0.0),
        float(baseline.get("deviation") or 0.0),
        float(baseline.get("type_mismatch") or 0.0),
    )


def _learned_risk_multiplier(
    contribution: float,
    request_weight: float,
    baseline_anomaly_score: float,
) -> float | None:
    denominator = request_weight * baseline_anomaly_score * 4.0
    if abs(denominator) < 1e-12:
        return None
    return contribution / denominator


def _reason_message(
    field_name: str,
    value: object,
    baseline: Mapping[str, object],
    contribution: float,
    request_weight: float,
    *,
    field_weight_rank: int | None = None,
    request_field_count: int | None = None,
    monotonic_baseline: bool = False,
    portable_mode: bool = False,
    field_importance_logit: float | None = None,
    field_weight_within_event: float | None = None,
    event_weight: float | None = None,
    model_dimension: int | None = None,
    field_context_layers: int | None = None,
    event_field_count: int | None = None,
) -> str:
    label = _field_label(field_name)
    weight_percent = request_weight * 100.0
    weight_text = (
        f"{weight_percent:.2f}%"
        if abs(weight_percent) >= 0.01
        else f"{weight_percent:.6f}%"
    )
    rank_suffix = f"({field_weight_rank}위)" if field_weight_rank is not None else ""
    weight_clause = f"AI 가중치는 {weight_text}{rank_suffix}였다"
    scope_clause = str(
        baseline.get("baseline_scope_label") or "같은 로그 유형 기준선"
    )
    if not baseline.get("known"):
        return f"{label}: 정상 로그에 없던 새 항목이며 {weight_clause}."
    if float(baseline.get("type_mismatch") or 0.0) >= 0.5:
        return f"{label}: 정상 로그와 자료형이 달랐고 {weight_clause}."
    if baseline.get("kind") == "numeric":
        median = baseline.get("median")
        robust_z = baseline.get("robust_z")
        number = parse_number(value)
        if robust_z is not None and median is not None and number is not None:
            median_number = float(median)
            if abs(median_number) > 1e-12:
                change = (number - median_number) / abs(median_number) * 100.0
                change_text = (
                    f"정상 중앙값 {median_number:.3g} 대비 {abs(change):.1f}% "
                    f"{'증가' if change >= 0 else '감소'}"
                )
            else:
                change_text = f"정상 중앙값 0에서 {number:.3g}(으)로 변화"
            return (
                f"{label}: {scope_clause}에서 관측값 {number:.3g}은 "
                f"{change_text}했고 "
                f"{weight_clause}."
            )
    if (
        baseline.get("kind") in {"categorical", "datetime"}
    ):
        count = int(baseline.get("normal_occurrences") or 0)
        observed = int(baseline.get("field_observations") or 0)
        frequency = count / max(1, observed) * 100.0
        baseline_deviation_pct = _baseline_anomaly_score(baseline) * 100.0
        unique_values = int(baseline.get("unique_values") or 0)
        identifier_field = any(
            token in field_name.lower() for token in ("_id", "hash", "_ip", "ip_")
        )
        if baseline.get("kind") == "datetime":
            subject = "해당 시간대는"
        elif identifier_field:
            subject = "현재 식별값은"
        else:
            subject = f"'{_human_value(value)}' 값은"
        return (
            f"{label}: {scope_clause}의 정상 로그에서 {subject} {observed}건 중 "
            f"{count}회({frequency:.1f}%) 나타났고, 정상값 {unique_values}종의 "
            f"출현 빈도와 비교한 기준선 이탈도는 {baseline_deviation_pct:.2f}%였으며, "
            f"{weight_clause}."
        )
    return f"{label}: 다른 필드와의 조합에서 {weight_clause}."


def _join_korean(items: Sequence[str]) -> str:
    values = [str(item) for item in items if str(item)]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{_and_phrase(values[0])} {values[1]}"
    return f"{', '.join(values[:-2])}, {_and_phrase(values[-2])} {values[-1]}"


def _weight_percent_text(value: float) -> str:
    return f"{value:.2f}%" if abs(value) >= 0.01 else f"{value:.6f}%"


def _number_text(value: float) -> str:
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.3f}".rstrip("0").rstrip(".")


def _behavior_clock_text(record: Mapping[str, object] | None) -> str:
    """Return a short, factual clock label for a behavior narrative."""

    if not record:
        return ""
    raw = str(_normalized_record(record).get("event_time") or "").strip()
    if not raw:
        return ""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return ""
    suffix = " UTC" if raw.endswith("Z") else ""
    return f"{parsed:%H:%M}{suffix}"


def _actor_history_missing(baseline: Mapping[str, object]) -> bool:
    actor_candidate = next(
        (
            candidate
            for candidate in baseline.get("baseline_candidates", [])
            if candidate.get("scope") == "actor"
        ),
        None,
    )
    return bool(
        baseline.get("baseline_fallback_used")
        and isinstance(actor_candidate, Mapping)
        and not actor_candidate.get("selected")
    )


def _focus_ueba_explanation(
    rows: Sequence[Mapping[str, object]],
    actor_display: str,
    record: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Explain the decision as usual behavior, current change, and UEBA judgment."""

    empty = {
        "usual_behavior": "비교할 수 있는 평소 행동 기록이 충분하지 않다.",
        "current_behavior": "현재 요청에서 정상 기준과 비교할 수 있는 주요 행동 항목을 찾지 못했다.",
        "ueba_judgment": "UEBA 행동 근거가 부족해 모델 점수와 정책 준비 상태를 함께 확인해야 한다.",
    }
    if not rows:
        return empty

    if len(rows) >= 2:
        first = rows[0]
        first_baseline = first.get("baseline")
        if isinstance(first_baseline, Mapping):
            comparison_keys = (
                "known",
                "kind",
                "field_observations",
                "normal_occurrences",
            )
            matching_rows = []
            for row in rows:
                baseline = row.get("baseline")
                if not isinstance(baseline, Mapping):
                    continue
                if all(
                    first_baseline.get(key) == baseline.get(key)
                    for key in comparison_keys
                ):
                    matching_rows.append(row)
            if (
                len(matching_rows) >= 2
                and first_baseline.get("known")
                and first_baseline.get("kind") in {"categorical", "datetime"}
            ):
                observations = int(first_baseline.get("field_observations") or 0)
                occurrences = int(first_baseline.get("normal_occurrences") or 0)
                labels = [
                    str(row.get("field_label") or "로그 항목")
                    for row in matching_rows
                ]
                label_text = ", ".join(labels)
                unique_counts = [
                    int((row.get("baseline") or {}).get("unique_values") or 0)
                    for row in matching_rows
                ]
                deviations = [
                    float(row.get("baseline_anomaly_score") or 0.0) * 100.0
                    for row in matching_rows
                ]
                weights = [
                    float(row.get("field_weight") or 0.0) * 100.0
                    for row in matching_rows
                ]
                scope_label = str(
                    first_baseline.get("baseline_scope_label")
                    or "같은 로그 유형 기준선"
                )
                if occurrences == 0:
                    actor_history_missing = _actor_history_missing(first_baseline)
                    comparison_group = (
                        "같은 유형의 정상 로그"
                        if scope_label == "같은 로그 유형 기준선"
                        else f"{scope_label}의 정상 로그"
                    )
                    if actor_history_missing:
                        usual_behavior = (
                            f"{actor_display}의 해당 유형 활동 이력이 충분하지 않아 "
                            f"{comparison_group} {observations}건을 평소 행동 기준으로 사용했다. "
                            f"이 정상 기록에서 {label_text}는 "
                            + (
                                f"각각 {unique_counts[0]}종의 기존 값 안에서 반복됐고, "
                                if unique_counts
                                and min(unique_counts) > 0
                                and len(set(unique_counts)) == 1
                                else "기존에 관찰된 값 안에서 반복됐고, "
                            )
                            + "이번 요청의 값은 모두 처음 나타났다."
                        )
                    else:
                        usual_behavior = (
                            f"평소 {actor_display}의 {comparison_group} {observations}건에는 "
                            f"{label_text}가 기존에 관찰된 값 안에서 반복됐으며, "
                            "이번 요청의 값은 모두 처음 나타났다."
                        )
                    label_set = set(labels)
                    clock_text = _behavior_clock_text(record)
                    clock_prefix = f"{clock_text}에 " if clock_text else ""
                    deviation_text = (
                        f"각각 {deviations[0]:.2f}%"
                        if max(deviations) - min(deviations) < 0.005
                        else ", ".join(f"{value:.2f}%" for value in deviations)
                    )
                    weight_text = ", ".join(
                        _weight_percent_text(value) for value in weights
                    )
                    if {"단말", "출발지 IP", "목적지 IP"}.issubset(label_set):
                        current_behavior = (
                            f"이번 요청은 {clock_prefix}비교 기준에 없던 단말과 출발지 IP에서, "
                            f"이전에 접속 기록이 없던 목적지 IP로 통신했다. "
                            f"세 값의 기준선 이탈도는 {deviation_text}였다."
                        )
                        judgment_intro = (
                            "UEBA는 접속 단말, 출발지 IP와 목적지 IP가 한 요청에서 "
                            "동시에 새로운 값으로 바뀐 것을 복합 이상 행동으로 판단했다."
                        )
                    else:
                        current_behavior = (
                            f"이번 요청은 {clock_prefix}기존 정상 기록에 없던 {label_text} 값을 "
                            f"함께 포함했다. 기준선 이탈도는 {deviation_text}였다."
                        )
                        judgment_intro = (
                            f"UEBA는 {label_text}의 새로운 값이 한 요청에 함께 나타난 것을 "
                            "복합 이상 행동으로 판단했다."
                        )
                    return {
                        "usual_behavior": usual_behavior,
                        "current_behavior": current_behavior,
                        "ueba_judgment": (
                            f"{judgment_intro} AI는 {label_text}에 각각 "
                            f"{weight_text}의 가중치를 반영해 의심 행동으로 분류했다."
                        ),
                    }
                else:
                    frequency = occurrences / max(1, observations) * 100.0
                    deviation_text = ", ".join(
                        f"{value:.2f}%" for value in deviations
                    )
                    weight_text = ", ".join(
                        _weight_percent_text(value) for value in weights
                    )
                    high_change = max(deviations) >= 20.0
                    return {
                        "usual_behavior": (
                            f"평소 행동 기준인 {scope_label}의 정상 로그 {observations}건에서 "
                            f"현재 {label_text} 값은 각각 {occurrences}회({frequency:.1f}%) 나타났다."
                        ),
                        "current_behavior": (
                            f"이번 요청에서 {label_text}의 기준선 이탈도는 각각 "
                            f"{deviation_text}였다."
                        ),
                        "ueba_judgment": (
                            f"UEBA는 이 행동들이 {'평소 범위를 벗어나 함께 나타난 점을 위험 신호로' if high_change else '평소 범위와 크게 다르지 않은 것으로'} "
                            f"판단했다. AI는 {label_text}에 각각 {weight_text}의 가중치를 반영했다."
                        ),
                    }

    selected_rows = list(rows[:3])
    first_baseline = selected_rows[0].get("baseline")
    if not isinstance(first_baseline, Mapping):
        return empty
    scope_label = str(
        first_baseline.get("baseline_scope_label") or "같은 로그 유형 기준선"
    )
    observations = int(first_baseline.get("field_observations") or 0)
    if _actor_history_missing(first_baseline):
        usual_intro = (
            f"{actor_display}의 해당 유형 활동 이력이 충분하지 않아 {scope_label}의 "
            f"정상 로그 {observations}건을 평소 행동 기준으로 사용했다."
        )
    elif first_baseline.get("baseline_scope") == "actor":
        usual_intro = (
            f"평소 {actor_display}의 정상 로그 {observations}건을 행동 기준으로 사용했다."
        )
    else:
        usual_intro = (
            f"{scope_label}의 정상 로그 {observations}건을 평소 행동 기준으로 사용했다."
        )

    usual_parts: list[str] = []
    current_parts: list[str] = []
    for row in selected_rows:
        baseline = row.get("baseline")
        if not isinstance(baseline, Mapping):
            continue
        label = str(row.get("field_label") or "로그 항목")
        value = row.get("observed_value")
        deviation = float(row.get("baseline_anomaly_score") or 0.0) * 100.0
        if baseline.get("kind") == "numeric" and baseline.get("median") is not None:
            median = float(baseline.get("median") or 0.0)
            number = parse_number(value)
            usual_parts.append(
                f"{label}의 정상 중앙값은 {_number_text(median)}이었다"
            )
            if number is not None:
                if abs(median) > 1e-12:
                    change = (number - median) / abs(median) * 100.0
                    current_parts.append(
                        f"현재 {label} 값은 {_number_text(number)}이며, 평소보다 "
                        f"{abs(change):.1f}% "
                        f"{'증가했다' if change >= 0 else '감소했다'}"
                    )
                else:
                    current_parts.append(
                        f"현재 {label} 값은 {_number_text(number)}이며, 정상 중앙값은 0이었다"
                    )
        elif baseline.get("kind") in {"categorical", "datetime"}:
            count = int(baseline.get("normal_occurrences") or 0)
            observed = int(baseline.get("field_observations") or 0)
            usual_parts.append(
                f"현재 {label} 값은 정상 로그 {observed}건에서 {count}회 나타났다"
            )
            current_parts.append(f"{label}의 기준선 이탈도는 {deviation:.2f}%였다")
        else:
            current_parts.append(f"{label}은 기존 정상 기록과 직접 비교할 수 없는 새 항목이었다")

    anomalous_rows = [
        row
        for row in selected_rows
        if float(row.get("baseline_anomaly_score") or 0.0) >= 0.20
    ]
    labels = [str(row.get("field_label") or "로그 항목") for row in selected_rows]
    weights = [float(row.get("field_weight") or 0.0) * 100.0 for row in selected_rows]
    label_text = ", ".join(labels)
    weight_text = ", ".join(_weight_percent_text(weight) for weight in weights)
    if len(labels) == 1:
        weight_sentence = (
            f"AI는 {labels[0]}에 {weight_text}의 가중치를 반영했다."
        )
    else:
        weight_sentence = (
            f"AI는 {label_text}에 각각 {weight_text}의 가중치를 반영했다."
        )
    if anomalous_rows:
        anomaly_labels = [
            str(row.get("field_label") or "로그 항목") for row in anomalous_rows
        ]
        if len(anomaly_labels) == 1:
            anomaly_reason = (
                f"{_subject_phrase(anomaly_labels[0])} 평소 행동 범위를 크게 벗어난 점"
            )
        else:
            anomaly_reason = (
                f"{_subject_phrase(_join_korean(anomaly_labels))} 평소 행동 범위를 함께 벗어난 점"
            )
        judgment = f"UEBA는 {anomaly_reason}을 위험 신호로 판단했다. {weight_sentence}"
    else:
        judgment = (
            f"UEBA는 주요 행동이 평소 범위와 크게 다르지 않은 것으로 판단했다. "
            f"{weight_sentence}"
        )
    return {
        "usual_behavior": " ".join(
            part for part in (usual_intro, "; ".join(usual_parts) + "." if usual_parts else "") if part
        ),
        "current_behavior": "; ".join(current_parts) + "." if current_parts else empty["current_behavior"],
        "ueba_judgment": judgment,
    }


def _focus_reason(
    rows: Sequence[Mapping[str, object]], actor_display: str
) -> str:
    explanation = _focus_ueba_explanation(rows, actor_display)
    return " ".join(explanation.values())


def _event_sentence(
    record: Mapping[str, object], actor_id: str
) -> tuple[str, str, str, str]:
    normalized = _normalized_record(record)
    event_type = EVENT_LABELS_KO.get(
        str(normalized.get("event_type") or "").lower(), "접근 활동"
    )
    actor_display = _actor_display(actor_id, record)
    target_display = _target_display(record)
    action_display = _action_display(record)
    title = f"{actor_display}의 {event_type}"
    if target_display == "야간 정기 백업 작업":
        sentence = f"{_subject_phrase(actor_display)} 야간 정기 백업 작업을 수행했다."
    elif "통신" in action_display:
        sentence = f"{actor_display}에서 {target_display}로 {action_display}이 발생했다."
    else:
        sentence = f"{actor_display}에서 {target_display}에 대한 {action_display}이 관찰됐다."
    return title, sentence, target_display, actor_display


def _label_counterfactual(value: Mapping[str, object] | None) -> dict:
    if not value:
        return {}

    def label_single(row: Mapping[str, object]) -> dict:
        field = str(row.get("field") or "")
        return {**row, "field_label": _field_label(field)}

    def label_pair(row: Mapping[str, object]) -> dict:
        fields = [str(field) for field in row.get("fields", [])]
        return {**row, "field_labels": [_field_label(field) for field in fields]}

    single_tests = [
        label_single(row) for row in value.get("single_field_tests", [])
    ]
    pair_tests = [label_pair(row) for row in value.get("pair_tests", [])]
    coalition_source = value.get("coalition_analysis")
    coalition = None
    if isinstance(coalition_source, Mapping):
        coalition = {
            **coalition_source,
            "field_marginal_contributions": [
                label_single(row)
                for row in coalition_source.get("field_marginal_contributions", [])
            ],
            "pair_interactions": [
                label_pair(row)
                for row in coalition_source.get("pair_interactions", [])
            ],
        }
    primary_source = value.get("primary_evidence")
    primary = None
    if isinstance(primary_source, Mapping):
        primary = (
            label_pair(primary_source)
            if primary_source.get("type") == "field_pair"
            else label_single(primary_source)
        )
    discovery_source = value.get("candidate_discovery")
    discovery = None
    if isinstance(discovery_source, Mapping):
        screening_source = discovery_source.get("screening")
        screening = dict(screening_source) if isinstance(screening_source, Mapping) else {}
        screening["strongest_screened_pairs"] = [
            label_pair(row) for row in screening.get("strongest_screened_pairs", [])
        ]
        discovery = {
            **discovery_source,
            "pool_field_labels": [
                _field_label(str(field))
                for field in discovery_source.get("pool_fields", [])
            ],
            "selected_field_labels": [
                _field_label(str(field))
                for field in discovery_source.get("selected_fields", [])
            ],
            "candidate_rows": [
                label_single(row)
                for row in discovery_source.get("candidate_rows", [])
            ],
            "screening": screening,
        }
    return {
        **value,
        "role": "model_influence_validation",
        "single_field_tests": single_tests,
        "pair_tests": pair_tests,
        "coalition_analysis": coalition,
        "candidate_discovery": discovery,
        "primary_evidence": primary,
    }


def _counterfactual_reason(
    value: Mapping[str, object] | None,
) -> tuple[str, str]:
    if value is None:
        return (
            "상세 필드 조합 재검사는 별도 분석 단계에서 수행한다.",
            "설명 신뢰도는 조합 재검사가 완료되면 계산한다.",
        )
    if value.get("status") == "unavailable":
        return (
            "필드 조합 재검사에 필요한 모델 위험 확률을 얻지 못했다.",
            "설명 신뢰도는 계산하지 못했다.",
        )
    baseline = float(value.get("baseline_raw_risk_probability") or 0.0)
    primary = value.get("primary_evidence")
    if isinstance(primary, Mapping) and primary.get("type") == "field_pair":
        labels = [str(label) for label in primary.get("field_labels", [])]
        label_text = _and_phrase(labels[0]) + f" {labels[1]}" if len(labels) >= 2 else "두 필드"
        changed = float(primary.get("counterfactual_raw_risk_probability") or 0.0)
        drop = float(primary.get("risk_probability_drop") or 0.0) * 100.0
        additional = float(
            primary.get("additional_drop_over_strongest_single") or 0.0
        ) * 100.0
        influence = (
            f"{label_text}를 함께 정상 기준 상태로 바꿔 재검사하자 모델 내부 위험 확률이 "
            f"{baseline * 100.0:.2f}%에서 {changed * 100.0:.2f}%로 {drop:.2f}%p 낮아졌다. "
            f"둘 중 하나만 바꾼 경우보다 {additional:.2f}%p 더 낮아져 두 필드의 결합이 "
            "판정에 영향을 준 것으로 확인됐다."
        )
    elif isinstance(primary, Mapping):
        label = str(primary.get("field_label") or "해당 필드")
        changed = float(primary.get("counterfactual_raw_risk_probability") or 0.0)
        drop = float(primary.get("risk_probability_drop") or 0.0) * 100.0
        reference = str(primary.get("reference_description") or "정상 기준 상태")
        influence = (
            f"{label}를 {reference}로 바꿔 재검사하자 모델 내부 위험 확률이 "
            f"{baseline * 100.0:.2f}%에서 {changed * 100.0:.2f}%로 {drop:.2f}%p 낮아졌다. "
            "따라서 이 필드는 점수에 실제 영향을 준 판정 근거다."
        )
    else:
        influence = (
            "후보 필드를 정상 기준 상태로 바꿔도 모델 내부 위험 확률이 "
            "0.50%p 이상 낮아지지 않아 단일 핵심 근거를 확정하지 않았다."
        )

    confidence = value.get("explanation_confidence")
    if not isinstance(confidence, Mapping):
        return influence, "설명 신뢰도는 계산하지 못했다."
    grade = str(confidence.get("grade_label") or "확인 필요")
    components = confidence.get("components") or {}
    baseline_adequacy = float(components.get("baseline_adequacy") or 0.0) * 100.0
    candidate_coverage = float(components.get("candidate_coverage") or 0.0) * 100.0
    consistency = float(
        components.get("coalition_direction_consistency") or 0.0
    ) * 100.0
    quality = (
        f"설명 품질은 {grade}으로 평가했다. 정상 기준선 충족도 "
        f"{baseline_adequacy:.0f}%, 후보 점수 포착률 {candidate_coverage:.0f}%, "
        f"조합별 영향 방향 일치율 {consistency:.0f}%를 함께 반영했다."
    )
    return influence, quality


def _analysis_scope_reason(value: Mapping[str, object] | None) -> str:
    if not value or value.get("status") != "available":
        return "상세 조합 분석 범위는 분석 완료 후 제공한다."
    discovery = value.get("candidate_discovery")
    coalition = value.get("coalition_analysis")
    if not isinstance(discovery, Mapping) or not isinstance(coalition, Mapping):
        return "상세 조합 분석 범위를 확인하지 못했다."
    screening = discovery.get("screening") or {}
    selected_labels = discovery.get("selected_field_labels") or discovery.get(
        "selected_fields", []
    )
    return (
        f"기여도·AI 가중치·기준선 이탈도로 {int(discovery.get('pool_size') or 0)}개 "
        f"후보를 모으고, 단일 {int(screening.get('evaluated_single_fields') or 0)}개와 "
        f"필드 쌍 {int(screening.get('evaluated_field_pairs') or 0)}개를 먼저 재검사했다. "
        f"그 결과 {_object_phrase(', '.join(str(label) for label in selected_labels))} 최종 후보로 골라 "
        f"{int(coalition.get('evaluated_coalitions') or 0)}개 부분 조합을 검사했다."
    )


def build_operator_explanation(
    record: Mapping[str, object],
    prediction: Mapping[str, object],
    baseline_registry: BaselineRegistry | None,
    *,
    top_k: int = 5,
    counterfactual: Mapping[str, object] | None = None,
) -> dict:
    """Return a faithful Korean explanation backed by model and baseline values."""

    actor_id = extract_actor_id(record)
    event_id = stable_event_id(record)
    occurred_at = normalize_event_time(_normalized_record(record).get("event_time"))
    policy = dict(prediction.get("policy") or {})
    stage = str(policy.get("stage") or "shadow")
    policy_label = POLICY_LABELS_KO.get(stage, stage)
    title, event_sentence, target_display, actor_display = _event_sentence(record, actor_id)
    normalized = _normalized_record(record)

    evidence_rows = []
    monotonic_baseline = bool(prediction.get("monotonic_baseline"))
    portable_mode = bool(prediction.get("portable_mode"))
    model_dimension = int(prediction.get("model_dimension") or 128)
    field_context_layers = int(prediction.get("field_context_layers") or 2)
    for returned_rank, row in enumerate(prediction.get("top_fields", []), start=1):
        if int(row.get("event_index", 0)) != 0:
            continue
        field_name = str(row.get("field") or "")
        if not field_name or field_name.lower() in GROUND_TRUTH_FIELDS:
            continue
        value = normalized.get(field_name.lower())
        contribution = float(row.get("risk_logit_contribution") or 0.0)
        baseline = _baseline_detail(baseline_registry, record, field_name, value)
        displayed_value = _display_value(field_name, value)
        request_weight = float(row.get("weight") or 0.0)
        event_weight = float(row.get("event_weight") or 0.0)
        field_weight_within_event = float(row.get("field_weight_within_event") or 0.0)
        field_importance_logit = row.get("field_importance_logit")
        field_weight_rank = row.get("field_weight_rank")
        risk_contribution_rank = row.get("risk_contribution_rank", returned_rank)
        request_field_count = row.get("request_field_count")
        event_field_count = row.get("event_field_count")
        baseline_anomaly_score = _baseline_anomaly_score(baseline)
        learned_risk_multiplier = (
            _learned_risk_multiplier(
                contribution, request_weight, baseline_anomaly_score
            )
            if monotonic_baseline
            else None
        )
        evidence_rows.append(
            {
                "reason_code": _reason_code(baseline),
                "field": field_name,
                "field_label": _field_label(field_name),
                "observed_value": displayed_value,
                "baseline": baseline,
                "field_weight": request_weight,
                "event_weight": event_weight,
                "field_weight_within_event": field_weight_within_event,
                "field_importance_logit": field_importance_logit,
                "field_weight_rank": field_weight_rank,
                "risk_contribution_rank": risk_contribution_rank,
                "request_field_count": request_field_count,
                "event_field_count": event_field_count,
                "baseline_anomaly_score": baseline_anomaly_score,
                "learned_risk_multiplier": learned_risk_multiplier,
                "model_risk_contribution": contribution,
                "selection_basis": {
                    "metric": "absolute_model_risk_contribution",
                    "rank": risk_contribution_rank,
                    "reason": (
                        "현재 요청의 모든 필드 중 모델 위험 기여도 절댓값이 "
                        f"{int(risk_contribution_rank)}위여서 판정 근거로 선택했다."
                        if risk_contribution_rank is not None
                        else "모델 위험 기여도 상위 필드여서 판정 근거로 선택했다."
                    ),
                },
                "direction": "risk_increase" if contribution >= 0 else "risk_decrease",
                "message": _reason_message(
                    field_name,
                    displayed_value,
                    baseline,
                    contribution,
                    request_weight,
                    field_weight_rank=(
                        int(field_weight_rank) if field_weight_rank is not None else None
                    ),
                    request_field_count=(
                        int(request_field_count) if request_field_count is not None else None
                    ),
                    monotonic_baseline=monotonic_baseline,
                    portable_mode=portable_mode,
                    field_importance_logit=(
                        float(field_importance_logit)
                        if field_importance_logit is not None
                        else None
                    ),
                    field_weight_within_event=field_weight_within_event,
                    event_weight=event_weight,
                    model_dimension=model_dimension,
                    field_context_layers=field_context_layers,
                    event_field_count=(
                        int(event_field_count) if event_field_count is not None else None
                    ),
                ),
            }
        )
        if len(evidence_rows) >= top_k:
            break

    increasing = [row for row in evidence_rows if row["model_risk_contribution"] >= 0]
    decreasing = [row for row in evidence_rows if row["model_risk_contribution"] < 0]
    counterfactual_evidence = _label_counterfactual(counterfactual)
    readable_increasing = [
        row for row in increasing if row.get("field_label") != "기타 로그 항목"
    ]
    focus = readable_increasing[:3] or increasing[:3] or evidence_rows[:3]
    ueba_explanation = _focus_ueba_explanation(focus, actor_display, record)
    usual_behavior = ueba_explanation["usual_behavior"]
    current_behavior = ueba_explanation["current_behavior"]
    ueba_judgment = ueba_explanation["ueba_judgment"]
    baseline_reason = " ".join((usual_behavior, current_behavior))
    influence_reason, confidence_reason = _counterfactual_reason(
        counterfactual_evidence
    )
    analysis_scope_reason = _analysis_scope_reason(counterfactual_evidence)
    reason_text = " ".join(
        part for part in (usual_behavior, current_behavior, ueba_judgment) if part
    )

    trust_score = prediction.get("trust_score")
    risk_score = prediction.get("risk_score")
    if risk_score is None:
        decision_sentence = f"기준선 또는 점수 보정이 준비되지 않아 {policy_label} 정책으로 기록했다."
    elif stage == "shadow":
        decision_sentence = (
            f"위험 점수 {float(risk_score):.3f}, Trust Score "
            f"{float(trust_score):.2f}이지만 정상 기준선·점수 보정·새 로그 검증이 "
            f"완료되지 않아 {policy_label} 정책을 제안했다."
        )
    else:
        situation = {
            "allow": "정상 상황",
            "monitor": "관찰이 필요한 상황",
            "step_up": "추가 확인이 필요한 의심 상황",
            "restrict": "고위험 상황",
            "deny": "정상 기준을 벗어난 의심 상황",
        }.get(stage, "정책 확인이 필요한 상황")
        decision_sentence = (
            f"위험 점수 {float(risk_score):.3f}, Trust Score "
            f"{float(trust_score):.2f}로 {situation}이며 {policy_label} 정책을 제안했다."
        )

    return {
        "event": {
            "event_id": event_id,
            "actor_id": actor_id,
            "actor_display": actor_display,
            "occurred_at": occurred_at,
            "source_type": normalized.get("source_type"),
            "event_type": normalized.get("event_type"),
            "target_display": target_display,
            "title": title,
        },
        "readable_log": {
            "headline": f"{policy_label}: {title}",
            "event_summary": event_sentence,
            "usual_behavior": usual_behavior,
            "current_behavior": current_behavior,
            "ueba_judgment": ueba_judgment,
            "baseline_reason": baseline_reason,
            "technical_validation_reason": influence_reason,
            "model_influence_reason": influence_reason,
            "explanation_confidence_reason": confidence_reason,
            "analysis_scope_reason": analysis_scope_reason,
            "judgment_reason": reason_text,
            "policy_reason": decision_sentence,
            # Backward-compatible aliases for existing API clients.
            "event": event_sentence,
            "reason": reason_text,
            "decision": decision_sentence,
            "full_text": f"{occurred_at} {event_sentence} {reason_text} {decision_sentence}",
        },
        "risk_increasing_evidence": increasing,
        "risk_decreasing_evidence": decreasing,
        "counterfactual_evidence": counterfactual_evidence,
        "audit": {
            "explanation_method": "hierarchical_ueba_behavior_explanation_v6",
            "ueba_baseline": prediction.get("ueba_baseline"),
            "counterfactual_role": "model_influence_validation",
            "uses_generative_ai": False,
            "portable_mode": portable_mode,
            "evidence_formula": (
                "field_weight * baseline_anomaly_score * "
                "learned_risk_multiplier * 4"
                if monotonic_baseline
                else "model field contribution + baseline-relative evidence"
            ),
            "ground_truth_fields_removed": True,
        },
    }


def common_link_indicators(left: Mapping[str, object], right: Mapping[str, object]) -> list[dict]:
    """Return concrete shared indicators that explain why two events are linked."""

    left_normalized = _normalized_record(left)
    right_normalized = _normalized_record(right)
    links = []
    for field in LINK_FIELDS:
        left_value = left_normalized.get(field)
        right_value = right_normalized.get(field)
        if left_value in {None, ""} or right_value in {None, ""}:
            continue
        if str(left_value) == str(right_value):
            links.append(
                {
                    "type": "shared_indicator",
                    "field": field,
                    "value": _display_value(field, left_value),
                    "message": f"두 이벤트가 같은 {_field_label(field)} 값을 사용했다.",
                }
            )
    return links
