"""Faithful counterfactual explanations over field coalitions."""

from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math
from typing import Mapping, Protocol, Sequence

from .baseline import BaselineRegistry
from .tokenizer import NORMAL_REFERENCE_FIELDS_KEY


class CounterfactualPredictor(Protocol):
    baseline_registry: BaselineRegistry | None

    def predict(self, payload: Mapping[str, object], *, top_k: int = 10) -> dict: ...


def _bounded_probability(value: object) -> float:
    return min(1.0 - 1e-9, max(1e-9, float(value)))


def _raw_risk(prediction: Mapping[str, object]) -> float:
    value = prediction.get("raw_model_risk_probability")
    if value is None:
        raise ValueError("prediction does not contain raw_model_risk_probability")
    return _bounded_probability(value)


def _logit(probability: float) -> float:
    probability = _bounded_probability(probability)
    return math.log(probability / (1.0 - probability))


def _record_key(record: Mapping[str, object], field_name: str) -> str | None:
    wanted = field_name.strip().lower()
    for key in record:
        if str(key).strip().lower() == wanted:
            return str(key)
    return None


def _selected_field_baseline(
    registry: BaselineRegistry | None,
    record: Mapping[str, object],
    field_name: str,
):
    if registry is None:
        return None, None
    selection = registry.resolve(record, field_name)
    profile = registry.selected_profile(selection)
    field = None if profile is None else profile.fields.get(field_name.lower())
    return field, selection


def _mark_normal_reference(record: dict[str, object], field_name: str) -> None:
    existing = record.get(NORMAL_REFERENCE_FIELDS_KEY, ())
    values = list(existing) if isinstance(existing, (list, tuple, set, frozenset)) else []
    if field_name not in values:
        values.append(field_name)
    record[NORMAL_REFERENCE_FIELDS_KEY] = values


def _intervene(
    record: dict[str, object],
    field_name: str,
    registry: BaselineRegistry | None,
) -> dict | None:
    """Move one field toward its selected normal baseline without raw-value lookup."""

    key = _record_key(record, field_name)
    if key is None:
        return None
    field, selection = _selected_field_baseline(registry, record, field_name)
    if field is not None and selection is not None:
        original = record[key]
        _mark_normal_reference(record, field_name)
        common = {
            "field": field_name,
            "baseline_kind": field.kind,
            "baseline_scope": selection.scope,
            "baseline_scope_label": selection.scope_label,
            "baseline_observations": int(field.observed),
            "uses_normal_reference": True,
        }
        if field.kind == "numeric":
            record[key] = field.median
            return {
                **common,
                "method": "replace_with_normal_median",
                "reference_value": field.median,
                "reference_description": "선택된 정상 기준선의 중앙값",
                "changed": original != field.median,
            }
        return {
            **common,
            "method": "neutralize_with_normal_baseline",
            "reference_value": None,
            "reference_description": (
                "원문 정상값을 복원하지 않고 선택된 기준선의 "
                "희귀도·편차 특징을 정상 상태로 중화"
            ),
            "changed": True,
        }

    del record[key]
    return {
        "field": field_name,
        "method": "mask_field_without_baseline",
        "reference_value": None,
        "reference_description": "사용 가능한 정상 기준선이 없어 필드를 제외",
        "baseline_kind": "unavailable",
        "baseline_scope": "unavailable",
        "baseline_scope_label": "사용 가능한 기준선 없음",
        "baseline_observations": 0,
        "uses_normal_reference": False,
        "changed": True,
    }


def _baseline_signal(
    registry: BaselineRegistry | None,
    event: Mapping[str, object],
    field_name: str,
) -> dict[str, float | int | str | bool]:
    key = _record_key(event, field_name)
    if registry is None or key is None:
        return {
            "baseline_anomaly": 0.0,
            "baseline_support": 0.0,
            "baseline_observations": 0,
            "baseline_scope": "unavailable",
            "baseline_ready": False,
        }
    features, selection = registry.field_features(event, field_name, event[key])
    return {
        "baseline_anomaly": max(float(value) for value in features[2:6]),
        "baseline_support": float(features[6]),
        "baseline_observations": int(selection.field_observations),
        "baseline_scope": selection.scope,
        "baseline_ready": bool(features[7] >= 1.0),
    }


def _candidate_rows(
    event: Mapping[str, object],
    prediction: Mapping[str, object],
    registry: BaselineRegistry | None,
    pool_limit: int,
) -> list[dict]:
    protected = {"event_id", "request_id", "trace_id"}
    if registry is not None:
        protected.update(field.lower() for field in registry.profile_fields)

    rows: list[dict] = []
    seen: set[str] = set()
    for original_rank, source in enumerate(prediction.get("top_fields", []), start=1):
        if int(source.get("event_index", 0)) != 0:
            continue
        field = str(source.get("field") or "").strip()
        normalized = field.lower()
        if not field or normalized in protected or normalized in seen:
            continue
        seen.add(normalized)
        baseline = _baseline_signal(registry, event, field)
        rows.append(
            {
                "field": field,
                "original_rank": original_rank,
                "absolute_model_contribution": abs(
                    float(source.get("risk_logit_contribution") or 0.0)
                ),
                "attention_weight": abs(float(source.get("weight") or 0.0)),
                **baseline,
            }
        )

    if not rows:
        return []
    maxima = {
        key: max(float(row[key]) for row in rows) or 1.0
        for key in (
            "absolute_model_contribution",
            "attention_weight",
            "baseline_anomaly",
        )
    }
    for row in rows:
        contribution = float(row["absolute_model_contribution"]) / maxima[
            "absolute_model_contribution"
        ]
        attention = float(row["attention_weight"]) / maxima["attention_weight"]
        anomaly = float(row["baseline_anomaly"]) / maxima["baseline_anomaly"]
        row["hybrid_candidate_score"] = (
            0.45 * contribution + 0.20 * attention + 0.35 * anomaly
        )

    rankings = [
        sorted(rows, key=lambda row: float(row[key]), reverse=True)
        for key in (
            "hybrid_candidate_score",
            "absolute_model_contribution",
            "baseline_anomaly",
            "attention_weight",
        )
    ]
    selected: list[dict] = []
    selected_names: set[str] = set()
    rank = 0
    safe_limit = max(1, int(pool_limit))
    while len(selected) < min(safe_limit, len(rows)):
        added = False
        for ranking in rankings:
            if rank >= len(ranking):
                continue
            row = ranking[rank]
            if row["field"] in selected_names:
                continue
            selected.append(row)
            selected_names.add(str(row["field"]))
            added = True
            if len(selected) >= safe_limit:
                break
        if not added and rank >= len(rows):
            break
        rank += 1
    selected.sort(key=lambda row: float(row["hybrid_candidate_score"]), reverse=True)
    for pool_rank, row in enumerate(selected, start=1):
        row["pool_rank"] = pool_rank
    return selected


def _all_subsets(fields: tuple[str, ...]) -> list[frozenset[str]]:
    return [
        frozenset(subset)
        for size in range(len(fields) + 1)
        for subset in combinations(fields, size)
    ]


def _changed_event(
    event: Mapping[str, object],
    changed_fields: Sequence[str],
    registry: BaselineRegistry | None,
) -> dict[str, object]:
    changed_event = deepcopy(dict(event))
    for field in changed_fields:
        _intervene(changed_event, field, registry)
    return changed_event


def _predict_raw_many(
    predictor: CounterfactualPredictor,
    events: Sequence[Mapping[str, object]],
) -> tuple[list[float], int]:
    if not events:
        return [], 0
    batch_method = getattr(predictor, "predict_raw_batch", None)
    if callable(batch_method):
        values = list(batch_method(events))
        if len(values) != len(events):
            raise ValueError("predict_raw_batch returned an unexpected result count")
        return [_bounded_probability(value) for value in values], 1
    return [
        _raw_risk(predictor.predict(event, top_k=2)) for event in events
    ], len(events)


def _evaluate_kept_sets(
    predictor: CounterfactualPredictor,
    event: Mapping[str, object],
    prediction: Mapping[str, object],
    fields: tuple[str, ...],
    kept_sets: Sequence[frozenset[str]],
    registry: BaselineRegistry | None,
) -> tuple[dict[frozenset[str], dict[str, float]], int]:
    full = frozenset(fields)
    baseline_risk = _raw_risk(prediction)
    scores: dict[frozenset[str], dict[str, float]] = {
        full: {
            "risk_probability": baseline_risk,
            "risk_logit": _logit(baseline_risk),
        }
    }
    pending = [kept for kept in kept_sets if kept != full and kept not in scores]
    variants = [
        _changed_event(event, [field for field in fields if field not in kept], registry)
        for kept in pending
    ]
    risks, forward_calls = _predict_raw_many(predictor, variants)
    for kept, risk in zip(pending, risks, strict=True):
        scores[kept] = {
            "risk_probability": risk,
            "risk_logit": _logit(risk),
        }
    return scores, forward_calls


def _field_marginals(
    fields: tuple[str, ...],
    scores: Mapping[frozenset[str], Mapping[str, float]],
) -> list[dict]:
    count = len(fields)
    if count == 0:
        return []
    denominator = math.factorial(count)
    rows: list[dict] = []
    for field in fields:
        others = tuple(item for item in fields if item != field)
        probability_contribution = 0.0
        logit_contribution = 0.0
        for subset in _all_subsets(others):
            coefficient = (
                math.factorial(len(subset))
                * math.factorial(count - len(subset) - 1)
                / denominator
            )
            with_field = subset | {field}
            probability_contribution += coefficient * (
                scores[with_field]["risk_probability"]
                - scores[subset]["risk_probability"]
            )
            logit_contribution += coefficient * (
                scores[with_field]["risk_logit"] - scores[subset]["risk_logit"]
            )
        rows.append(
            {
                "field": field,
                "shapley_risk_probability_contribution": probability_contribution,
                "shapley_risk_logit_contribution": logit_contribution,
                "direction": (
                    "risk_increase" if logit_contribution >= 0 else "risk_decrease"
                ),
            }
        )
    total_absolute = sum(abs(row["shapley_risk_logit_contribution"]) for row in rows)
    for row in rows:
        row["influence_share"] = (
            abs(row["shapley_risk_logit_contribution"]) / total_absolute
            if total_absolute > 0
            else 0.0
        )
    rows.sort(key=lambda row: abs(row["shapley_risk_logit_contribution"]), reverse=True)
    return rows


def _pair_interactions(
    fields: tuple[str, ...],
    scores: Mapping[frozenset[str], Mapping[str, float]],
) -> list[dict]:
    count = len(fields)
    if count < 2:
        return []
    denominator = math.factorial(count - 1)
    rows: list[dict] = []
    for first, second in combinations(fields, 2):
        others = tuple(item for item in fields if item not in {first, second})
        probability_interaction = 0.0
        logit_interaction = 0.0
        for subset in _all_subsets(others):
            coefficient = (
                math.factorial(len(subset))
                * math.factorial(count - len(subset) - 2)
                / denominator
            )
            first_only = subset | {first}
            second_only = subset | {second}
            both = subset | {first, second}
            probability_interaction += coefficient * (
                scores[both]["risk_probability"]
                - scores[first_only]["risk_probability"]
                - scores[second_only]["risk_probability"]
                + scores[subset]["risk_probability"]
            )
            logit_interaction += coefficient * (
                scores[both]["risk_logit"]
                - scores[first_only]["risk_logit"]
                - scores[second_only]["risk_logit"]
                + scores[subset]["risk_logit"]
            )
        tolerance = 1e-12
        rows.append(
            {
                "fields": [first, second],
                "shapley_interaction_probability": probability_interaction,
                "shapley_interaction_logit": logit_interaction,
                "interaction_type": (
                    "synergy"
                    if logit_interaction > tolerance
                    else "redundancy"
                    if logit_interaction < -tolerance
                    else "independent"
                ),
            }
        )
    rows.sort(key=lambda row: abs(row["shapley_interaction_logit"]), reverse=True)
    return rows


def _screen_candidate_pool(
    predictor: CounterfactualPredictor,
    event: Mapping[str, object],
    prediction: Mapping[str, object],
    candidate_rows: list[dict],
    final_limit: int,
    registry: BaselineRegistry | None,
) -> tuple[tuple[str, ...], dict]:
    fields = tuple(str(row["field"]) for row in candidate_rows)
    if len(fields) <= final_limit:
        return fields, {
            "performed": False,
            "evaluated_single_fields": 0,
            "evaluated_field_pairs": 0,
            "model_reinferences": 0,
            "model_forward_calls": 0,
        }

    full = frozenset(fields)
    single_sets = [full - {field} for field in fields]
    pair_sets = [full - {first, second} for first, second in combinations(fields, 2)]
    scores, forward_calls = _evaluate_kept_sets(
        predictor,
        event,
        prediction,
        fields,
        [*single_sets, *pair_sets],
        registry,
    )
    baseline_logit = _logit(_raw_risk(prediction))
    single_effects = {
        field: baseline_logit - scores[full - {field}]["risk_logit"]
        for field in fields
    }
    pair_effects: dict[str, float] = {field: 0.0 for field in fields}
    screened_pairs: list[dict] = []
    for first, second in combinations(fields, 2):
        combined_drop = baseline_logit - scores[full - {first, second}]["risk_logit"]
        interaction = combined_drop - single_effects[first] - single_effects[second]
        magnitude = abs(interaction)
        pair_effects[first] = max(pair_effects[first], magnitude)
        pair_effects[second] = max(pair_effects[second], magnitude)
        screened_pairs.append(
            {
                "fields": [first, second],
                "combined_risk_logit_drop": combined_drop,
                "interaction_magnitude": magnitude,
            }
        )
    max_single = max((abs(value) for value in single_effects.values()), default=1.0) or 1.0
    max_pair = max(pair_effects.values(), default=1.0) or 1.0
    for row in candidate_rows:
        field = str(row["field"])
        row["screen_single_effect"] = single_effects[field]
        row["screen_pair_interaction"] = pair_effects[field]
        row["final_candidate_score"] = (
            0.35 * float(row["hybrid_candidate_score"])
            + 0.35 * abs(single_effects[field]) / max_single
            + 0.30 * pair_effects[field] / max_pair
        )
    ranked = sorted(
        candidate_rows,
        key=lambda row: float(row["final_candidate_score"]),
        reverse=True,
    )
    selected = tuple(str(row["field"]) for row in ranked[:final_limit])
    screened_pairs.sort(key=lambda row: row["interaction_magnitude"], reverse=True)
    return selected, {
        "performed": True,
        "evaluated_single_fields": len(single_sets),
        "evaluated_field_pairs": len(pair_sets),
        "model_reinferences": len(single_sets) + len(pair_sets),
        "model_forward_calls": forward_calls,
        "strongest_screened_pairs": screened_pairs[:5],
    }


def _explanation_confidence(
    candidate_rows: Sequence[Mapping[str, object]],
    selected_fields: Sequence[str],
    interventions: Mapping[str, Mapping[str, object]],
    single_tests: Sequence[Mapping[str, object]],
    marginals: Sequence[Mapping[str, object]],
    combined_logit_contribution: float,
) -> dict:
    selected = set(selected_fields)
    selected_rows = [row for row in candidate_rows if row.get("field") in selected]
    baseline_adequacy = (
        sum(
            min(1.0, float(row.get("baseline_observations") or 0) / 20.0)
            for row in selected_rows
        )
        / len(selected_rows)
        if selected_rows
        else 0.0
    )
    total_candidate_mass = sum(
        float(row.get("hybrid_candidate_score") or 0.0) for row in candidate_rows
    )
    selected_mass = sum(
        float(row.get("hybrid_candidate_score") or 0.0) for row in selected_rows
    )
    candidate_coverage = (
        selected_mass / total_candidate_mass if total_candidate_mass > 0 else 1.0
    )
    intervention_realism = (
        sum(
            1.0 if interventions[field].get("uses_normal_reference") else 0.0
            for field in selected_fields
        )
        / len(selected_fields)
        if selected_fields
        else 0.0
    )
    single_by_field = {str(row.get("field")): row for row in single_tests}
    comparable = 0
    agreeing = 0
    for marginal in marginals:
        field = str(marginal.get("field"))
        single = float(single_by_field.get(field, {}).get("risk_logit_drop") or 0.0)
        shapley = float(marginal.get("shapley_risk_logit_contribution") or 0.0)
        if abs(single) <= 1e-9 or abs(shapley) <= 1e-9:
            continue
        comparable += 1
        agreeing += int((single > 0) == (shapley > 0))
    coalition_direction_consistency = agreeing / comparable if comparable else 1.0
    reconstructed = sum(
        float(row.get("shapley_risk_logit_contribution") or 0.0)
        for row in marginals
    )
    efficiency_error = abs(reconstructed - combined_logit_contribution)
    coalition_efficiency = 1.0 - min(
        1.0, efficiency_error / max(1e-9, abs(combined_logit_contribution))
    )
    score = (
        0.25 * baseline_adequacy
        + 0.20 * candidate_coverage
        + 0.20 * intervention_realism
        + 0.20 * coalition_direction_consistency
        + 0.15 * coalition_efficiency
    )
    grade = "high" if score >= 0.80 else "medium" if score >= 0.60 else "low"
    limitations: list[str] = []
    if baseline_adequacy < 0.80:
        limitations.append("선택된 일부 필드의 정상 기준선 표본이 20건보다 적다.")
    if candidate_coverage < 0.80:
        limitations.append("후보 풀 영향의 일부가 최종 조합 분석에서 제외됐다.")
    if intervention_realism < 1.0:
        limitations.append("정상 기준선이 없는 일부 필드는 삭제 개입을 사용했다.")
    if coalition_direction_consistency < 0.80:
        limitations.append("필드 영향 방향이 다른 필드 조합에 따라 달라졌다.")
    return {
        "score": score,
        "grade": grade,
        "grade_label": {"high": "높음", "medium": "보통", "low": "낮음"}[grade],
        "components": {
            "baseline_adequacy": baseline_adequacy,
            "candidate_coverage": candidate_coverage,
            "normal_reference_ratio": intervention_realism,
            "coalition_direction_consistency": coalition_direction_consistency,
            "coalition_efficiency": coalition_efficiency,
        },
        "limitations": limitations,
        "meaning": "모델 정확도가 아니라 현재 판정 근거의 재현성과 기준선 품질",
    }


def _unavailable(reason: str, candidate_fields: Sequence[str] = ()) -> dict:
    return {
        "method": "counterfactual_coalition_v3",
        "status": "unavailable",
        "unavailable_reason": reason,
        "candidate_fields": list(candidate_fields),
        "candidate_discovery": None,
        "single_field_tests": [],
        "pair_tests": [],
        "coalition_analysis": None,
        "explanation_confidence": None,
        "primary_evidence": None,
    }


def analyze_counterfactuals(
    predictor: CounterfactualPredictor,
    event: Mapping[str, object],
    prediction: Mapping[str, object],
    *,
    candidate_limit: int = 4,
    candidate_pool_limit: int | None = None,
    pair_candidate_limit: int = 4,
    min_probability_drop: float = 0.005,
) -> dict:
    """Discover candidate fields and measure exact influence over the final set."""

    registry = getattr(predictor, "baseline_registry", None)
    try:
        baseline_risk = _raw_risk(prediction)
    except ValueError:
        return _unavailable("raw_model_risk_probability_missing")

    final_limit = max(1, min(int(candidate_limit), 8))
    pool_limit = (
        final_limit
        if candidate_pool_limit is None
        else max(final_limit, min(int(candidate_pool_limit), 16))
    )
    candidate_rows = _candidate_rows(event, prediction, registry, pool_limit)
    interventions: dict[str, dict] = {}
    usable_rows: list[dict] = []
    for row in candidate_rows:
        field = str(row["field"])
        changed_event = deepcopy(dict(event))
        intervention = _intervene(changed_event, field, registry)
        if intervention is not None and intervention["changed"]:
            interventions[field] = intervention
            usable_rows.append(row)
    if not usable_rows:
        return _unavailable("no_explainable_candidate_fields")

    try:
        candidates, screening = _screen_candidate_pool(
            predictor,
            event,
            prediction,
            usable_rows,
            final_limit,
            registry,
        )
        selected_set = set(candidates)
        for row in usable_rows:
            field = str(row["field"])
            row["selected_for_coalition"] = field in selected_set
            parts = [
                f"모델 기여도 {float(row.get('absolute_model_contribution') or 0.0):.4f}",
                f"AI 가중치 {float(row.get('attention_weight') or 0.0) * 100.0:.2f}%",
                f"기준선 이탈도 {float(row.get('baseline_anomaly') or 0.0) * 100.0:.2f}%",
            ]
            if screening["performed"]:
                parts.extend(
                    [
                        f"단독 정상화 효과 {float(row.get('screen_single_effect') or 0.0):.4f}",
                        f"최대 쌍 상호작용 {float(row.get('screen_pair_interaction') or 0.0):.4f}",
                    ]
                )
            row["selection_reason"] = (
                ", ".join(parts)
                + ("를 종합해 최종 조합 분석 대상으로 선택했다."
                   if field in selected_set
                   else "를 종합했으나 최종 조합 분석에서는 제외했다.")
            )
        kept_sets = _all_subsets(candidates)
        scores, coalition_forward_calls = _evaluate_kept_sets(
            predictor,
            event,
            prediction,
            candidates,
            kept_sets,
            registry,
        )
    except ValueError:
        return _unavailable(
            "coalition_reinference_missing_raw_risk",
            [str(row["field"]) for row in usable_rows],
        )

    full = frozenset(candidates)
    baseline_logit = scores[full]["risk_logit"]
    single_tests: list[dict] = []
    for field in candidates:
        changed = scores[full - {field}]
        single_tests.append(
            {
                **interventions[field],
                "counterfactual_raw_risk_probability": changed["risk_probability"],
                "risk_probability_drop": baseline_risk - changed["risk_probability"],
                "risk_logit_drop": baseline_logit - changed["risk_logit"],
            }
        )

    single_by_field = {row["field"]: row for row in single_tests}
    pair_fields = candidates[: max(2, min(pair_candidate_limit, len(candidates)))]
    pair_tests: list[dict] = []
    for first, second in combinations(pair_fields, 2):
        changed = scores[full - {first, second}]
        probability_drop = baseline_risk - changed["risk_probability"]
        logit_drop = baseline_logit - changed["risk_logit"]
        strongest_single_drop = max(
            float(single_by_field[first]["risk_probability_drop"]),
            float(single_by_field[second]["risk_probability_drop"]),
        )
        pair_tests.append(
            {
                "fields": [first, second],
                "methods": [interventions[first]["method"], interventions[second]["method"]],
                "counterfactual_raw_risk_probability": changed["risk_probability"],
                "risk_probability_drop": probability_drop,
                "risk_logit_drop": logit_drop,
                "additional_drop_over_strongest_single": (
                    probability_drop - strongest_single_drop
                ),
                "interaction_probability_effect":
                float(single_by_field[first]["risk_probability_drop"])
                + float(single_by_field[second]["risk_probability_drop"])
                - probability_drop,
                "interaction_logit_effect":
                float(single_by_field[first]["risk_logit_drop"])
                + float(single_by_field[second]["risk_logit_drop"])
                - logit_drop,
            }
        )

    single_tests.sort(key=lambda row: row["risk_logit_drop"], reverse=True)
    pair_tests.sort(key=lambda row: row["risk_logit_drop"], reverse=True)
    best_single = next(
        (row for row in single_tests if row["risk_probability_drop"] >= min_probability_drop),
        None,
    )
    best_pair = next(
        (row for row in pair_tests if row["risk_probability_drop"] >= min_probability_drop),
        None,
    )
    if best_pair is not None and (
        best_single is None
        or best_pair["additional_drop_over_strongest_single"] >= min_probability_drop
    ):
        primary = {"type": "field_pair", **best_pair}
    elif best_single is not None:
        primary = {"type": "single_field", **best_single}
    else:
        primary = None

    empty = scores[frozenset()]
    combined_logit = baseline_logit - empty["risk_logit"]
    marginals = _field_marginals(candidates, scores)
    interactions = _pair_interactions(candidates, scores)
    confidence = _explanation_confidence(
        usable_rows,
        candidates,
        interventions,
        single_tests,
        marginals,
        combined_logit,
    )
    total_reinferences = int(screening["model_reinferences"]) + max(0, len(scores) - 1)
    total_forward_calls = int(screening["model_forward_calls"]) + coalition_forward_calls
    coalition_analysis = {
        "method": "exact_subset_shapley_v2",
        "value_space": "raw_risk_logit",
        "candidate_count": len(candidates),
        "evaluated_coalitions": len(scores),
        "model_reinferences": total_reinferences,
        "model_forward_calls": total_forward_calls,
        "batched_reinference": total_forward_calls < total_reinferences,
        "reference_raw_risk_probability": empty["risk_probability"],
        "reference_raw_risk_logit": empty["risk_logit"],
        "combined_risk_probability_contribution": baseline_risk - empty["risk_probability"],
        "combined_risk_logit_contribution": combined_logit,
        "field_marginal_contributions": marginals,
        "pair_interactions": interactions,
    }
    candidate_discovery = {
        "method": "hybrid_attention_anomaly_contribution_with_pair_screening_v1",
        "pool_size": len(usable_rows),
        "pool_fields": [str(row["field"]) for row in usable_rows],
        "selected_fields": list(candidates),
        "selection_metrics": [
            "model_risk_contribution",
            "attention_weight",
            "baseline_anomaly",
            "single_normalization_effect",
            "pair_interaction_effect",
        ],
        "candidate_rows": usable_rows,
        "screening": screening,
    }
    return {
        "method": "counterfactual_coalition_v3",
        "status": "available",
        "baseline_raw_risk_probability": baseline_risk,
        "candidate_fields": list(candidates),
        "candidate_discovery": candidate_discovery,
        "single_field_tests": single_tests,
        "pair_tests": pair_tests,
        "coalition_analysis": coalition_analysis,
        "explanation_confidence": confidence,
        "primary_evidence": primary,
    }
