from ztna_ueba.counterfactual import analyze_counterfactuals
from ztna_ueba.baseline import BaselineRegistry
from ztna_ueba.tokenizer import DEFAULT_EXCLUDED_FIELDS, NORMAL_REFERENCE_FIELDS_KEY


class InteractionPredictor:
    baseline_registry = None

    def predict(self, payload, *, top_k=10):
        first = "first_signal" in payload
        second = "second_signal" in payload
        if first and second:
            risk = 0.90
        elif first or second:
            risk = 0.40
        else:
            risk = 0.20
        return {
            "raw_model_risk_probability": risk,
            "risk_score": risk,
            "top_fields": [
                {
                    "event_index": 0,
                    "field": field,
                    "risk_logit_contribution": 1.0,
                }
                for field in ("first_signal", "second_signal")
                if field in payload
            ][:top_k],
        }


def test_counterfactual_reinference_finds_field_pair_effect():
    predictor = InteractionPredictor()
    event = {"first_signal": "new", "second_signal": "new", "context": "vpn"}
    prediction = predictor.predict(event)

    result = analyze_counterfactuals(
        predictor,
        event,
        prediction,
        candidate_limit=2,
        pair_candidate_limit=2,
    )

    assert len(result["single_field_tests"]) == 2
    assert len(result["pair_tests"]) == 1
    assert result["method"] == "counterfactual_coalition_v3"
    assert result["status"] == "available"
    assert result["coalition_analysis"]["evaluated_coalitions"] == 4
    assert result["coalition_analysis"]["model_reinferences"] == 3
    marginals = result["coalition_analysis"]["field_marginal_contributions"]
    assert len(marginals) == 2
    assert sum(row["shapley_risk_logit_contribution"] for row in marginals) == (
        result["coalition_analysis"]["combined_risk_logit_contribution"]
    )
    interaction = result["coalition_analysis"]["pair_interactions"][0]
    assert interaction["interaction_type"] == "synergy"
    assert interaction["shapley_interaction_logit"] > 0.0
    assert result["primary_evidence"]["type"] == "field_pair"
    assert result["primary_evidence"]["fields"] == [
        "first_signal",
        "second_signal",
    ]
    assert result["primary_evidence"]["counterfactual_raw_risk_probability"] == 0.2
    assert result["primary_evidence"]["risk_probability_drop"] == 0.7
    assert result["primary_evidence"]["interaction_logit_effect"] > 0.0
    assert result["explanation_confidence"]["meaning"].startswith("모델 정확도가 아니라")


def test_counterfactual_reinference_does_not_relabel_calibrated_risk_as_raw():
    predictor = InteractionPredictor()
    event = {"first_signal": "new", "second_signal": "new"}
    prediction = predictor.predict(event)
    prediction.pop("raw_model_risk_probability")

    result = analyze_counterfactuals(predictor, event, prediction)

    assert result["status"] == "unavailable"
    assert result["unavailable_reason"] == "raw_model_risk_probability_missing"
    assert "baseline_raw_risk_probability" not in result
    assert result["primary_evidence"] is None


class ThreeFieldPredictor:
    baseline_registry = None

    def predict(self, payload, *, top_k=10):
        fields = ("first_signal", "second_signal", "third_signal")
        present = sum(field in payload for field in fields)
        risk = 0.90 if present == 3 else 0.55 if present == 2 else 0.20
        return {
            "raw_model_risk_probability": risk,
            "top_fields": [
                {
                    "event_index": 0,
                    "field": field,
                    "risk_logit_contribution": 1.0,
                }
                for field in fields
                if field in payload
            ][:top_k],
        }


def test_exact_coalitions_include_three_field_contexts():
    predictor = ThreeFieldPredictor()
    event = {
        "first_signal": "new",
        "second_signal": "new",
        "third_signal": "new",
    }

    result = analyze_counterfactuals(
        predictor,
        event,
        predictor.predict(event),
        candidate_limit=3,
        pair_candidate_limit=3,
    )

    analysis = result["coalition_analysis"]
    assert analysis["candidate_count"] == 3
    assert analysis["evaluated_coalitions"] == 8
    assert analysis["model_reinferences"] == 7
    assert len(analysis["field_marginal_contributions"]) == 3
    assert len(analysis["pair_interactions"]) == 3
    assert abs(
        sum(
            row["shapley_risk_logit_contribution"]
            for row in analysis["field_marginal_contributions"]
        )
        - analysis["combined_risk_logit_contribution"]
    ) < 1e-12


class HiddenPairPredictor:
    baseline_registry = None

    def predict(self, payload, *, top_k=10):
        hidden_first = "hidden_first" in payload
        hidden_second = "hidden_second" in payload
        risk = 0.95 if hidden_first and hidden_second else 0.45 if hidden_first or hidden_second else 0.10
        ordered = (
            ("distractor_a", 1.00),
            ("distractor_b", 0.90),
            ("distractor_c", 0.80),
            ("distractor_d", 0.70),
            ("hidden_first", 0.10),
            ("hidden_second", 0.09),
        )
        return {
            "raw_model_risk_probability": risk,
            "top_fields": [
                {
                    "event_index": 0,
                    "field": field,
                    "weight": contribution,
                    "risk_logit_contribution": contribution,
                }
                for field, contribution in ordered
                if field in payload
            ][:top_k],
        }


def test_pair_screening_recovers_joint_signal_outside_original_top_four():
    predictor = HiddenPairPredictor()
    event = {
        "distractor_a": "x",
        "distractor_b": "x",
        "distractor_c": "x",
        "distractor_d": "x",
        "hidden_first": "new",
        "hidden_second": "new",
    }

    result = analyze_counterfactuals(
        predictor,
        event,
        predictor.predict(event),
        candidate_limit=4,
        candidate_pool_limit=6,
        pair_candidate_limit=4,
    )

    assert result["candidate_discovery"]["screening"]["performed"] is True
    assert {"hidden_first", "hidden_second"}.issubset(result["candidate_fields"])
    assert result["primary_evidence"]["type"] == "field_pair"
    assert set(result["primary_evidence"]["fields"]) == {
        "hidden_first",
        "hidden_second",
    }


class NormalReferencePredictor:
    def __init__(self):
        normals = [
            {
                "dataset": "company",
                "source_type": "vpn",
                "event_type": "session",
                "geo_zone": "seoul",
            }
            for _ in range(24)
        ]
        self.baseline_registry = BaselineRegistry.fit(normals, DEFAULT_EXCLUDED_FIELDS)
        self.counterfactual_payloads = []

    def predict(self, payload, *, top_k=10):
        references = set(payload.get(NORMAL_REFERENCE_FIELDS_KEY, ()))
        if references:
            self.counterfactual_payloads.append(dict(payload))
        risk = 0.10 if "geo_zone" in references else 0.90
        return {
            "raw_model_risk_probability": risk,
            "top_fields": [
                {
                    "event_index": 0,
                    "field": "geo_zone",
                    "weight": 0.8,
                    "risk_logit_contribution": 2.0,
                }
            ],
        }


def test_categorical_intervention_keeps_field_and_neutralizes_baseline_features():
    predictor = NormalReferencePredictor()
    event = {
        "dataset": "company",
        "source_type": "vpn",
        "event_type": "session",
        "geo_zone": "unknown-region",
    }

    result = analyze_counterfactuals(
        predictor,
        event,
        predictor.predict(event),
        candidate_limit=1,
    )

    test = result["single_field_tests"][0]
    assert test["method"] == "neutralize_with_normal_baseline"
    assert test["uses_normal_reference"] is True
    assert test["counterfactual_raw_risk_probability"] == 0.10
    assert predictor.counterfactual_payloads
    assert all(payload["geo_zone"] == "unknown-region" for payload in predictor.counterfactual_payloads)
    assert all(
        "geo_zone" in payload[NORMAL_REFERENCE_FIELDS_KEY]
        for payload in predictor.counterfactual_payloads
    )
