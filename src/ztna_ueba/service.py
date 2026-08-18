"""Application service for explainable Trust Score assessment and timelines."""

from __future__ import annotations

from typing import Mapping, Protocol

from .baseline import BaselineRegistry
from .counterfactual import analyze_counterfactuals
from .operator_explain import build_operator_explanation
from .timeline import TimelineStore


class Predictor(Protocol):
    baseline_registry: BaselineRegistry | None

    def predict(self, payload: Mapping[str, object], *, top_k: int = 10) -> dict: ...


class ExplainableTrustService:
    """Connect field-attention inference to readable logs and actor history."""

    def __init__(
        self,
        predictor: Predictor,
        store: TimelineStore,
        *,
        evidence_limit: int = 5,
        counterfactual_candidate_limit: int = 6,
        counterfactual_candidate_pool_limit: int = 12,
        counterfactual_pair_limit: int = 6,
    ) -> None:
        self.predictor = predictor
        self.store = store
        self.evidence_limit = evidence_limit
        self.counterfactual_candidate_limit = counterfactual_candidate_limit
        self.counterfactual_candidate_pool_limit = counterfactual_candidate_pool_limit
        self.counterfactual_pair_limit = counterfactual_pair_limit

    def _prediction_top_k(self) -> int:
        return max(
            16,
            self.evidence_limit * 2,
            self.counterfactual_candidate_pool_limit * 2,
        )

    def assess(self, event: Mapping[str, object], *, persist: bool = True) -> dict:
        """Run the legacy synchronous path including exact coalition analysis."""
        if not isinstance(event, Mapping):
            raise TypeError("event must be a JSON object")
        prediction = self.predictor.predict(event, top_k=self._prediction_top_k())
        counterfactual = analyze_counterfactuals(
            self.predictor,
            event,
            prediction,
            candidate_limit=self.counterfactual_candidate_limit,
            candidate_pool_limit=self.counterfactual_candidate_pool_limit,
            pair_candidate_limit=self.counterfactual_pair_limit,
        )
        explanation = build_operator_explanation(
            event,
            prediction,
            getattr(self.predictor, "baseline_registry", None),
            top_k=self.evidence_limit,
            counterfactual=counterfactual,
        )
        return self._response(
            event,
            prediction,
            explanation,
            persist=persist,
            explanation_status="completed",
        )

    def assess_fast(
        self,
        event: Mapping[str, object],
        *,
        persist: bool = True,
        explanation_status: str = "pending",
    ) -> dict:
        """Return the Trust Score without running combinatorial explanations.

        The response still includes baseline-relative field evidence. Exact subset
        influence checks can be attached later by :meth:`complete_explanation`.
        """

        response, _ = self.prepare_fast(
            event,
            persist=persist,
            explanation_status=explanation_status,
        )
        return response

    def prepare_fast(
        self,
        event: Mapping[str, object],
        *,
        persist: bool = True,
        explanation_status: str = "pending",
    ) -> tuple[dict, dict]:
        """Return a fast public response and reusable internal prediction."""

        if not isinstance(event, Mapping):
            raise TypeError("event must be a JSON object")
        prediction = self.predictor.predict(event, top_k=self._prediction_top_k())
        explanation = build_operator_explanation(
            event,
            prediction,
            getattr(self.predictor, "baseline_registry", None),
            top_k=self.evidence_limit,
        )
        response = self._response(
            event,
            prediction,
            explanation,
            persist=persist,
            explanation_status=explanation_status,
        )
        return response, prediction

    def complete_explanation(
        self,
        event: Mapping[str, object],
        *,
        prediction: Mapping[str, object] | None = None,
    ) -> dict:
        """Compute and persist the expensive exact coalition explanation."""

        if not isinstance(event, Mapping):
            raise TypeError("event must be a JSON object")
        base_prediction = (
            dict(prediction)
            if prediction is not None
            else self.predictor.predict(
                event, top_k=self._prediction_top_k()
            )
        )
        counterfactual = analyze_counterfactuals(
            self.predictor,
            event,
            base_prediction,
            candidate_limit=self.counterfactual_candidate_limit,
            candidate_pool_limit=self.counterfactual_candidate_pool_limit,
            pair_candidate_limit=self.counterfactual_pair_limit,
        )
        explanation = build_operator_explanation(
            event,
            base_prediction,
            getattr(self.predictor, "baseline_registry", None),
            top_k=self.evidence_limit,
            counterfactual=counterfactual,
        )
        event_id = str(explanation["event"]["event_id"])
        self.store.update_explanation(event_id, explanation, status="completed")
        return explanation

    def _response(
        self,
        event: Mapping[str, object],
        prediction: Mapping[str, object],
        explanation: Mapping[str, object],
        *,
        persist: bool,
        explanation_status: str,
    ) -> dict:
        event_id = str(explanation["event"]["event_id"])
        if persist:
            event_id = self.store.record(
                event,
                prediction,
                explanation,
                explanation_status=explanation_status,
            )
        return {
            "event_id": event_id,
            "actor_id": explanation["event"]["actor_id"],
            "explanation_status": explanation_status,
            "event": explanation["event"],
            "decision": {
                "trust_score": prediction.get("trust_score"),
                "risk_score": prediction.get("risk_score"),
                "raw_model_risk_probability": prediction.get(
                    "raw_model_risk_probability"
                ),
                "confidence": prediction.get("confidence"),
                "policy": prediction.get("policy"),
                "shadow_mode_required": prediction.get("shadow_mode_required"),
            },
            "readable_log": explanation["readable_log"],
            "evidence": {
                "risk_increasing": explanation["risk_increasing_evidence"],
                "risk_decreasing": explanation["risk_decreasing_evidence"],
                "influence_check": explanation["counterfactual_evidence"],
            },
            "audit": {
                **explanation["audit"],
                "model_version": prediction.get("model_version"),
                "baseline_ready": prediction.get("baseline_ready"),
                "ueba_baseline": prediction.get("ueba_baseline"),
                "score_calibration_ready": prediction.get("score_calibration_ready"),
                "raw_event_reference": f"timeline://events/{event_id}",
            },
        }

    def actor_timeline(
        self,
        actor_id: str,
        *,
        suspicious_only: bool = True,
        limit: int = 100,
        start: str | None = None,
        end: str | None = None,
    ) -> dict:
        return self.store.actor_timeline(
            actor_id,
            suspicious_only=suspicious_only,
            limit=limit,
            start=start,
            end=end,
        )

    def raw_event(self, event_id: str) -> dict | None:
        return self.store.raw_event(event_id)
