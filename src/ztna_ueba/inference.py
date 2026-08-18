"""Reusable Trust Score inference engine shared by the CLI and PoC API."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

from .baseline import BaselineRegistry, SCOPE_ORDER
from .calibration import NormalScoreCalibrator
from .explain import explain_request
from .model import HierarchicalFieldAttention, ModelConfig
from .policy import map_risk_to_policy
from .tokenizer import (
    NORMAL_REFERENCE_FIELDS_KEY,
    FieldTokenizer,
    TokenizerConfig,
    collate_requests,
)


RUNTIME_METADATA_FIELDS = frozenset(
    {"event_id", "request_id", "trace_id", NORMAL_REFERENCE_FIELDS_KEY}
)


class TrustPredictor:
    """Load a checkpoint once and score arbitrary key-value events repeatedly."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        tokenizer_values = dict(checkpoint["tokenizer_config"])
        tokenizer_values["excluded_fields"] = (
            frozenset(tokenizer_values["excluded_fields"]) | RUNTIME_METADATA_FIELDS
        )
        if "profile_fields" in tokenizer_values:
            tokenizer_values["profile_fields"] = tuple(tokenizer_values["profile_fields"])
        baseline_values = checkpoint.get("baseline_registry")
        self.baseline_registry = (
            BaselineRegistry.from_dict(baseline_values) if baseline_values else None
        )
        calibrator_values = checkpoint.get("score_calibrator")
        self.score_calibrator = (
            NormalScoreCalibrator.from_dict(calibrator_values) if calibrator_values else None
        )
        self.tokenizer_config = TokenizerConfig(**tokenizer_values)
        self.tokenizer = FieldTokenizer(
            self.tokenizer_config, baseline_registry=self.baseline_registry
        )
        self.model = HierarchicalFieldAttention(ModelConfig(**checkpoint["model_config"]))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.threshold = float(checkpoint["threshold"])
        self.seen_fields = set(checkpoint.get("seen_fields", []))
        self.seen_datasets = set(checkpoint.get("seen_datasets", []))
        self.seen_source_types = set(checkpoint.get("seen_source_types", []))
        self.model_version = str(
            checkpoint.get("model_version")
            or checkpoint.get("run_id")
            or Path(self.checkpoint_path).parent.name
        )

    def predict(
        self,
        payload: Mapping[str, object] | Sequence[Mapping[str, object]],
        *,
        top_k: int = 10,
    ) -> dict:
        records = payload["events"] if isinstance(payload, Mapping) and "events" in payload else payload
        record_list = [records] if isinstance(records, Mapping) else list(records)
        tokenized = self.tokenizer.tokenize_request(records)
        batch = collate_requests([tokenized])
        tensor_batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            output = self.model(tensor_batch)

        calibrated_risk = output["risk_probability"]
        score_calibration_ready = not self.tokenizer_config.portable_mode
        if self.score_calibrator is not None:
            calibrated_risk, calibration_ready = self.score_calibrator.normalize_batch(
                batch["profile_keys"],
                output["event_anomaly_score"],
                output["event_weights"],
                tensor_batch["event_mask"],
            )
            score_calibration_ready = bool(calibration_ready[0].cpu())

        field_names = batch["field_names"][0]
        explanation = explain_request(
            field_names,
            output["event_weights"][0].cpu(),
            output["field_weights"][0].cpu(),
            output["field_contributions"][0].cpu(),
            output["field_weight_logits"][0].cpu(),
            top_k=top_k,
        )
        current = {name for event in field_names for name in event}
        novelty_ratio = len(current - self.seen_fields) / max(1, len(current))
        current_datasets = {
            str(record["dataset"])
            for record in record_list
            if record.get("dataset") not in {None, ""}
        }
        current_source_types = {
            str(record["source_type"])
            for record in record_list
            if record.get("source_type") not in {None, ""}
        }
        unseen_datasets = sorted(current_datasets - self.seen_datasets)
        unseen_source_types = sorted(current_source_types - self.seen_source_types)
        source_metadata_missing = not current_datasets and not current_source_types
        baseline_coverages = (
            [
                self.baseline_registry.coverage(record, self.tokenizer_config.excluded_fields)
                for record in record_list
            ]
            if self.baseline_registry is not None
            else []
        )
        minimum_baseline_coverage = (
            min(item["coverage"] for item in baseline_coverages) if baseline_coverages else 0.0
        )
        baseline_ready = bool(baseline_coverages) and all(
            item["profile_known"] and item["coverage"] >= 0.80
            for item in baseline_coverages
        )
        attention_concentration = float(output["attention_concentration"][0].cpu())
        confidence_margin = (
            abs(2.0 * float(calibrated_risk[0].cpu()) - 1.0)
            if score_calibration_ready
            else abs(2.0 * float(output["risk_probability"][0].cpu()) - 1.0)
        )
        raw_confidence = 0.5 * attention_concentration + 0.5 * confidence_margin
        ood_factor = (
            0.5 if unseen_datasets or unseen_source_types or source_metadata_missing else 1.0
        )
        if self.tokenizer_config.portable_mode and (
            not baseline_ready or not score_calibration_ready
        ):
            ood_factor *= 0.25
        confidence = raw_confidence * (1.0 - 0.5 * novelty_ratio) * ood_factor
        risk_probability = (
            float(calibrated_risk[0].cpu()) if score_calibration_ready else None
        )
        shadow_mode_required = (
            confidence < 0.40
            or novelty_ratio > 0.30
            or bool(unseen_datasets)
            or bool(unseen_source_types)
            or source_metadata_missing
            or (
                self.tokenizer_config.portable_mode
                and (not baseline_ready or not score_calibration_ready)
            )
        )
        hierarchical_scope_counts: dict[str, int] = {}
        for coverage in baseline_coverages:
            for scope, count in coverage.get("field_scope_counts", {}).items():
                hierarchical_scope_counts[scope] = (
                    hierarchical_scope_counts.get(scope, 0) + int(count)
                )
        return {
            "model_version": self.model_version,
            "trust_score": (
                100.0 * (1.0 - risk_probability) if risk_probability is not None else None
            ),
            "risk_score": risk_probability,
            "raw_model_risk_probability": float(output["risk_probability"][0].cpu()),
            "monotonic_baseline": bool(self.model.config.monotonic_baseline),
            "portable_mode": bool(self.tokenizer_config.portable_mode),
            "model_dimension": int(self.model.config.model_dim),
            "field_context_layers": int(self.model.config.field_layers),
            "confidence": confidence,
            "novel_field_ratio": novelty_ratio,
            "unseen_datasets": unseen_datasets,
            "unseen_source_types": unseen_source_types,
            "source_metadata_missing": source_metadata_missing,
            "baseline_ready": baseline_ready,
            "score_calibration_ready": score_calibration_ready,
            "minimum_baseline_coverage": minimum_baseline_coverage,
            "baseline_coverages": baseline_coverages,
            "ueba_baseline": {
                "mode": "hierarchical_field_level",
                "scope_order": list(SCOPE_ORDER),
                "field_scope_counts": dict(sorted(hierarchical_scope_counts.items())),
                "hierarchical_profiles_available": bool(
                    self.baseline_registry
                    and any(self.baseline_registry.hierarchical_profiles.values())
                ),
            },
            "decision_threshold": self.threshold,
            "shadow_mode_required": shadow_mode_required,
            "policy": map_risk_to_policy(
                risk_probability
                if risk_probability is not None
                else float(output["risk_probability"][0].cpu()),
                shadow_mode_required,
            ),
            "source_weights": output["event_weights"][0, : len(field_names)].cpu().tolist(),
            **explanation,
        }

    def predict_raw_batch(
        self, payloads: Sequence[Mapping[str, object]]
    ) -> list[float]:
        """Evaluate counterfactual events in one model forward pass."""

        records = list(payloads)
        if not records:
            return []
        requests = [self.tokenizer.tokenize_request(record) for record in records]
        batch = collate_requests(requests)
        tensor_batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            output = self.model(tensor_batch)
        return [float(value) for value in output["risk_probability"].cpu().tolist()]
