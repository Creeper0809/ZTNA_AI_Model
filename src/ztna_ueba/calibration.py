"""Normal-only score calibration for portable event risk."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ScoreReference:
    count: int
    q50: float
    q95: float
    q99: float
    scale: float


class NormalScoreCalibrator:
    """Map model scores to risk relative to a profile's normal-only tail."""

    def __init__(
        self,
        profiles: dict[str, ScoreReference] | None = None,
        minimum_records: int = 20,
        slope: float = 4.0,
    ):
        self.profiles = profiles or {}
        self.minimum_records = minimum_records
        self.slope = slope

    @classmethod
    def fit(
        cls,
        profile_keys: Iterable[str],
        scores: Iterable[float],
        minimum_records: int = 20,
        slope: float = 4.0,
    ) -> "NormalScoreCalibrator":
        grouped: dict[str, list[float]] = defaultdict(list)
        for profile_key, score in zip(profile_keys, scores, strict=True):
            numeric_score = float(score)
            if math.isfinite(numeric_score):
                grouped[str(profile_key)].append(numeric_score)

        profiles = {}
        for profile_key, values in grouped.items():
            if len(values) < minimum_records:
                continue
            array = np.asarray(values, dtype=np.float64)
            q50, q75, q95, q99 = np.quantile(array, [0.50, 0.75, 0.95, 0.99])
            robust_scale = max(float(q95 - q50), float(q75 - q50), 1e-4)
            profiles[profile_key] = ScoreReference(
                count=len(values),
                q50=float(q50),
                q95=float(q95),
                q99=float(q99),
                scale=robust_scale,
            )
        return cls(
            profiles=profiles,
            minimum_records=minimum_records,
            slope=slope,
        )

    def is_ready(self, profile_key: str) -> bool:
        return profile_key in self.profiles

    def normalize(self, profile_key: str, score: float) -> float | None:
        reference = self.profiles.get(profile_key)
        if reference is None:
            return None
        numeric_score = float(score)
        if numeric_score <= reference.q95:
            lower_scale = max(reference.q95 - reference.q50, 1e-4)
            margin = self.slope * (numeric_score - reference.q95) / lower_scale
        else:
            # q95 starts review/step-up (risk 0.5), while the normal q99 tail
            # maps to strong restriction territory (risk 0.9).
            upper_scale = max(reference.q99 - reference.q95, 1e-4)
            margin = math.log(9.0) * (numeric_score - reference.q95) / upper_scale
        margin = max(-30.0, min(30.0, margin))
        return 1.0 / (1.0 + math.exp(-margin))

    def normalize_batch(
        self,
        profile_keys: Sequence[Sequence[str]],
        event_scores: torch.Tensor,
        event_weights: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = torch.zeros_like(event_scores)
        ready = torch.zeros_like(event_mask)
        for batch_index, request_profiles in enumerate(profile_keys):
            for event_index, profile_key in enumerate(request_profiles):
                value = self.normalize(
                    profile_key,
                    float(event_scores[batch_index, event_index].detach().cpu()),
                )
                if value is not None:
                    normalized[batch_index, event_index] = value
                    ready[batch_index, event_index] = True
        effective_weights = event_weights * ready.to(event_weights.dtype)
        weight_sum = effective_weights.sum(dim=1)
        request_risk = (effective_weights * normalized).sum(dim=1) / weight_sum.clamp_min(1e-12)
        request_ready = (ready | ~event_mask).all(dim=1) & (weight_sum > 0)
        return request_risk, request_ready

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "minimum_records": self.minimum_records,
            "slope": self.slope,
            "profiles": {
                profile_key: asdict(reference)
                for profile_key, reference in self.profiles.items()
            },
        }

    @classmethod
    def from_dict(cls, value: dict) -> "NormalScoreCalibrator":
        return cls(
            profiles={
                profile_key: ScoreReference(**reference)
                for profile_key, reference in value.get("profiles", {}).items()
            },
            minimum_records=int(value.get("minimum_records", 20)),
            slope=float(value.get("slope", 4.0)),
        )
