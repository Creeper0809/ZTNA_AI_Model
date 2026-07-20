"""Convert faithful model contributions into operator-facing explanations."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import torch


BNDT_KEYWORDS = {
    "B": ("user", "auth", "login", "session", "action", "access", "behavior"),
    "N": ("ip", "port", "protocol", "network", "packet", "byte", "flow", "service"),
    "D": ("device", "host", "endpoint", "os", "edr", "posture", "mdm"),
    "T": ("attack", "threat", "malware", "indicator", "signature", "cve", "exploit"),
}


def bndt_bucket(field_name: str) -> str:
    normalized = field_name.lower()
    for bucket, keywords in BNDT_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return bucket
    return "C"


def explain_request(
    field_names: list[list[str]],
    event_weights: torch.Tensor,
    field_weights: torch.Tensor,
    field_contributions: torch.Tensor,
    top_k: int = 10,
) -> dict:
    rows = []
    grouped = defaultdict(float)
    for event_index, names in enumerate(field_names):
        for field_index, name in enumerate(names):
            weight = float(event_weights[event_index] * field_weights[event_index, field_index])
            contribution = float(field_contributions[event_index, field_index])
            bucket = bndt_bucket(name)
            grouped[bucket] += contribution
            rows.append(
                {
                    "event_index": event_index,
                    "field": name,
                    "weight": weight,
                    "risk_logit_contribution": contribution,
                    "bndt_explanation_bucket": bucket,
                }
            )
    rows.sort(key=lambda row: abs(row["risk_logit_contribution"]), reverse=True)
    return {
        "top_fields": rows[:top_k],
        "bndt_posthoc_contributions": dict(sorted(grouped.items())),
    }
