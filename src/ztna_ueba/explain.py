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
    field_weight_logits: torch.Tensor | None = None,
    top_k: int = 10,
) -> dict:
    rows = []
    grouped = defaultdict(float)
    for event_index, names in enumerate(field_names):
        for field_index, name in enumerate(names):
            weight = float(event_weights[event_index] * field_weights[event_index, field_index])
            event_weight = float(event_weights[event_index])
            field_weight = float(field_weights[event_index, field_index])
            importance_logit = (
                float(field_weight_logits[event_index, field_index])
                if field_weight_logits is not None
                else None
            )
            contribution = float(field_contributions[event_index, field_index])
            bucket = bndt_bucket(name)
            grouped[bucket] += contribution
            rows.append(
                {
                    "event_index": event_index,
                    "field": name,
                    "event_field_count": len(names),
                    "weight": weight,
                    "event_weight": event_weight,
                    "field_weight_within_event": field_weight,
                    "field_importance_logit": importance_logit,
                    "risk_logit_contribution": contribution,
                    "bndt_explanation_bucket": bucket,
                }
            )
    weight_order = sorted(
        range(len(rows)), key=lambda index: rows[index]["weight"], reverse=True
    )
    for rank, index in enumerate(weight_order, start=1):
        rows[index]["field_weight_rank"] = rank
        rows[index]["request_field_count"] = len(rows)
    contribution_order = sorted(
        range(len(rows)),
        key=lambda index: abs(rows[index]["risk_logit_contribution"]),
        reverse=True,
    )
    for rank, index in enumerate(contribution_order, start=1):
        rows[index]["risk_contribution_rank"] = rank
    rows.sort(key=lambda row: row["risk_contribution_rank"])
    return {
        "top_fields": rows[:top_k],
        "bndt_posthoc_contributions": dict(sorted(grouped.items())),
    }
