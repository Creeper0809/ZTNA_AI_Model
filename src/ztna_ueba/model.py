"""Hierarchical set-attention model for field and source weighting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    name_buckets: int = 32768
    value_buckets: int = 131072
    type_count: int = 6
    embedding_dim: int = 48
    model_dim: int = 128
    heads: int = 4
    field_layers: int = 2
    event_layers: int = 1
    dropout: float = 0.10
    numeric_feature_dim: int = 8
    monotonic_baseline: bool = False


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    minimum = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~mask, minimum)
    weights = torch.softmax(masked, dim=dim)
    return weights.masked_fill(~mask, 0.0)


def masked_mean_embedding(
    embedding: nn.Embedding, ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    vectors = embedding(ids)
    expanded = mask.unsqueeze(-1).to(vectors.dtype)
    total = (vectors * expanded).sum(dim=-2)
    count = expanded.sum(dim=-2).clamp_min(1.0)
    return total / count


class HierarchicalFieldAttention(nn.Module):
    """Learn field weights, then source/event weights, without positional inputs."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.name_embedding = nn.Embedding(
            cfg.name_buckets + 1, cfg.embedding_dim, padding_idx=0
        )
        self.value_embedding = nn.Embedding(
            cfg.value_buckets + 1, cfg.embedding_dim, padding_idx=0
        )
        self.type_embedding = nn.Embedding(cfg.type_count, cfg.embedding_dim)
        input_dim = cfg.embedding_dim * 3 + cfg.numeric_feature_dim
        self.token_projection = nn.Sequential(
            nn.Linear(input_dim, cfg.model_dim),
            nn.GELU(),
            nn.LayerNorm(cfg.model_dim),
        )
        field_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.heads,
            dim_feedforward=cfg.model_dim * 3,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.field_encoder = nn.TransformerEncoder(
            field_layer, num_layers=cfg.field_layers, enable_nested_tensor=False
        )
        event_layer = nn.TransformerEncoderLayer(
            d_model=cfg.model_dim,
            nhead=cfg.heads,
            dim_feedforward=cfg.model_dim * 3,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.event_encoder = nn.TransformerEncoder(
            event_layer, num_layers=cfg.event_layers, enable_nested_tensor=False
        )
        self.field_weight_head = nn.Linear(cfg.model_dim, 1)
        self.field_evidence_head = nn.Linear(cfg.model_dim, 1)
        self.event_weight_head = nn.Linear(cfg.model_dim, 1)
        self.global_bias = nn.Parameter(torch.zeros(()))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        name_ids = batch["name_ids"]
        batch_size, event_count, field_count, _ = name_ids.shape
        name_vectors = masked_mean_embedding(
            self.name_embedding, name_ids, batch["name_piece_mask"]
        )
        value_vectors = masked_mean_embedding(
            self.value_embedding, batch["value_ids"], batch["value_piece_mask"]
        )
        type_vectors = self.type_embedding(batch["type_ids"])
        tokens = self.token_projection(
            torch.cat(
                [name_vectors, value_vectors, type_vectors, batch["numeric_features"]],
                dim=-1,
            )
        )

        flat_tokens = tokens.reshape(batch_size * event_count, field_count, -1)
        flat_mask = batch["field_mask"].reshape(batch_size * event_count, field_count)
        # Padded events have no fields. Give their first slot a temporary valid mask;
        # event_mask removes the resulting placeholder from the second hierarchy.
        safe_mask = flat_mask.clone()
        empty_events = ~safe_mask.any(dim=1)
        safe_mask[empty_events, 0] = True
        encoded_fields = self.field_encoder(
            flat_tokens, src_key_padding_mask=~safe_mask
        )
        field_weight_logits = self.field_weight_head(encoded_fields).squeeze(-1)
        field_weights = masked_softmax(field_weight_logits, safe_mask, dim=1)
        field_weights = field_weights * (~empty_events).unsqueeze(1).to(field_weights.dtype)
        raw_field_evidence = self.field_evidence_head(encoded_fields).squeeze(-1)
        if self.config.monotonic_baseline:
            portable_features = batch["numeric_features"][..., 8:]
            profile_known = portable_features[..., 0]
            field_known = portable_features[..., 1]
            categorical_rarity = portable_features[..., 3]
            numeric_deviation = portable_features[..., 4]
            type_mismatch = portable_features[..., 5]
            unseen_field = profile_known * (1.0 - field_known)
            anomaly = torch.stack(
                [categorical_rarity, numeric_deviation, type_mismatch, unseen_field],
                dim=-1,
            ).amax(dim=-1)
            flat_anomaly = anomaly.reshape(batch_size * event_count, field_count)
            bounded_importance = 0.5 + torch.sigmoid(raw_field_evidence)
            field_evidence = bounded_importance * 4.0 * flat_anomaly
        else:
            field_evidence = raw_field_evidence
        event_vectors = torch.sum(field_weights.unsqueeze(-1) * encoded_fields, dim=1)
        event_field_logit = torch.sum(field_weights * field_evidence, dim=1)

        event_vectors = event_vectors.reshape(batch_size, event_count, -1)
        event_field_logit = event_field_logit.reshape(batch_size, event_count)
        encoded_events = self.event_encoder(
            event_vectors, src_key_padding_mask=~batch["event_mask"]
        )
        event_weight_logits = self.event_weight_head(encoded_events).squeeze(-1)
        event_weights = masked_softmax(event_weight_logits, batch["event_mask"], dim=1)
        risk_logit = self.global_bias + torch.sum(event_weights * event_field_logit, dim=1)
        event_anomaly_score = self.global_bias + event_field_logit
        risk_probability = torch.sigmoid(risk_logit)
        trust_score = 100.0 * (1.0 - risk_probability)

        flat_field_weights = field_weights.reshape(batch_size, event_count, field_count)
        flat_field_weight_logits = field_weight_logits.reshape(
            batch_size, event_count, field_count
        )
        flat_field_evidence = field_evidence.reshape(batch_size, event_count, field_count)
        field_contributions = (
            event_weights.unsqueeze(-1) * flat_field_weights * flat_field_evidence
        )
        valid_field_count = batch["field_mask"].sum(dim=(1, 2)).clamp_min(1)
        joint_weights = event_weights.unsqueeze(-1) * flat_field_weights
        entropy = -torch.sum(
            torch.where(
                joint_weights > 0,
                joint_weights * torch.log(joint_weights.clamp_min(1e-12)),
                torch.zeros_like(joint_weights),
            ),
            dim=(1, 2),
        )
        max_entropy = torch.log(valid_field_count.to(entropy.dtype)).clamp_min(1.0)
        concentration = 1.0 - (entropy / max_entropy).clamp(0.0, 1.0)
        margin = torch.abs(2.0 * risk_probability - 1.0)
        confidence = (0.5 * concentration + 0.5 * margin).clamp(0.0, 1.0)

        return {
            "risk_logit": risk_logit,
            "event_anomaly_score": event_anomaly_score,
            "risk_probability": risk_probability,
            "trust_score": trust_score,
            "confidence": confidence,
            "attention_concentration": concentration,
            "field_weights": flat_field_weights,
            "field_weight_logits": flat_field_weight_logits,
            "event_weights": event_weights,
            "field_evidence": flat_field_evidence,
            "field_contributions": field_contributions,
        }
