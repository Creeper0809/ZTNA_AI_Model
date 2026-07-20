from __future__ import annotations

import json

import pytest
import torch

from ztna_ueba.baseline import BaselineRegistry
from ztna_ueba.calibration import NormalScoreCalibrator
from ztna_ueba.model import HierarchicalFieldAttention, ModelConfig
from ztna_ueba.tokenizer import (
    DEFAULT_EXCLUDED_FIELDS,
    FieldTokenizer,
    TokenizerConfig,
    collate_requests,
)


def _normal_records() -> list[dict[str, object]]:
    return [
        {
            "dataset": "company-a",
            "source_type": "auth",
            "event_type": "login",
            "action": "allow",
            "duration_ms": 100 + (index % 5),
        }
        for index in range(40)
    ]


def _portable_tokenizer() -> tuple[BaselineRegistry, FieldTokenizer]:
    registry = BaselineRegistry.fit(
        _normal_records(), excluded_fields=DEFAULT_EXCLUDED_FIELDS
    )
    tokenizer = FieldTokenizer(
        TokenizerConfig(
            name_buckets=1024,
            value_buckets=2048,
            portable_mode=True,
        ),
        baseline_registry=registry,
    )
    return registry, tokenizer


def test_registry_round_trip_and_serialization_do_not_expose_raw_values():
    registry, _ = _portable_tokenizer()
    serialized = registry.to_dict()
    encoded = json.dumps(serialized, sort_keys=True)

    assert "allow" not in encoded
    restored = BaselineRegistry.from_dict(serialized)
    normal = _normal_records()[0]
    assert restored.coverage(normal, DEFAULT_EXCLUDED_FIELDS)["coverage"] == 1.0
    assert restored.features(restored.profile_key(normal), "action", "allow") == registry.features(
        registry.profile_key(normal), "action", "allow"
    )


def test_portable_tokens_hide_values_and_express_only_baseline_relative_anomaly():
    _, tokenizer = _portable_tokenizer()
    known = tokenizer.tokenize_event(
        {
            "dataset": "company-a",
            "source_type": "auth",
            "event_type": "login",
            "action": "allow",
            "duration_ms": 102,
        }
    )
    anomalous = tokenizer.tokenize_event(
        {
            "dataset": "company-a",
            "source_type": "auth",
            "event_type": "login",
            "action": "never-seen-secret-value",
            "duration_ms": 10000,
        }
    )

    assert known.field_names == ["action", "duration_ms"]
    assert known.value_ids == anomalous.value_ids
    assert all(features[:8] == [0.0] * 8 for features in known.numeric_features)
    action_index = known.field_names.index("action")
    duration_index = known.field_names.index("duration_ms")
    assert anomalous.numeric_features[action_index][11] > known.numeric_features[action_index][11]
    assert anomalous.numeric_features[duration_index][12] > known.numeric_features[duration_index][12]


def test_unknown_profile_has_zero_coverage_and_is_not_ready():
    registry, _ = _portable_tokenizer()
    unknown = {
        "dataset": "company-b",
        "source_type": "proxy",
        "event_type": "request",
        "url_category": "new",
    }
    coverage = registry.coverage(unknown, DEFAULT_EXCLUDED_FIELDS)
    assert coverage["profile_known"] is False
    assert coverage["coverage"] == 0.0


def test_unique_identifiers_are_not_treated_like_repeatable_categories():
    records = [
        {
            "dataset": "company-a",
            "source_type": "auth",
            "event_type": "login",
            "session_id": f"unique-{index}",
            "device_state": "healthy",
        }
        for index in range(40)
    ]
    registry = BaselineRegistry.fit(records, excluded_fields=DEFAULT_EXCLUDED_FIELDS)
    profile = registry.profile_key(records[0])

    session_rarity = registry.features(profile, "session_id", "brand-new")[3]
    state_rarity = registry.features(profile, "device_state", "compromised")[3]
    assert session_rarity == 0.0
    assert state_rarity > 0.9


def test_timestamps_use_hour_of_week_instead_of_exact_value_rarity():
    records = [
        {
            "dataset": "company-a",
            "source_type": "auth",
            "event_type": "login",
            "event_time": f"2026-06-{1 + index:02d}T09:00:00Z",
        }
        for index in range(20)
    ]
    registry = BaselineRegistry.fit(records, excluded_fields=DEFAULT_EXCLUDED_FIELDS)
    profile = registry.profile_key(records[0])
    field = registry.profiles[profile].fields["event_time"]

    assert field.kind == "datetime"
    assert all(key.startswith("hour-of-week:") for key in field.category_counts or {})
    assert "2026-06" not in json.dumps(registry.to_dict())


def test_monotonic_portable_head_uses_nonnegative_anomaly_evidence():
    _, tokenizer = _portable_tokenizer()
    normal = tokenizer.tokenize_request(
        {
            "dataset": "company-a",
            "source_type": "auth",
            "event_type": "login",
            "action": "allow",
            "duration_ms": 102,
        }
    )
    anomalous = tokenizer.tokenize_request(
        {
            "dataset": "company-a",
            "source_type": "auth",
            "event_type": "login",
            "action": "never-seen",
            "duration_ms": 10000,
        }
    )
    batch = collate_requests([normal, anomalous])
    torch.manual_seed(11)
    model = HierarchicalFieldAttention(
        ModelConfig(
            name_buckets=1024,
            value_buckets=2048,
            embedding_dim=16,
            model_dim=32,
            heads=4,
            field_layers=1,
            event_layers=1,
            dropout=0.0,
            numeric_feature_dim=16,
            monotonic_baseline=True,
        )
    ).eval()
    with torch.no_grad():
        output = model(batch)

    assert torch.all(output["field_evidence"] >= 0)
    assert torch.all(output["field_evidence"] <= 6.0)
    assert output["risk_logit"][1] > output["risk_logit"][0]
    reconstructed = model.global_bias + output["field_contributions"].sum(dim=(1, 2))
    torch.testing.assert_close(reconstructed, output["risk_logit"], atol=1e-6, rtol=1e-6)


def test_normal_only_score_calibration_round_trip_and_unknown_profile():
    profiles = ["company-a|auth|login"] * 100
    scores = [index / 100.0 for index in range(100)]
    calibrator = NormalScoreCalibrator.fit(profiles, scores)
    restored = NormalScoreCalibrator.from_dict(calibrator.to_dict())
    reference = restored.profiles[profiles[0]]

    assert restored.normalize("company-a|auth|login", 10.0) > 0.99
    assert restored.normalize("company-a|auth|login", 0.0) < 0.5
    assert restored.normalize(profiles[0], reference.q95) == 0.5
    assert restored.normalize(profiles[0], reference.q99) == pytest.approx(0.9)
    assert restored.normalize("unknown|auth|login", 10.0) is None

    event_scores = torch.tensor([[10.0], [0.0]])
    event_weights = torch.ones(2, 1)
    event_mask = torch.ones(2, 1, dtype=torch.bool)
    risk, ready = restored.normalize_batch(
        [[profiles[0]], [profiles[0]]], event_scores, event_weights, event_mask
    )
    assert ready.tolist() == [True, True]
    assert risk[0] > risk[1]
