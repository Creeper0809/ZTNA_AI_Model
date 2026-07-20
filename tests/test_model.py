from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from ztna_ueba.model import HierarchicalFieldAttention, ModelConfig
from ztna_ueba.tokenizer import FieldTokenizer, TokenizerConfig, collate_requests


@pytest.fixture
def tokenizer():
    return FieldTokenizer(
        TokenizerConfig(name_buckets=1024, value_buckets=2048)
    )


@pytest.fixture
def model():
    torch.manual_seed(7)
    instance = HierarchicalFieldAttention(
        ModelConfig(
            name_buckets=1024,
            value_buckets=2048,
            embedding_dim=16,
            model_dim=32,
            heads=4,
            field_layers=1,
            event_layers=1,
            dropout=0.0,
        )
    )
    return instance.eval()


def test_leak_fields_and_missing_values_are_removed(tokenizer):
    event = tokenizer.tokenize_event(
        {
            "label": "abnormal",
            "is_attack": 1,
            "result": "abnormal",
            "dataset": "cicids2019",
            "src_port": 443,
            "optional": None,
        }
    )
    assert event.field_names == ["src_port"]


def test_identifier_values_are_redacted_but_field_names_remain(tokenizer):
    first = tokenizer.tokenize_event({"session_id": "secret-a", "src_ip": "10.1.2.3"})
    second = tokenizer.tokenize_event({"session_id": "secret-b", "src_ip": "192.168.5.6"})
    assert first.field_names == second.field_names
    assert first.value_ids == second.value_ids
    assert first.numeric_features != second.numeric_features


def test_field_order_is_permutation_invariant(tokenizer, model):
    first = OrderedDict(
        [("src_port", 443), ("protocol", "tcp"), ("device_posture", "healthy")]
    )
    second = OrderedDict(reversed(list(first.items())))
    batch_a = collate_requests([tokenizer.tokenize_request(first)])
    batch_b = collate_requests([tokenizer.tokenize_request(second)])
    with torch.no_grad():
        output_a = model(batch_a)
        output_b = model(batch_b)
    torch.testing.assert_close(output_a["risk_logit"], output_b["risk_logit"], atol=1e-6, rtol=1e-6)


def test_arbitrary_fields_and_multiple_sources_have_normalized_weights(tokenizer, model):
    first = {f"vendor_a_field_{index}": index for index in range(37)}
    second = {
        "completely_new_signal": "rare-value",
        "event_time": "2026-07-20T03:15:00Z",
        "src_ip": "10.1.2.3",
    }
    tokenized = tokenizer.tokenize_request([first, second])
    batch = collate_requests([tokenized])
    with torch.no_grad():
        output = model(batch)
    assert output["field_weights"].shape == (1, 2, 37)
    torch.testing.assert_close(output["event_weights"].sum(dim=1), torch.ones(1))
    for event_index, event in enumerate(tokenized.events):
        actual = output["field_weights"][0, event_index, : len(event.field_names)].sum()
        torch.testing.assert_close(actual, torch.tensor(1.0))
    assert 0.0 <= float(output["trust_score"][0]) <= 100.0


def test_reported_field_contributions_reconstruct_risk_logit(tokenizer, model):
    request = tokenizer.tokenize_request(
        [
            {"user_action": "download", "hour": 3},
            {"dst_port": 445, "bytes_total": 900000},
        ]
    )
    batch = collate_requests([request])
    with torch.no_grad():
        output = model(batch)
    reconstructed = model.global_bias + output["field_contributions"].sum(dim=(1, 2))
    torch.testing.assert_close(reconstructed, output["risk_logit"], atol=1e-6, rtol=1e-6)
