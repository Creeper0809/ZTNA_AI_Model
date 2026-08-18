from __future__ import annotations

import torch
from torch import nn

from ztna_ueba.fine_tune import (
    configure_head_only_fine_tuning,
    fine_tune_batch,
    is_fine_tunable_parameter,
)
from ztna_ueba.model import HierarchicalFieldAttention, ModelConfig
from ztna_ueba.tokenizer import FieldTokenizer, TokenizerConfig, collate_requests


def _small_model() -> HierarchicalFieldAttention:
    torch.manual_seed(23)
    return HierarchicalFieldAttention(
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


def _labeled_batch() -> dict[str, object]:
    tokenizer = FieldTokenizer(
        TokenizerConfig(name_buckets=1024, value_buckets=2048)
    )
    requests = [
        tokenizer.tokenize_request(
            [
                {"device_state": "healthy", "latency_ms": 100 + index},
                {"traffic": "normal", "bytes": 1000 + index},
            ]
        )
        for index in range(4)
    ]
    requests += [
        tokenizer.tokenize_request(
            [
                {"device_state": "compromised", "latency_ms": 900 + index},
                {"traffic": "abnormal", "bytes": 900000 + index},
            ]
        )
        for index in range(4)
    ]
    batch = collate_requests(requests)
    batch["labels"] = torch.tensor([0.0] * 4 + [1.0] * 4)
    return batch


def test_head_only_configuration_freezes_general_feature_extractor():
    model = _small_model()
    trainable = configure_head_only_fine_tuning(model)

    assert trainable
    assert "global_bias" in trainable
    assert all(is_fine_tunable_parameter(name) for name in trainable)
    assert all(
        parameter.requires_grad == is_fine_tunable_parameter(name)
        for name, parameter in model.named_parameters()
    )
    assert model.name_embedding.weight.requires_grad is False
    assert model.token_projection[0].weight.requires_grad is False


def test_fine_tune_batch_changes_only_weight_and_evidence_heads():
    model = _small_model().eval()
    configure_head_only_fine_tuning(model)
    before = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    loss_function = nn.BCEWithLogitsLoss()

    fine_tune_batch(
        model,
        _labeled_batch(),
        optimizer,
        loss_function,
        device=torch.device("cpu"),
        field_dropout=0.0,
        attention_entropy_weight=0.0,
    )

    changed = {
        name
        for name, value in model.state_dict().items()
        if not torch.equal(before[name], value)
    }
    assert changed
    assert changed <= {
        name for name in before if is_fine_tunable_parameter(name)
    }
    assert "global_bias" in changed
    assert any(name.startswith("field_weight_head.") for name in changed)
    assert any(name.startswith("field_evidence_head.") for name in changed)
