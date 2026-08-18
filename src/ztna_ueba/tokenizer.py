"""Turn arbitrary key-value logs into padded sets of field tokens."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
import re
from typing import Iterable, Mapping, Sequence

import torch

from .baseline import BASELINE_FEATURE_DIM, BaselineRegistry, DEFAULT_PROFILE_FIELDS
from .hashing import field_name_pieces, stable_bucket, value_pieces


NORMAL_REFERENCE_FIELDS_KEY = "__ueba_normal_reference_fields__"

DEFAULT_EXCLUDED_FIELDS = frozenset(
    {
        "label",
        "is_attack",
        "attack_category",
        "raw_label",
        "result",
        "source_file",
        "dataset",
        "split",
        "_sample_hash",
        "_group_id",
        "target",
        "ground_truth",
        "y",
        NORMAL_REFERENCE_FIELDS_KEY,
    }
)

TYPE_MISSING = 0
TYPE_BOOL = 1
TYPE_NUMERIC = 2
TYPE_DATETIME = 3
TYPE_IP = 4
TYPE_TEXT = 5

DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]")
IP_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")
IDENTIFIER_FIELD_RE = re.compile(
    r"(?:^|_)(?:id|hash|uuid|guid|ip|mac|hostname)(?:$|_)", re.IGNORECASE
)


@dataclass(frozen=True)
class TokenizerConfig:
    name_buckets: int = 32768
    value_buckets: int = 131072
    max_name_pieces: int = 12
    max_value_pieces: int = 8
    excluded_fields: frozenset[str] = field(default_factory=lambda: DEFAULT_EXCLUDED_FIELDS)
    max_fields: int | None = None
    max_events: int | None = None
    redact_identifier_values: bool = True
    portable_mode: bool = False
    profile_fields: tuple[str, ...] = DEFAULT_PROFILE_FIELDS


@dataclass
class TokenizedEvent:
    profile_key: str
    field_names: list[str]
    name_ids: list[list[int]]
    value_ids: list[list[int]]
    type_ids: list[int]
    numeric_features: list[list[float]]


@dataclass
class TokenizedRequest:
    events: list[TokenizedEvent]


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "<na>", "nan", "nat"}


def _parse_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "on"}:
        return True
    if normalized in {"false", "no", "off"}:
        return False
    return None


def _parse_numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_datetime(value: object) -> datetime | None:
    text = str(value).strip()
    if not DATETIME_RE.match(text):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _value_encoding(value: object) -> tuple[int, list[float]]:
    features = [0.0] * 8
    boolean = _parse_bool(value)
    if boolean is not None:
        features[0] = 1.0 if boolean else -1.0
        return TYPE_BOOL, features

    number = _parse_numeric(value)
    if number is not None:
        magnitude = math.log1p(abs(number))
        features[0] = 1.0
        features[1] = 0.0 if number == 0 else math.copysign(1.0, number)
        features[2] = math.tanh(magnitude / 8.0)
        features[3] = math.tanh(number / 1000.0)
        features[4] = 1.0 if number == 0 else 0.0
        features[5] = abs(number) % 1.0
        return TYPE_NUMERIC, features

    timestamp = _parse_datetime(value)
    if timestamp is not None:
        hour_angle = 2.0 * math.pi * timestamp.hour / 24.0
        day_angle = 2.0 * math.pi * timestamp.weekday() / 7.0
        features[0] = math.sin(hour_angle)
        features[1] = math.cos(hour_angle)
        features[2] = math.sin(day_angle)
        features[3] = math.cos(day_angle)
        features[4] = 1.0 if timestamp.weekday() >= 5 else 0.0
        return TYPE_DATETIME, features

    text = str(value).strip()
    if IP_RE.match(text):
        octets = [min(255, int(piece)) for piece in text.split(".")]
        for index, octet in enumerate(octets[:4]):
            features[index] = octet / 255.0
        features[4] = 1.0 if octets[0] in {10, 127} else 0.0
        features[5] = 1.0 if octets[:2] == [192, 168] else 0.0
        features[6] = 1.0 if octets[0] == 172 and 16 <= octets[1] <= 31 else 0.0
        return TYPE_IP, features

    features[0] = min(len(text), 512) / 512.0
    features[1] = min(len(text.split()), 32) / 32.0
    return TYPE_TEXT, features


class FieldTokenizer:
    def __init__(
        self,
        config: TokenizerConfig | None = None,
        baseline_registry: BaselineRegistry | None = None,
    ):
        self.config = config or TokenizerConfig()
        self.baseline_registry = baseline_registry

    def tokenize_event(self, record: Mapping[str, object]) -> TokenizedEvent:
        names: list[str] = []
        name_ids: list[list[int]] = []
        value_ids: list[list[int]] = []
        type_ids: list[int] = []
        numeric_features: list[list[float]] = []
        profile_key = (
            self.baseline_registry.profile_key(record)
            if self.baseline_registry is not None
            else "|".join(
                str(record.get(field, "<unknown>")).strip().lower()
                for field in self.config.profile_fields
            )
        )
        metadata_fields = set(self.config.profile_fields) if self.config.portable_mode else set()
        raw_reference_fields = record.get(NORMAL_REFERENCE_FIELDS_KEY, ())
        reference_fields = (
            {
                str(name).strip().lower()
                for name in raw_reference_fields
            }
            if isinstance(raw_reference_fields, (list, tuple, set, frozenset))
            else set()
        )
        fields = [
            (str(name), value)
            for name, value in record.items()
            if str(name).strip().lower() not in self.config.excluded_fields
            and str(name).strip().lower() != NORMAL_REFERENCE_FIELDS_KEY
            and str(name).strip().lower() not in metadata_fields
            and not _is_missing(value)
        ]
        if self.config.max_fields is not None:
            fields = sorted(fields, key=lambda pair: pair[0])[: self.config.max_fields]
        for name, value in fields:
            type_id, numeric = _value_encoding(value)
            name_parts = field_name_pieces(name, self.config.max_name_pieces)
            if self.config.portable_mode:
                value_parts = [f"<type:{type_id}>"]
                numeric = [0.0] * 8
                baseline = (
                    self.baseline_registry.normal_reference_features_for_record(
                        record, name
                    )[0]
                    if self.baseline_registry is not None
                    and name.strip().lower() in reference_fields
                    else self.baseline_registry.features_for_record(record, name, value)
                    if self.baseline_registry is not None
                    else [0.0] * BASELINE_FEATURE_DIM
                )
                numeric.extend(baseline)
            elif self.config.redact_identifier_values and IDENTIFIER_FIELD_RE.search(name):
                value_parts = ["<identifier>"]
            elif type_id == TYPE_NUMERIC:
                value_parts = ["<numeric>"]
            elif type_id == TYPE_DATETIME:
                value_parts = ["<datetime>"]
            elif type_id == TYPE_IP:
                value_parts = ["<ip>"]
            else:
                value_parts = value_pieces(value, self.config.max_value_pieces)
            names.append(name)
            name_ids.append(
                [
                    stable_bucket(part, self.config.name_buckets, b"ztna-field-name")
                    for part in name_parts
                ]
            )
            value_ids.append(
                [
                    stable_bucket(part, self.config.value_buckets, b"ztna-field-value")
                    for part in value_parts
                ]
                or [stable_bucket("<empty>", self.config.value_buckets, b"ztna-field-value")]
            )
            type_ids.append(type_id)
            numeric_features.append(numeric)
        if not names:
            raise ValueError("record contains no usable fields after exclusions")
        return TokenizedEvent(
            profile_key, names, name_ids, value_ids, type_ids, numeric_features
        )

    def tokenize_request(
        self, records: Mapping[str, object] | Sequence[Mapping[str, object]]
    ) -> TokenizedRequest:
        events = [records] if isinstance(records, Mapping) else list(records)
        if self.config.max_events is not None:
            events = events[: self.config.max_events]
        if not events:
            raise ValueError("request must contain at least one event")
        return TokenizedRequest([self.tokenize_event(event) for event in events])


def collate_requests(requests: Iterable[TokenizedRequest]) -> dict[str, object]:
    batch = list(requests)
    if not batch:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(batch)
    max_events = max(len(request.events) for request in batch)
    max_fields = max(len(event.field_names) for request in batch for event in request.events)
    max_name = max(
        len(ids)
        for request in batch
        for event in request.events
        for ids in event.name_ids
    )
    max_value = max(
        len(ids)
        for request in batch
        for event in request.events
        for ids in event.value_ids
    )
    name_ids = torch.zeros(batch_size, max_events, max_fields, max_name, dtype=torch.long)
    name_piece_mask = torch.zeros_like(name_ids, dtype=torch.bool)
    value_ids = torch.zeros(batch_size, max_events, max_fields, max_value, dtype=torch.long)
    value_piece_mask = torch.zeros_like(value_ids, dtype=torch.bool)
    type_ids = torch.zeros(batch_size, max_events, max_fields, dtype=torch.long)
    numeric_dim = max(
        len(features)
        for request in batch
        for event in request.events
        for features in event.numeric_features
    )
    numeric = torch.zeros(
        batch_size, max_events, max_fields, numeric_dim, dtype=torch.float32
    )
    field_mask = torch.zeros(batch_size, max_events, max_fields, dtype=torch.bool)
    event_mask = torch.zeros(batch_size, max_events, dtype=torch.bool)
    metadata: list[list[list[str]]] = []
    profile_keys: list[list[str]] = []

    for batch_index, request in enumerate(batch):
        request_names: list[list[str]] = []
        request_profiles: list[str] = []
        for event_index, event in enumerate(request.events):
            event_mask[batch_index, event_index] = True
            request_names.append(list(event.field_names))
            request_profiles.append(event.profile_key)
            for field_index, field_name in enumerate(event.field_names):
                field_mask[batch_index, event_index, field_index] = True
                names = event.name_ids[field_index]
                values = event.value_ids[field_index]
                name_ids[batch_index, event_index, field_index, : len(names)] = torch.tensor(names)
                name_piece_mask[batch_index, event_index, field_index, : len(names)] = True
                value_ids[batch_index, event_index, field_index, : len(values)] = torch.tensor(values)
                value_piece_mask[batch_index, event_index, field_index, : len(values)] = True
                type_ids[batch_index, event_index, field_index] = event.type_ids[field_index]
                numeric[batch_index, event_index, field_index] = torch.tensor(
                    event.numeric_features[field_index]
                )
        metadata.append(request_names)
        profile_keys.append(request_profiles)

    return {
        "name_ids": name_ids,
        "name_piece_mask": name_piece_mask,
        "value_ids": value_ids,
        "value_piece_mask": value_piece_mask,
        "type_ids": type_ids,
        "numeric_features": numeric,
        "field_mask": field_mask,
        "event_mask": event_mask,
        "field_names": metadata,
        "profile_keys": profile_keys,
    }
