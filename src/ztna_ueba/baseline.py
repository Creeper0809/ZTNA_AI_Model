"""Privacy-preserving normal baselines for portable field anomaly features."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import math
from statistics import median
from typing import Iterable, Mapping


BASELINE_FEATURE_DIM = 8
DEFAULT_PROFILE_FIELDS = ("dataset", "source_type", "event_type")


def is_missing(value: object) -> bool:
    if value is None or type(value).__name__ in {"NAType", "NaTType"}:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "<na>", "nan", "nat"}


def parse_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_time_bucket(value: object) -> int | None:
    text = str(value).strip()
    if len(text) < 10 or text[4:5] != "-" or text[7:8] != "-":
        return None
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return timestamp.weekday() * 24 + timestamp.hour


def value_fingerprint(value: object) -> str:
    normalized = str(value).strip().lower().encode("utf-8", errors="replace")
    return hashlib.blake2b(
        normalized, digest_size=8, person=b"ztna-baseline"
    ).hexdigest()


def make_profile_key(
    record: Mapping[str, object], profile_fields: tuple[str, ...] = DEFAULT_PROFILE_FIELDS
) -> str:
    values = []
    for field in profile_fields:
        value = record.get(field)
        values.append(str(value).strip().lower() if not is_missing(value) else "<unknown>")
    return "|".join(values)


@dataclass
class FieldBaseline:
    observed: int
    kind: str
    median: float = 0.0
    scale: float = 1.0
    category_counts: dict[str, int] | None = None
    unique_count: int = 0


@dataclass
class ProfileBaseline:
    records: int
    fields: dict[str, FieldBaseline]


class BaselineRegistry:
    def __init__(
        self,
        profiles: dict[str, ProfileBaseline] | None = None,
        profile_fields: tuple[str, ...] = DEFAULT_PROFILE_FIELDS,
    ):
        self.profiles = profiles or {}
        self.profile_fields = profile_fields

    @classmethod
    def fit(
        cls,
        records: Iterable[Mapping[str, object]],
        excluded_fields: frozenset[str],
        profile_fields: tuple[str, ...] = DEFAULT_PROFILE_FIELDS,
        max_categories: int = 4096,
    ) -> "BaselineRegistry":
        profile_counts: Counter[str] = Counter()
        observed: Counter[tuple[str, str]] = Counter()
        numeric_values: dict[tuple[str, str], list[float]] = defaultdict(list)
        temporal_values: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        categorical_values: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for record in records:
            profile_key = make_profile_key(record, profile_fields)
            profile_counts[profile_key] += 1
            for raw_name, value in record.items():
                name = str(raw_name).strip().lower()
                if name in excluded_fields or name in profile_fields or is_missing(value):
                    continue
                key = (profile_key, name)
                observed[key] += 1
                number = parse_number(value)
                if number is not None:
                    numeric_values[key].append(number)
                else:
                    time_bucket = parse_time_bucket(value)
                    if time_bucket is not None:
                        temporal_values[key][f"hour-of-week:{time_bucket}"] += 1
                    categorical_values[key][value_fingerprint(value)] += 1

        profiles: dict[str, ProfileBaseline] = {}
        for profile_key, record_count in profile_counts.items():
            field_names = {
                field_name
                for current_profile, field_name in observed
                if current_profile == profile_key
            }
            fields: dict[str, FieldBaseline] = {}
            for field_name in field_names:
                key = (profile_key, field_name)
                field_observed = observed[key]
                numbers = numeric_values.get(key, [])
                numeric_ratio = len(numbers) / max(1, field_observed)
                time_counts = temporal_values.get(key, Counter())
                time_ratio = sum(time_counts.values()) / max(1, field_observed)
                if len(numbers) >= 5 and numeric_ratio >= 0.80:
                    center = float(median(numbers))
                    deviations = [abs(value - center) for value in numbers]
                    mad = float(median(deviations))
                    sorted_numbers = sorted(numbers)
                    q25 = sorted_numbers[int(0.25 * (len(sorted_numbers) - 1))]
                    q75 = sorted_numbers[int(0.75 * (len(sorted_numbers) - 1))]
                    robust_scale = max(1e-6, 1.4826 * mad, (q75 - q25) / 1.349)
                    fields[field_name] = FieldBaseline(
                        observed=field_observed,
                        kind="numeric",
                        median=center,
                        scale=robust_scale,
                    )
                elif sum(time_counts.values()) >= 5 and time_ratio >= 0.80:
                    fields[field_name] = FieldBaseline(
                        observed=field_observed,
                        kind="datetime",
                        category_counts=dict(time_counts),
                        unique_count=len(time_counts),
                    )
                else:
                    counts = categorical_values.get(key, Counter())
                    fields[field_name] = FieldBaseline(
                        observed=field_observed,
                        kind="categorical",
                        category_counts=dict(counts.most_common(max_categories)),
                        unique_count=len(counts),
                    )
            profiles[profile_key] = ProfileBaseline(records=record_count, fields=fields)
        return cls(profiles=profiles, profile_fields=profile_fields)

    def profile_key(self, record: Mapping[str, object]) -> str:
        return make_profile_key(record, self.profile_fields)

    def features(self, profile_key: str, field_name: str, value: object) -> list[float]:
        profile = self.profiles.get(profile_key)
        if profile is None:
            return [0.0] * BASELINE_FEATURE_DIM
        field = profile.fields.get(field_name.strip().lower())
        if field is None:
            return [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

        presence_surprise = 1.0 - min(1.0, field.observed / max(1, profile.records))
        support = math.tanh(math.log1p(field.observed) / 8.0)
        categorical_rarity = 0.0
        numeric_deviation = 0.0
        type_mismatch = 0.0
        number = parse_number(value)
        if field.kind == "numeric":
            if number is None:
                type_mismatch = 1.0
                numeric_deviation = 1.0
            else:
                robust_z = abs(number - field.median) / max(field.scale, 1e-6)
                numeric_deviation = math.tanh(robust_z / 8.0)
        else:
            if number is not None:
                type_mismatch = 1.0
            counts = field.category_counts or {}
            if field.kind == "datetime":
                bucket = parse_time_bucket(value)
                if bucket is None:
                    type_mismatch = 1.0
                    count = 0
                else:
                    count = counts.get(f"hour-of-week:{bucket}", 0)
            else:
                count = counts.get(value_fingerprint(value), 0)
            vocabulary = max(1, len(counts))
            probability = (count + 0.5) / (field.observed + 0.5 * (vocabulary + 1))
            denominator = max(1.0, math.log(field.observed + vocabulary + 2.0))
            categorical_rarity = min(1.0, -math.log(max(probability, 1e-12)) / denominator)
            if field.kind == "categorical":
                repeatability = 1.0 - min(
                    1.0, field.unique_count / max(1, field.observed)
                )
                categorical_rarity *= math.sqrt(repeatability)
        ready = 1.0 if field.observed >= 20 else field.observed / 20.0
        return [
            1.0,
            1.0,
            presence_surprise,
            categorical_rarity,
            numeric_deviation,
            type_mismatch,
            support,
            ready,
        ]

    def coverage(self, record: Mapping[str, object], excluded_fields: frozenset[str]) -> dict:
        key = self.profile_key(record)
        profile = self.profiles.get(key)
        usable = [
            str(name).strip().lower()
            for name, value in record.items()
            if str(name).strip().lower() not in excluded_fields
            and str(name).strip().lower() not in self.profile_fields
            and not is_missing(value)
        ]
        known = 0 if profile is None else sum(name in profile.fields for name in usable)
        return {
            "profile_key": key,
            "profile_known": profile is not None,
            "usable_fields": len(usable),
            "known_fields": known,
            "coverage": known / max(1, len(usable)),
        }

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "profile_fields": list(self.profile_fields),
            "profiles": {
                key: {
                    "records": profile.records,
                    "fields": {
                        name: asdict(field) for name, field in profile.fields.items()
                    },
                }
                for key, profile in self.profiles.items()
            },
        }

    @classmethod
    def from_dict(cls, value: dict) -> "BaselineRegistry":
        profiles = {
            key: ProfileBaseline(
                records=int(profile["records"]),
                fields={
                    name: FieldBaseline(**field)
                    for name, field in profile["fields"].items()
                },
            )
            for key, profile in value.get("profiles", {}).items()
        }
        return cls(
            profiles=profiles,
            profile_fields=tuple(value.get("profile_fields", DEFAULT_PROFILE_FIELDS)),
        )
