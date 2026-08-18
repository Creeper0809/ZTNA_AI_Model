"""Privacy-preserving hierarchical UEBA baselines for portable field features."""

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
DEFAULT_ACTOR_FIELDS = (
    "actor_id",
    "actor_alias",
    "user_id_hash",
    "user_id",
    "user",
    "username",
    "subject",
    "account",
    "principal",
    "tenant_actor",
)
DEFAULT_DEVICE_FIELDS = (
    "device_id_hash",
    "device_id",
    "host_id_hash",
    "host_id",
    "hostname",
    "host",
    "endpoint_id",
)
DEFAULT_PEER_FIELDS = (
    "peer_group",
    "department",
    "team",
    "user_role",
    "job_role",
    "workload_role",
    "entity_type",
)
DEFAULT_SCOPE_MINIMUM_RECORDS = {
    "actor": 20,
    "device": 20,
    "peer_group": 20,
    "log_profile": 1,
}
SCOPE_ORDER = ("actor", "device", "peer_group", "log_profile")
SCOPE_LABELS_KO = {
    "actor": "사용자·서비스 계정 기준선",
    "device": "단말 기준선",
    "peer_group": "동료 집단 기준선",
    "log_profile": "같은 로그 유형 기준선",
    "unavailable": "사용 가능한 기준선 없음",
}


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
    normalized = {str(name).strip().lower(): value for name, value in record.items()}
    values = []
    for field in profile_fields:
        value = normalized.get(field.lower())
        values.append(str(value).strip().lower() if not is_missing(value) else "<unknown>")
    return "|".join(values)


def _first_present(
    record: Mapping[str, object], fields: tuple[str, ...]
) -> tuple[str, object] | None:
    normalized = {str(name).strip().lower(): value for name, value in record.items()}
    for field in fields:
        value = normalized.get(field)
        if not is_missing(value):
            return field, value
    return None


def _hierarchy_key(scope: str, profile_key: str, value: object) -> str:
    digest = value_fingerprint(f"{scope}:{value}")
    return f"{profile_key}|{scope}:{digest}"


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


@dataclass(frozen=True)
class BaselineSelection:
    profile_key: str
    baseline_key: str | None
    scope: str
    scope_label: str
    selector_field: str | None
    records: int
    field_observations: int
    fallback_used: bool
    selection_reason: str
    candidates: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_key": self.profile_key,
            "baseline_key": self.baseline_key,
            "scope": self.scope,
            "scope_label": self.scope_label,
            "selector_field": self.selector_field,
            "records": self.records,
            "field_observations": self.field_observations,
            "fallback_used": self.fallback_used,
            "selection_reason": self.selection_reason,
            "candidates": [dict(candidate) for candidate in self.candidates],
        }


class BaselineRegistry:
    def __init__(
        self,
        profiles: dict[str, ProfileBaseline] | None = None,
        profile_fields: tuple[str, ...] = DEFAULT_PROFILE_FIELDS,
        *,
        hierarchical_profiles: dict[str, dict[str, ProfileBaseline]] | None = None,
        actor_fields: tuple[str, ...] = DEFAULT_ACTOR_FIELDS,
        device_fields: tuple[str, ...] = DEFAULT_DEVICE_FIELDS,
        peer_fields: tuple[str, ...] = DEFAULT_PEER_FIELDS,
        minimum_records: Mapping[str, int] | None = None,
    ):
        self.profiles = profiles or {}
        self.profile_fields = tuple(profile_fields)
        self.hierarchical_profiles = {
            scope: dict((hierarchical_profiles or {}).get(scope, {}))
            for scope in SCOPE_ORDER[:-1]
        }
        self.actor_fields = tuple(actor_fields)
        self.device_fields = tuple(device_fields)
        self.peer_fields = tuple(peer_fields)
        configured = dict(DEFAULT_SCOPE_MINIMUM_RECORDS)
        configured.update({key: int(value) for key, value in (minimum_records or {}).items()})
        self.minimum_records = configured

    @staticmethod
    def _candidate_descriptors(
        record: Mapping[str, object],
        profile_fields: tuple[str, ...],
        actor_fields: tuple[str, ...],
        device_fields: tuple[str, ...],
        peer_fields: tuple[str, ...],
    ) -> list[tuple[str, str, str | None]]:
        profile_key = make_profile_key(record, profile_fields)
        descriptors: list[tuple[str, str, str | None]] = []
        for scope, fields in (
            ("actor", actor_fields),
            ("device", device_fields),
            ("peer_group", peer_fields),
        ):
            selector = _first_present(record, fields)
            if selector is None:
                continue
            selector_field, value = selector
            descriptors.append(
                (scope, _hierarchy_key(scope, profile_key, value), selector_field)
            )
        descriptors.append(("log_profile", profile_key, None))
        return descriptors

    @classmethod
    def fit(
        cls,
        records: Iterable[Mapping[str, object]],
        excluded_fields: frozenset[str],
        profile_fields: tuple[str, ...] = DEFAULT_PROFILE_FIELDS,
        max_categories: int = 4096,
        *,
        actor_fields: tuple[str, ...] = DEFAULT_ACTOR_FIELDS,
        device_fields: tuple[str, ...] = DEFAULT_DEVICE_FIELDS,
        peer_fields: tuple[str, ...] = DEFAULT_PEER_FIELDS,
        minimum_records: Mapping[str, int] | None = None,
    ) -> "BaselineRegistry":
        rows = list(records)
        configured_minimums = dict(DEFAULT_SCOPE_MINIMUM_RECORDS)
        configured_minimums.update(
            {key: int(value) for key, value in (minimum_records or {}).items()}
        )

        candidate_counts: Counter[tuple[str, str]] = Counter()
        for record in rows:
            for scope, key, _ in cls._candidate_descriptors(
                record, profile_fields, actor_fields, device_fields, peer_fields
            ):
                candidate_counts[(scope, key)] += 1

        profile_counts: Counter[tuple[str, str]] = Counter()
        observed: Counter[tuple[str, str, str]] = Counter()
        numeric_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        temporal_values: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        categorical_values: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

        excluded = {str(field).strip().lower() for field in excluded_fields}
        profile_metadata = {field.lower() for field in profile_fields}
        for record in rows:
            descriptors = cls._candidate_descriptors(
                record, profile_fields, actor_fields, device_fields, peer_fields
            )
            active = [
                (scope, key)
                for scope, key, _ in descriptors
                if scope == "log_profile"
                or candidate_counts[(scope, key)] >= configured_minimums[scope]
            ]
            for scope, key in active:
                profile_counts[(scope, key)] += 1
            for raw_name, value in record.items():
                name = str(raw_name).strip().lower()
                if name in excluded or name in profile_metadata or is_missing(value):
                    continue
                number = parse_number(value)
                time_bucket = None if number is not None else parse_time_bucket(value)
                for scope, key in active:
                    aggregate_key = (scope, key, name)
                    observed[aggregate_key] += 1
                    if number is not None:
                        numeric_values[aggregate_key].append(number)
                    else:
                        if time_bucket is not None:
                            temporal_values[aggregate_key][f"hour-of-week:{time_bucket}"] += 1
                        categorical_values[aggregate_key][value_fingerprint(value)] += 1

        all_profiles: dict[str, dict[str, ProfileBaseline]] = {
            scope: {} for scope in SCOPE_ORDER
        }
        for (scope, key), record_count in profile_counts.items():
            field_names = {
                field_name
                for current_scope, current_key, field_name in observed
                if current_scope == scope and current_key == key
            }
            fields: dict[str, FieldBaseline] = {}
            for field_name in field_names:
                aggregate_key = (scope, key, field_name)
                field_observed = observed[aggregate_key]
                numbers = numeric_values.get(aggregate_key, [])
                numeric_ratio = len(numbers) / max(1, field_observed)
                time_counts = temporal_values.get(aggregate_key, Counter())
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
                    counts = categorical_values.get(aggregate_key, Counter())
                    fields[field_name] = FieldBaseline(
                        observed=field_observed,
                        kind="categorical",
                        category_counts=dict(counts.most_common(max_categories)),
                        unique_count=len(counts),
                    )
            all_profiles[scope][key] = ProfileBaseline(
                records=record_count, fields=fields
            )

        return cls(
            profiles=all_profiles["log_profile"],
            profile_fields=profile_fields,
            hierarchical_profiles={
                scope: all_profiles[scope] for scope in SCOPE_ORDER[:-1]
            },
            actor_fields=actor_fields,
            device_fields=device_fields,
            peer_fields=peer_fields,
            minimum_records=configured_minimums,
        )

    def profile_key(self, record: Mapping[str, object]) -> str:
        """Return the legacy log-profile key used by score calibration."""

        return make_profile_key(record, self.profile_fields)

    def _profiles_for_scope(self, scope: str) -> dict[str, ProfileBaseline]:
        return self.profiles if scope == "log_profile" else self.hierarchical_profiles[scope]

    def resolve(
        self,
        record: Mapping[str, object],
        field_name: str | None = None,
    ) -> BaselineSelection:
        """Choose the most specific sufficiently supported UEBA baseline."""

        profile_key = self.profile_key(record)
        descriptors = self._candidate_descriptors(
            record,
            self.profile_fields,
            self.actor_fields,
            self.device_fields,
            self.peer_fields,
        )
        normalized_field = field_name.strip().lower() if field_name else None
        candidates: list[dict[str, object]] = []
        earlier_identifier_seen = False
        for scope, key, selector_field in descriptors:
            profile = self._profiles_for_scope(scope).get(key)
            required = self.minimum_records.get(scope, 1)
            field = None if profile is None or normalized_field is None else profile.fields.get(normalized_field)
            observations = 0 if field is None else field.observed
            sufficient_records = profile is not None and profile.records >= required
            sufficient_field = (
                normalized_field is None
                or (scope == "log_profile" and profile is not None)
                or (
                    field is not None
                    and observations >= required
                )
            )
            selected = sufficient_records and sufficient_field
            if profile is None:
                reason = "기준선 없음"
            elif not sufficient_records:
                reason = f"정상 표본 {profile.records}건으로 최소 {required}건 미만"
            elif normalized_field is not None and field is None and scope != "log_profile":
                reason = "해당 필드의 정상 기준 없음"
            elif normalized_field is not None and observations < required and scope != "log_profile":
                reason = f"필드 표본 {observations}건으로 최소 {required}건 미만"
            else:
                reason = "선택"
            candidates.append(
                {
                    "scope": scope,
                    "scope_label": SCOPE_LABELS_KO[scope],
                    "baseline_key": key,
                    "selector_field": selector_field,
                    "records": 0 if profile is None else profile.records,
                    "field_observations": observations,
                    "selected": selected,
                    "reason": reason,
                }
            )
            if selected:
                if normalized_field is None:
                    selection_reason = (
                        f"{SCOPE_LABELS_KO[scope]}의 정상 로그 {profile.records}건을 사용했다."
                    )
                elif observations > 0:
                    selection_reason = (
                        f"{SCOPE_LABELS_KO[scope]}에서 정상 로그 {profile.records}건과 "
                        f"해당 필드 관측 {observations}건을 사용했다."
                    )
                else:
                    selection_reason = (
                        f"{SCOPE_LABELS_KO[scope]}의 정상 로그 {profile.records}건에 "
                        "해당 필드가 없어 새 필드로 비교했다."
                    )
                return BaselineSelection(
                    profile_key=profile_key,
                    baseline_key=key,
                    scope=scope,
                    scope_label=SCOPE_LABELS_KO[scope],
                    selector_field=selector_field,
                    records=profile.records,
                    field_observations=observations,
                    fallback_used=earlier_identifier_seen,
                    selection_reason=selection_reason,
                    candidates=tuple(candidates),
                )
            if scope != "log_profile":
                earlier_identifier_seen = True

        return BaselineSelection(
            profile_key=profile_key,
            baseline_key=None,
            scope="unavailable",
            scope_label=SCOPE_LABELS_KO["unavailable"],
            selector_field=None,
            records=0,
            field_observations=0,
            fallback_used=earlier_identifier_seen,
            selection_reason="정상 표본이 없어 기준선을 선택하지 못했다.",
            candidates=tuple(candidates),
        )

    def selected_profile(
        self, selection: BaselineSelection
    ) -> ProfileBaseline | None:
        if selection.baseline_key is None or selection.scope == "unavailable":
            return None
        return self._profiles_for_scope(selection.scope).get(selection.baseline_key)

    @staticmethod
    def _features_from_profile(
        profile: ProfileBaseline | None, field_name: str, value: object
    ) -> list[float]:
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
            categorical_rarity = min(
                1.0, -math.log(max(probability, 1e-12)) / denominator
            )
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

    def features(self, profile_key: str, field_name: str, value: object) -> list[float]:
        """Return legacy log-profile features for checkpoint compatibility."""

        return self._features_from_profile(
            self.profiles.get(profile_key), field_name, value
        )

    def field_features(
        self, record: Mapping[str, object], field_name: str, value: object
    ) -> tuple[list[float], BaselineSelection]:
        selection = self.resolve(record, field_name)
        return (
            self._features_from_profile(
                self.selected_profile(selection), field_name, value
            ),
            selection,
        )

    def normal_reference_features_for_record(
        self, record: Mapping[str, object], field_name: str
    ) -> tuple[list[float], BaselineSelection]:
        """Return the selected baseline encoded as an ordinary observation.

        Categorical raw values are intentionally not stored in the registry. This
        representation preserves field presence and baseline support while
        neutralizing value rarity, numeric deviation and type mismatch.
        """

        selection = self.resolve(record, field_name)
        profile = self.selected_profile(selection)
        if profile is None:
            return [0.0] * BASELINE_FEATURE_DIM, selection
        field = profile.fields.get(field_name.strip().lower())
        if field is None:
            return [0.0] * BASELINE_FEATURE_DIM, selection
        support = math.tanh(math.log1p(field.observed) / 8.0)
        ready = 1.0 if field.observed >= 20 else field.observed / 20.0
        return [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, support, ready], selection

    def features_for_record(
        self, record: Mapping[str, object], field_name: str, value: object
    ) -> list[float]:
        return self.field_features(record, field_name, value)[0]

    def coverage(self, record: Mapping[str, object], excluded_fields: frozenset[str]) -> dict:
        excluded = {str(field).strip().lower() for field in excluded_fields}
        profile_metadata = {field.lower() for field in self.profile_fields}
        usable = [
            str(name).strip().lower()
            for name, value in record.items()
            if str(name).strip().lower() not in excluded
            and str(name).strip().lower() not in profile_metadata
            and not is_missing(value)
        ]
        selections = [self.resolve(record, name) for name in usable]
        known = sum(
            bool(
                (profile := self.selected_profile(selection))
                and name in profile.fields
            )
            for name, selection in zip(usable, selections, strict=True)
        )
        scope_counts = Counter(
            selection.scope for selection in selections if selection.scope != "unavailable"
        )
        dominant_scope = (
            scope_counts.most_common(1)[0][0] if scope_counts else "unavailable"
        )
        return {
            "profile_key": self.profile_key(record),
            "profile_known": bool(known),
            "usable_fields": len(usable),
            "known_fields": known,
            "coverage": known / max(1, len(usable)),
            "dominant_scope": dominant_scope,
            "dominant_scope_label": SCOPE_LABELS_KO[dominant_scope],
            "field_scope_counts": dict(sorted(scope_counts.items())),
            "fallback_fields": sum(selection.fallback_used for selection in selections),
        }

    def merged(self, other: "BaselineRegistry") -> "BaselineRegistry":
        hierarchical = {
            scope: {
                **self.hierarchical_profiles.get(scope, {}),
                **other.hierarchical_profiles.get(scope, {}),
            }
            for scope in SCOPE_ORDER[:-1]
        }
        return BaselineRegistry(
            profiles={**self.profiles, **other.profiles},
            profile_fields=self.profile_fields,
            hierarchical_profiles=hierarchical,
            actor_fields=self.actor_fields,
            device_fields=self.device_fields,
            peer_fields=self.peer_fields,
            minimum_records=self.minimum_records,
        )

    def to_dict(self) -> dict:
        def serialize(profiles: Mapping[str, ProfileBaseline]) -> dict:
            return {
                key: {
                    "records": profile.records,
                    "fields": {
                        name: asdict(field) for name, field in profile.fields.items()
                    },
                }
                for key, profile in profiles.items()
            }

        return {
            "version": 2,
            "profile_fields": list(self.profile_fields),
            "actor_fields": list(self.actor_fields),
            "device_fields": list(self.device_fields),
            "peer_fields": list(self.peer_fields),
            "minimum_records": dict(self.minimum_records),
            "profiles": serialize(self.profiles),
            "hierarchical_profiles": {
                scope: serialize(profiles)
                for scope, profiles in self.hierarchical_profiles.items()
            },
        }

    @classmethod
    def from_dict(cls, value: dict) -> "BaselineRegistry":
        def deserialize(serialized: Mapping[str, object]) -> dict[str, ProfileBaseline]:
            return {
                key: ProfileBaseline(
                    records=int(profile["records"]),
                    fields={
                        name: FieldBaseline(**field)
                        for name, field in profile["fields"].items()
                    },
                )
                for key, profile in serialized.items()
            }

        hierarchical = {
            scope: deserialize(profiles)
            for scope, profiles in value.get("hierarchical_profiles", {}).items()
        }
        return cls(
            profiles=deserialize(value.get("profiles", {})),
            profile_fields=tuple(value.get("profile_fields", DEFAULT_PROFILE_FIELDS)),
            hierarchical_profiles=hierarchical,
            actor_fields=tuple(value.get("actor_fields", DEFAULT_ACTOR_FIELDS)),
            device_fields=tuple(value.get("device_fields", DEFAULT_DEVICE_FIELDS)),
            peer_fields=tuple(value.get("peer_fields", DEFAULT_PEER_FIELDS)),
            minimum_records=value.get("minimum_records", DEFAULT_SCOPE_MINIMUM_RECORDS),
        )
