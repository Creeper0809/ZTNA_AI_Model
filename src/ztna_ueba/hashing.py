"""Stable hashing helpers used instead of a closed field vocabulary."""

from __future__ import annotations

import hashlib
import re


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def stable_bucket(text: str, buckets: int, namespace: bytes) -> int:
    """Return a deterministic bucket id, reserving zero for padding."""
    digest = hashlib.blake2b(
        text.encode("utf-8", errors="replace"), digest_size=8, person=namespace[:16]
    ).digest()
    return int.from_bytes(digest, "little") % buckets + 1


def normalize_text(value: object) -> str:
    return " ".join(TOKEN_RE.findall(str(value).strip().lower()))


def field_name_pieces(name: str, limit: int = 12) -> list[str]:
    normalized = normalize_text(name).replace(" ", "_")
    pieces = [f"whole:{normalized}"]
    words = TOKEN_RE.findall(normalized)
    pieces.extend(f"word:{word}" for word in words)
    compact = "_".join(words)
    for width in (3, 4, 5):
        if len(compact) < width:
            continue
        pieces.extend(
            f"char:{compact[index:index + width]}"
            for index in range(len(compact) - width + 1)
        )
    return list(dict.fromkeys(pieces))[:limit]


def value_pieces(value: object, limit: int = 8) -> list[str]:
    normalized = normalize_text(value)
    if not normalized:
        return []
    words = normalized.split()
    pieces = [f"whole:{normalized}"]
    pieces.extend(f"word:{word}" for word in words)
    return list(dict.fromkeys(pieces))[:limit]
