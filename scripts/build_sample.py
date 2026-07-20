"""Build a deterministic dataset/label-stratified sample from a large gzip CSV."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import pandas as pd


HASH_COLUMNS = (
    "dataset",
    "event_time",
    "user_id_hash",
    "device_id_hash",
    "session_id",
    "src_ip_hash",
    "source_file",
)
# Public NDR sets often collapse millions of flows onto only a few synthetic
# device/IP identifiers.  Session is therefore the strongest usable leakage
# boundary here.  Production UEBA data must replace this with entity-time split.
IDENTITY_COLUMNS = ("session_id", "user_id_hash", "device_id_hash", "src_ip_hash")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--per-group", type=int, default=5000)
    parser.add_argument("--chunksize", type=int, default=100000)
    parser.add_argument("--max-scan-rows", type=int)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def stable_split(group_id: str, seed: int) -> str:
    digest = hashlib.blake2b(
        f"{seed}:{group_id}".encode(), digest_size=8, person=b"ztna-split"
    ).digest()
    bucket = int.from_bytes(digest, "little") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def group_identity(row: pd.Series) -> str:
    dataset = str(row.get("dataset", "unknown"))
    for column in IDENTITY_COLUMNS:
        value = row.get(column)
        if pd.notna(value) and str(value).strip():
            return f"{dataset}:{column}:{value}"
    return f"{dataset}:row:{int(row['_sample_hash'])}"


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    reservoirs: dict[tuple[str, str], pd.DataFrame] = {}
    counts = Counter()
    source_counts = Counter()
    rows = 0
    started = time.perf_counter()
    reader = pd.read_csv(
        source,
        compression="gzip",
        dtype="string",
        chunksize=args.chunksize,
        low_memory=False,
        nrows=args.max_scan_rows,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        rows += len(chunk)
        chunk = chunk[chunk["label"].isin(["normal", "abnormal"])].copy()
        for (dataset, label), count in chunk.groupby(["dataset", "label"]).size().items():
            counts[(str(dataset), str(label))] += int(count)
        source_counts.update(chunk["source_type"].value_counts(dropna=False).to_dict())
        available_hash_columns = [column for column in HASH_COLUMNS if column in chunk]
        row_hash = pd.util.hash_pandas_object(
            chunk[available_hash_columns].fillna(""), index=False
        ).astype("uint64")
        chunk["_sample_hash"] = row_hash ^ args.seed
        for key, group in chunk.groupby(["dataset", "label"], dropna=False):
            candidate = group.nsmallest(args.per_group, "_sample_hash")
            previous = reservoirs.get((str(key[0]), str(key[1])))
            if previous is not None:
                candidate = pd.concat([previous, candidate], ignore_index=True).nsmallest(
                    args.per_group, "_sample_hash"
                )
            reservoirs[(str(key[0]), str(key[1]))] = candidate
        if chunk_index % 20 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"SAMPLE_PROGRESS rows={rows} groups={len(reservoirs)} "
                f"rows_per_second={rows / elapsed:.0f}",
                flush=True,
            )

    if not reservoirs:
        raise RuntimeError("no supported normal/abnormal rows were found")
    sample = pd.concat(reservoirs.values(), ignore_index=True)
    sample["_group_id"] = sample.apply(group_identity, axis=1)
    sample["split"] = sample["_group_id"].map(lambda value: stable_split(value, args.seed))
    sample = sample.sort_values(["dataset", "label", "_sample_hash"]).reset_index(drop=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output, index=False, compression="gzip")

    split_counts = (
        sample.groupby(["split", "dataset", "label"]).size().sort_index().to_dict()
    )
    for dataset in sample["dataset"].dropna().unique():
        labels = set(sample.loc[sample["dataset"] == dataset, "label"].dropna())
        for split in ("train", "validation", "test"):
            present = set(
                sample.loc[
                    (sample["dataset"] == dataset) & (sample["split"] == split), "label"
                ].dropna()
            )
            if labels == {"normal", "abnormal"} and present != labels:
                raise RuntimeError(
                    f"class missing after split: dataset={dataset} split={split} "
                    f"present={sorted(present)}"
                )
    profile = {
        "input": str(source.resolve()),
        "input_bytes": source.stat().st_size,
        "rows_scanned": rows,
        "sample_rows": len(sample),
        "per_group_limit": args.per_group,
        "seed": args.seed,
        "full_counts": {f"{key[0]}|{key[1]}": int(value) for key, value in counts.items()},
        "source_counts": {
            "<missing>" if pd.isna(key) else str(key): int(value)
            for key, value in source_counts.items()
        },
        "sample_split_counts": {
            "|".join(map(str, key)): int(value) for key, value in split_counts.items()
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "gzip_stream_validated_to_eof": args.max_scan_rows is None,
    }
    profile_path = Path(args.profile)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"SAMPLE_DONE scanned={rows} sample={len(sample)} groups={len(reservoirs)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
