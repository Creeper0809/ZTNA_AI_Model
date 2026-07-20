"""Stream large gzip CSV files and produce a compact data profile."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

import pandas as pd


GROUP_COLUMNS = (
    "dataset",
    "source_type",
    "label",
    "is_attack",
    "event_type",
    "attack_category",
    "raw_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def counter_json(counter: Counter) -> dict[str, int]:
    return {
        "<missing>" if pd.isna(key) else str(key): int(value)
        for key, value in counter.most_common()
    }


def profile(path: Path, chunksize: int) -> dict:
    started = time.perf_counter()
    header = pd.read_csv(path, compression="gzip", nrows=0).columns.tolist()
    missing = Counter()
    groups = {column: Counter() for column in GROUP_COLUMNS if column in header}
    rows = 0
    event_time_min = None
    event_time_max = None

    reader = pd.read_csv(
        path,
        compression="gzip",
        chunksize=chunksize,
        dtype="string",
        keep_default_na=True,
        low_memory=False,
    )
    for chunk_index, chunk in enumerate(reader, start=1):
        rows += len(chunk)
        for column in header:
            missing[column] += int(chunk[column].isna().sum())
        for column, counts in groups.items():
            counts.update(chunk[column].value_counts(dropna=False).to_dict())
        if "event_time" in chunk:
            times = chunk["event_time"].dropna()
            if not times.empty:
                local_min = str(times.min())
                local_max = str(times.max())
                event_time_min = (
                    local_min if event_time_min is None else min(event_time_min, local_min)
                )
                event_time_max = (
                    local_max if event_time_max is None else max(event_time_max, local_max)
                )
        if chunk_index % 10 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"PROFILE_PROGRESS file={path.name} rows={rows} "
                f"rows_per_second={rows / elapsed:.0f}",
                flush=True,
            )

    # Reading to EOF also forces gzip to validate its CRC and uncompressed size.
    with path.open("rb") as raw:
        gzip_magic = raw.read(2).hex()
    elapsed = time.perf_counter() - started
    return {
        "path": str(path.resolve()),
        "compressed_bytes": path.stat().st_size,
        "gzip_magic": gzip_magic,
        "gzip_stream_validated_to_eof": True,
        "columns": header,
        "column_count": len(header),
        "rows": rows,
        "event_time_min": event_time_min,
        "event_time_max": event_time_max,
        "missing": {column: int(missing[column]) for column in header},
        "distributions": {
            column: counter_json(counts) for column, counts in groups.items()
        },
        "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": round(rows / elapsed),
    }


def main() -> None:
    args = parse_args()
    profiles = [profile(Path(name), args.chunksize) for name in args.files]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"files": profiles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for result in profiles:
        print(
            f"PROFILE_DONE file={Path(result['path']).name} rows={result['rows']} "
            f"seconds={result['elapsed_seconds']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
