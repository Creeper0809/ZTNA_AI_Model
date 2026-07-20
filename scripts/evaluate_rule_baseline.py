"""Evaluate one JSON event with the paper-derived deterministic baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ztna_ueba.rule_baseline import evaluate_paper_rule_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to one JSON event")
    parser.add_argument(
        "--familiar-zone",
        action="append",
        default=None,
        help="Known normal zone; may be repeated (default: seoul)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = evaluate_paper_rule_baseline(
        event,
        familiar_geo_zones=args.familiar_zone or ("seoul",),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
