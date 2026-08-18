"""Run one Trust Score prediction and emit field-level explanations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .inference import TrustPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", help="JSON file; stdin is used when omitted")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload_text = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    payload = json.loads(payload_text)
    predictor = TrustPredictor(args.checkpoint, device=args.device)
    result = predictor.predict(payload, top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
