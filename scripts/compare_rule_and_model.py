"""Compare the rule-based model and the trained field-attention model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from ztna_ueba.rule_baseline import evaluate_paper_rule_baseline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--familiar-zone", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    event = json.loads(input_path.read_text(encoding="utf-8"))
    rule_result = evaluate_paper_rule_baseline(
        event,
        familiar_geo_zones=args.familiar_zone or ("seoul",),
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ztna_ueba.predict",
            "--checkpoint",
            args.checkpoint,
            "--input",
            str(input_path),
            "--top-k",
            str(args.top_k),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    model_result = json.loads(completed.stdout)
    rule_stage = rule_result["policy"]["stage"]
    model_stage = model_result["policy"]["stage"]
    disagreement_direction = f"rule_{rule_stage}_model_{model_stage}"
    observed_metrics = rule_result["coverage"]["observed_or_project_mapped_metrics"]
    total_metrics = rule_result["coverage"]["total_metrics"]
    assumed_metrics = total_metrics - observed_metrics
    result = {
        "input": event,
        "rule_baseline": rule_result,
        "proposed_model": {
            "trust_score": model_result["trust_score"],
            "risk_score": model_result["risk_score"],
            "confidence": model_result["confidence"],
            "shadow_mode_required": model_result["shadow_mode_required"],
            "policy": model_result["policy"],
            "top_fields": model_result["top_fields"],
        },
        "comparison": {
            "rule_outcome": rule_result["policy"]["action"],
            "proposed_outcome": model_result["policy"]["action"],
            "rule_trust_score": rule_result["trust_score"],
            "proposed_trust_score": model_result["trust_score"],
            "disagreement_direction": disagreement_direction,
            "interpretation": (
                "This is an observed policy disagreement between the deterministic scorecard "
                "and the trained checkpoint. It does not establish which decision is correct "
                "without an independent ground-truth or operational adjudication label."
            ),
            "fairness_note": (
                f"The rule baseline observes or project-maps {observed_metrics} of "
                f"{total_metrics} paper sub-metrics; the other {assumed_metrics} receive favorable "
                "normal scores. The two policy scales are reported as defined, without forcing "
                "equivalent thresholds."
            ),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
