#!/usr/bin/env python3
"""CLI utility to evaluate a directory of predictions using the official CHIMERA evaluator.

Usage:
  python scripts/evaluate_predictions.py --pred results/predictions/variant_b_router/task1/ --gt data/ground_truth/task1/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "evaluation"))
import evaluate

evaluate.USE_RATIONALE_JUDGE = False


def main():
    parser = argparse.ArgumentParser(description="Official CHIMERA Task 1 Evaluator")
    parser.add_argument("--pred", type=Path, default=REPO_ROOT / "results/predictions/variant_b_router/task1", help="Path to predictions folder")
    parser.add_argument("--gt", type=Path, default=REPO_ROOT / "data/ground_truth/task1", help="Path to ground truth folder")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    print(f"Loading ground truth from {args.gt} ...")
    gt_records = evaluate.load_ground_truth_records(args.gt, "task1")
    gt_map = {r["case_id"]: r for r in gt_records}
    print(f"Loaded {len(gt_map)} ground truth cases.")

    print(f"Evaluating predictions in {args.pred} ...")
    evaluated_cases = []
    for cid, gt in gt_map.items():
        case_dir = args.pred / cid
        dec_file = case_dir / "prostate-biopsy-decision.json"
        reas_file = case_dir / "prostate-biopsy-decision-reasoning.json"

        if not dec_file.exists() or not reas_file.exists():
            continue

        with open(dec_file) as f:
            decision = json.load(f)
        with open(reas_file) as f:
            reasoning = json.load(f)

        pred = dict(reasoning)
        pred["biopsy_decision"] = decision
        pred["case_id"] = cid

        res = evaluate.evaluate_case(gt, pred, None, None)
        evaluate.attach_kappa_fields(res, gt, pred)
        evaluated_cases.append(res)

    agg = evaluate.compute_aggregate_metrics(evaluated_cases)
    for k in ["confidence_score", "variable_weight_score", "important_decisive_factor_score", "tool_score", "section_grounding_score"]:
        vals = [r[k] for r in evaluated_cases if r.get(k) is not None]
        agg[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0

    print("\n" + "=" * 60)
    print("OFFICIAL EVALUATION METRICS")
    print("=" * 60)
    print(f"  * Task 1 Ranking Score:         {agg.get('ranking_score', 0.0) or 0.0:.4f}")
    print(f"  * Mean Case Score:              {agg.get('mean_case_score', 0.0) or 0.0:.4f}")
    print(f"  * Biopsy F1 Score (Positive):   {agg.get('decision_f1_yes', 0.0) or 0.0:.4f}")
    print(f"  * Decision Accuracy:            {agg.get('decision_accuracy', 0.0) or 0.0:.4f} ({agg.get('n_decision_correct', 0)}/{agg.get('n_evaluated', 0)})")
    print(f"  * Variable Weight Score:        {agg.get('mean_variable_weight_score', 0.0) or 0.0:.4f}")
    print(f"  * Confidence Score:             {agg.get('mean_confidence_score', 0.0) or 0.0:.4f}")
    print(f"  * Tool Efficiency Score:        {agg.get('mean_tool_score', 0.0) or 0.0:.4f}")
    print(f"  * Section Grounding Score:      {agg.get('mean_section_grounding_score', 0.0) or 0.0:.4f}")
    print("=" * 60)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"Saved evaluation metrics to {args.out}")


if __name__ == "__main__":
    main()
