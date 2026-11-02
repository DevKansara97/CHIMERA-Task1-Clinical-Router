#!/usr/bin/env python3
"""Run 5-Fold Cross-Validation (5 Repeats) Benchmark for Variant B Clinical Router.

Computes the official CHIMERA Grand Challenge Task 1 evaluation metrics:
  - Task 1 Ranking Score = (Mean Case Score + Biopsy F1) / 2
  - Evaluated on all 91 labeled cases
  - Exports official submission predictions to results/predictions/variant_b_router/task1/
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Ensure package and evaluation modules are importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

warnings.filterwarnings("ignore")

os.environ["SECTION_MAPPING_FILE"] = str(REPO_ROOT / "evaluation/section_variable_mapping.json")

import evaluate
from src.router import (
    BASE_FEATURES, EXT_NOTES, EXT_PSA, THR_ENS,
    build_feature_table, decide, fit_specialist, make_ensemble, rule_route
)
from src.reasoning import fit_policy, reasoning, route_text

evaluate.USE_RATIONALE_JUDGE = False

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
GT_DIR = DATA_DIR / "ground_truth/task1"

print("=" * 80)
print("CHIMERA TASK 1 - VARIANT B CLINICAL ROUTER BENCHMARK")
print("=" * 80)

clean = pd.read_csv(DATA_DIR / "features_clean.csv")
raw = pd.read_csv(DATA_DIR / "features.csv")
audit = pd.read_csv(DATA_DIR / "reasoning_audit.csv").set_index("case_id")

T = build_feature_table(raw, clean)
is_lab = raw["label"].notna().values
TL = T[is_lab].reset_index(drop=True)
TU = T[~is_lab].reset_index(drop=True)
y = raw.loc[is_lab, "label"].astype(int).values
ids = raw.loc[is_lab, "case_id"].values

print(f"Cohort: {len(y)} labeled cases (yes-rate {y.mean():.2f}), {len(TU)} unlabelled cases")
print(f"Known-cancer cases (prior positive biopsy): {int(TL.bx_pos.sum())}/{len(y)}")

# Ground truth for official scoring
gt_records = evaluate.load_ground_truth_records(GT_DIR, "task1")
gts = {r["case_id"]: r for r in gt_records}
wrows = [gts[c]["variable_weights"] for c in ids]
crows = [gts[c]["confidence"] for c in ids]

raw_idx = raw.set_index("case_id")
TL_idx = TL.copy(); TL_idx.index = ids

R = 5
variants = [
    "V0 Ensemble (t=0.48)",
    "V1 Rule PI>=4",
    "V2 Rule + Ensemble",
    "V3 Variant B (route_honest)",
    "V3 Variant B (route_data)",
    "V3 Variant B (global reasoning)",
    "V3c Variant B (no specialist)",
]
scores = {v: [] for v in variants}
oof_predictions_variant_b = {}

print("\nRunning 5-fold cross-validation with 5 repeats ...")
for rep in range(R):
    P = {v: {} for v in variants}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(TL, y):
        ens = make_ensemble().fit(TL.iloc[tr][BASE_FEATURES], y[tr])
        gm = TL.iloc[tr].bx_pos.values == 1
        spec = fit_specialist(TL.iloc[tr][gm].reset_index(drop=True), y[tr][gm])
        policy = fit_policy(tr, TL, wrows, crows)

        Xte, cids, bxs = TL.iloc[te], ids[te], TL.iloc[te].bx_pos.values
        pe = ens.predict_proba(Xte[BASE_FEATURES])[:, 1]
        hi = (Xte.pirads.values >= 4)

        dec = {
            "V0 Ensemble (t=0.48)":           [("yes" if p >= THR_ENS else "no", "ens", float(p)) for p in pe],
            "V1 Rule PI>=4":                 [("yes" if h else "no", "rule", np.nan) for h in hi],
            "V2 Rule + Ensemble":            [("yes" if (h or p >= THR_ENS) else "no", "rule" if h else "ens", float(p)) for h, p in zip(hi, pe)],
            "V3":                            decide(Xte, ens, spec),
            "V3c Variant B (no specialist)": decide(Xte, ens, {"use": False}),
        }

        def _make_batch_preds(variant_key, kind):
            batch = {}
            for c, (dec_val, route_val, risk_val), b in zip(cids, dec[variant_key], bxs):
                cf, w, rev = reasoning(route_val, int(b), policy, kind)
                r_row = raw_idx.loc[c] if c in raw_idx.index else {}
                f_row = TL_idx.loc[c] if c in TL_idx.index else {}
                isup = f_row.get("prior_isup_grade", 0)
                txt = route_text(route_val, dec_val, risk_val, None, dict(r_row), dict(f_row), isup)
                batch[c] = {
                    "biopsy_decision": dec_val,
                    "confidence": cf,
                    "variable_weights": w,
                    "reveal_sequence": rev,
                    "free_text": txt,
                }
            return batch

        P["V0 Ensemble (t=0.48)"].update(_make_batch_preds("V0 Ensemble (t=0.48)", "global"))
        P["V1 Rule PI>=4"].update(_make_batch_preds("V1 Rule PI>=4", "global"))
        P["V2 Rule + Ensemble"].update(_make_batch_preds("V2 Rule + Ensemble", "global"))
        P["V3c Variant B (no specialist)"].update(_make_batch_preds("V3c Variant B (no specialist)", "route_data"))
        P["V3 Variant B (route_honest)"].update(_make_batch_preds("V3", "route_honest"))
        P["V3 Variant B (route_data)"].update(_make_batch_preds("V3", "route_data"))
        P["V3 Variant B (global reasoning)"].update(_make_batch_preds("V3", "global"))

        if rep == 0:
            for cid_k, pred_v in P["V3 Variant B (route_honest)"].items():
                oof_predictions_variant_b[cid_k] = pred_v

    for v in variants:
        cases = []
        for c in ids:
            r = evaluate.evaluate_case(gts[c], P[v][c], None, None)
            evaluate.attach_kappa_fields(r, gts[c], P[v][c])
            cases.append(r)
        agg = evaluate.compute_aggregate_metrics(cases)
        for k in ["confidence_score", "variable_weight_score", "important_decisive_factor_score", "tool_score", "section_grounding_score"]:
            vals = [r[k] for r in cases if r.get(k) is not None]
            agg[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0
        scores[v].append(agg)

    print(f"Repeat {rep + 1}/{R} complete.")

# Aggregate Scorecard
summary_rows = []
for v in variants:
    summary_rows.append({
        "Model Design": v,
        "Ranking Score": np.mean([s["ranking_score"] for s in scores[v]]),
        "Mean Case Score": np.mean([s["mean_case_score"] for s in scores[v]]),
        "Biopsy F1": np.mean([s["decision_f1_yes"] for s in scores[v]]),
        "Accuracy": np.mean([s["decision_accuracy"] for s in scores[v]]),
        "Confidence Score": np.mean([s["mean_confidence_score"] for s in scores[v]]),
        "Variable Weight": np.mean([s["mean_variable_weight_score"] for s in scores[v]]),
        "Factor F1": np.mean([s["mean_important_decisive_factor_score"] for s in scores[v]]),
        "Tool Score": np.mean([s["mean_tool_score"] for s in scores[v]]),
        "Grounding Score": np.mean([s["mean_section_grounding_score"] for s in scores[v]]),
    })

comparison_df = pd.DataFrame(summary_rows).set_index("Model Design").round(4)
print("\n" + "=" * 80)
print("OFFICIAL CHIMERA TASK 1 EVALUATION SCORECARD")
print("=" * 80)
print(comparison_df.to_string())

# Save Grand Challenge format predictions
pred_dir = RESULTS_DIR / "predictions/variant_b_router/task1"
pred_dir.mkdir(parents=True, exist_ok=True)
print(f"\nWriting official Grand Challenge formatted predictions to {pred_dir} ...")

for cid in ids:
    c_out = pred_dir / cid
    c_out.mkdir(parents=True, exist_ok=True)
    pdata = oof_predictions_variant_b[cid]
    
    with open(c_out / "prostate-biopsy-decision.json", "w") as f:
        json.dump(pdata["biopsy_decision"], f, indent=2)

    with open(c_out / "prostate-biopsy-decision-reasoning.json", "w") as f:
        reas_body = {
            "confidence": pdata["confidence"],
            "variable_weights": pdata["variable_weights"],
            "reveal_sequence": pdata["reveal_sequence"],
            "free_text": pdata["free_text"],
        }
        json.dump(reas_body, f, indent=2)

print(f"Generated {len(ids)} case prediction folders.")

# Save evaluation summary and per-case CSV
eval_cases = []
for cid in ids:
    gt = gts[cid]
    pred = oof_predictions_variant_b[cid]
    res = evaluate.evaluate_case(gt, pred, None, None)
    evaluate.attach_kappa_fields(res, gt, pred)
    eval_cases.append(res)

agg_official = evaluate.compute_aggregate_metrics(eval_cases)
for k in ["confidence_score", "variable_weight_score", "important_decisive_factor_score", "tool_score", "section_grounding_score"]:
    vals = [r[k] for r in eval_cases if r.get(k) is not None]
    agg_official[f"mean_{k}"] = float(np.mean(vals)) if vals else 0.0

per_case_rows = [{
    "case_id": r["case_id"],
    "case_score": round(r["case_score"], 4),
    "decision_correct": bool(r["decision_correct"]),
    "gt_decision": r["gt_decision"],
    "pred_decision": r["pred_decision"],
    "confidence_score": round(r.get("confidence_score") or 0.0, 4),
    "variable_weight_score": round(r.get("variable_weight_score") or 0.0, 4),
    "factor_score": round(r.get("important_decisive_factor_score") or 0.0, 4),
    "tool_score": round(r.get("tool_score") or 0.0, 4),
    "section_grounding_score": round(r.get("section_grounding_score") or 0.0, 4),
} for r in eval_cases]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(per_case_rows).to_csv(RESULTS_DIR / "variant_b_per_case_evaluation.csv", index=False)

summary_json = {
    "cohort_size": len(ids),
    "architecture": "Variant B Clinical-Context Router",
    "cv_strategy": "5-fold Stratified CV, 5 repeats",
    "scorecard": comparison_df.to_dict(orient="index"),
    "primary_oof_aggregate": agg_official,
}
with open(RESULTS_DIR / "variant_b_official_eval_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)

print(f"\nSaved scorecard to {RESULTS_DIR / 'variant_b_official_eval_summary.json'}")
print("Done!")
