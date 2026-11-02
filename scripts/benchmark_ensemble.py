#!/usr/bin/env python3
"""Run 5-Fold Cross-Validation Benchmark for the Baseline 3-Model Ensemble.

Fits Logistic Regression + Random Forest + Gradient Boosting and evaluates
against the official CHIMERA ground truth.
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "evaluation"))

warnings.filterwarnings("ignore")

import evaluate
from src.router import BASE_FEATURES, build_feature_table, make_ensemble
from src.reasoning import fit_policy, reasoning, route_text

evaluate.USE_RATIONALE_JUDGE = False

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
GT_DIR = DATA_DIR / "ground_truth/task1"

clean = pd.read_csv(DATA_DIR / "features_clean.csv")
raw = pd.read_csv(DATA_DIR / "features.csv")
audit = pd.read_csv(DATA_DIR / "reasoning_audit.csv").set_index("case_id")

T = build_feature_table(raw, clean)
is_lab = raw["label"].notna().values
TL = T[is_lab].reset_index(drop=True)
y = raw.loc[is_lab, "label"].astype(int).values
ids = raw.loc[is_lab, "case_id"].values

gt_records = evaluate.load_ground_truth_records(GT_DIR, "task1")
gts = {r["case_id"]: r for r in gt_records}
wrows = [gts[c]["variable_weights"] for c in ids]
crows = [gts[c]["confidence"] for c in ids]

raw_idx = raw.set_index("case_id")
TL_idx = TL.copy(); TL_idx.index = ids

print("=" * 80)
print("CHIMERA TASK 1 - 3-MODEL ENSEMBLE BENCHMARK (t=0.48 vs t=0.67)")
print("=" * 80)

for threshold, name in [(0.48, "F1-Optimized (t=0.48)"), (0.67, "Youden-J (t=0.67)")]:
    scores = []
    for rep in range(5):
        preds = {}
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=rep).split(TL, y):
            ens = make_ensemble().fit(TL.iloc[tr][BASE_FEATURES], y[tr])
            policy = fit_policy(tr, TL, wrows, crows)
            Xte, cids, bxs = TL.iloc[te], ids[te], TL.iloc[te].bx_pos.values
            pe = ens.predict_proba(Xte[BASE_FEATURES])[:, 1]

            for c, p, b in zip(cids, pe, bxs):
                dec = "yes" if p >= threshold else "no"
                cf, w, rev = reasoning("2A", int(b), policy, "global")
                r_row = raw_idx.loc[c] if c in raw_idx.index else {}
                f_row = TL_idx.loc[c] if c in TL_idx.index else {}
                txt = route_text("2A", dec, float(p), None, dict(r_row), dict(f_row), 0)
                preds[c] = {
                    "biopsy_decision": dec,
                    "confidence": cf,
                    "variable_weights": w,
                    "reveal_sequence": rev,
                    "free_text": txt,
                }
        cases = []
        for c in ids:
            r = evaluate.evaluate_case(gts[c], preds[c], None, None)
            evaluate.attach_kappa_fields(r, gts[c], preds[c])
            cases.append(r)
        scores.append(evaluate.compute_aggregate_metrics(cases))

    rank = np.mean([s["ranking_score"] for s in scores])
    case = np.mean([s["mean_case_score"] for s in scores])
    f1 = np.mean([s["decision_f1_yes"] for s in scores])
    acc = np.mean([s["decision_accuracy"] for s in scores])
    print(f"Ensemble {name:22s} | Ranking: {rank:.4f} | Case: {case:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

print("=" * 80)
