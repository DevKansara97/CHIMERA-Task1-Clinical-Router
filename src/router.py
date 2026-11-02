"""Clinical-Context Router Architecture for CHIMERA Task 1.

Routes patients into:
  - R1: Low suspicion (PI-RADS <= 2) -> Rule: No Biopsy
  - R2: High suspicion (PI-RADS >= 4, biopsy-naive) -> Rule: Yes Biopsy
  - 2A: Equivocal MRI (PI-RADS 3 or missing) -> Stage 2A 3-Model Ensemble (t = 0.48)
  - 2B: Known Cancer (Prior Positive Biopsy) -> Stage 2B Specialist (Inner-CV Signal Gated)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = [
    "age", "months", "cspca", "pirads", "n_comorbidities", "dre_abnormal",
    "psa_log", "psap_log", "psad_log", "psav_log", "vol_log", "lesion_mm",
    "lesion_found", "enc_followup", "enc_prior_pca", "enc_prior_neg",
    "bx_neg", "bx_none", "bx_pos"
]

EXT_PSA = [
    "psa_first_log", "psa_max_log", "psa_slope_per_step_log",
    "psa_last_over_first_log", "psa_trend_n"
]

EXT_NOTES = [
    "prior_isup_grade", "has_prior_isup", "active_surveillance", "n_previous_notes"
]

TINY_COLS = ["prior_isup_grade", "psa_last_over_first_log", "psa_slope_per_step_log"]
ENS_COLS = BASE_FEATURES + EXT_PSA + EXT_NOTES

GATE_AUC = 0.60
THR_ENS = 0.48


def rule_route(pirads: float, bx_pos: int) -> str:
    """Clinical router rule based on MRI lesion score and prior cancer history."""
    if not np.isnan(pirads) and pirads <= 2:
        return "R1"
    if bx_pos:
        return "2B"
    if not np.isnan(pirads) and pirads >= 4:
        return "R2"
    return "2A"


def make_ensemble() -> VotingClassifier:
    """3-Model Ensemble (Logistic Regression + Random Forest + Gradient Boosting)."""
    lr = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, max_iter=3000))
    ])
    rf = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42))
    ])
    gb = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("gb", GradientBoostingClassifier(n_estimators=50, learning_rate=0.05, max_depth=2, min_samples_leaf=5, random_state=42))
    ])
    return VotingClassifier([("lr", lr), ("rf", rf), ("gb", gb)], voting="soft")


def make_tiny() -> Pipeline:
    """Compact regularized Logistic Regression on 3 key features."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))


CANDS = {"tiny_lr": (TINY_COLS, make_tiny), "ensemble_ext": (ENS_COLS, make_ensemble)}


def inner_oof(make_fn, X_in: pd.DataFrame, y_in: np.ndarray, reps: int = 3):
    """Calculates out-of-fold probability estimates and AUC over inner folds."""
    p_oof, aucs = np.zeros(len(y_in)), []
    for s in range(reps):
        pp = np.zeros(len(y_in))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=100 + s).split(X_in, y_in):
            pp[te] = make_fn().fit(X_in.iloc[tr], y_in[tr]).predict_proba(X_in.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y_in, pp))
        p_oof += pp / reps
    return p_oof, float(np.mean(aucs))


def fit_specialist(Xg: pd.DataFrame, y_group: np.ndarray) -> dict:
    """Fits the known-cancer specialist model, guarded by an inner-CV AUC >= 0.60 gate."""
    if min(np.bincount(y_group, minlength=2)) < 5:
        return {"use": False, "auc": np.nan, "name": "-", "why": "too few cases"}
    best = None
    for name, (cols, make_fn) in CANDS.items():
        p_oof, auc = inner_oof(make_fn, Xg[cols], y_group)
        if best is None or auc > best["auc"]:
            best = {"name": name, "cols": cols, "make": make_fn, "auc": auc, "oof": p_oof}
    if best["auc"] < GATE_AUC:
        return {"use": False, "auc": best["auc"], "name": best["name"], "why": "below gate"}
    fpr, tpr, thr = roc_curve(y_group, best["oof"])
    best_thr = float(thr[np.argmax(tpr - fpr)])
    return {
        "use": True,
        "auc": best["auc"],
        "name": best["name"],
        "cols": best["cols"],
        "model": best["make"]().fit(Xg[best["cols"]], y_group),
        "threshold": best_thr,
    }


def decide(Trows: pd.DataFrame, ens_all, spec: dict) -> list[tuple[str, str, float]]:
    """Evaluates the clinical router on a batch of patient rows."""
    pe = ens_all.predict_proba(Trows[BASE_FEATURES])[:, 1]
    ps = spec["model"].predict_proba(Trows[spec["cols"]])[:, 1] if spec.get("use") else None
    out = []
    for i, (pir, bxp) in enumerate(zip(Trows.pirads.values, Trows.bx_pos.values)):
        r = rule_route(pir, bxp)
        if r == "R1":
            out.append(("no", "R1", np.nan))
        elif r == "R2":
            out.append(("yes", "R2", np.nan))
        elif r == "2A":
            out.append(("yes" if pe[i] >= THR_ENS else "no", "2A", float(pe[i])))
        elif spec.get("use"):
            out.append(("yes" if ps[i] >= spec["threshold"] else "no", "2B_model", float(ps[i])))
        else:
            out.append(("yes", "2B_default", np.nan))
    return out


def build_feature_table(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the unified feature matrix for the router."""
    Fb = pd.DataFrame(index=clean_df.index)
    for c in ["age", "months", "cspca", "pirads", "n_comorbidities", "dre_abnormal", "psa_log", "psap_log", "psad_log", "psav_log", "vol_log"]:
        Fb[c] = clean_df[c]
    Fb["lesion_mm"] = clean_df["lesion_max_mm"]
    Fb["lesion_found"] = (clean_df["lesion_max_mm"] > 0).astype(float)
    Fb["enc_followup"] = clean_df["enc_type_Follow-up - PSA + MRI"]
    Fb["enc_prior_pca"] = clean_df["enc_type_Re-evaluation - prior PCa diagnosis"]
    Fb["enc_prior_neg"] = clean_df["enc_type_Re-evaluation - prior negative biopsy"]
    Fb["bx_neg"] = clean_df["bx_cat_Negative"]
    Fb["bx_none"] = clean_df["bx_cat_None"]
    Fb["bx_pos"] = clean_df["bx_cat_Positive"]

    T = pd.concat([Fb, clean_df[EXT_PSA + EXT_NOTES]], axis=1)
    T["bx_pos"] = (raw_df["bx"] == "Positive").astype(int).values
    return T
