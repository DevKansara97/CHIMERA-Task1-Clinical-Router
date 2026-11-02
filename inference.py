#!/usr/bin/env python3
"""Grand Challenge entrypoint for the Variant B Clinical Router (Task 1).

Reads inputs from /input:
  - structured-prompt.json
  - prostate-biopsy-decision-clinical-data.json
  - prostate-modality-level-neural-representations.json (optional)

Writes outputs to /output:
  - prostate-biopsy-decision.json
  - prostate-biopsy-decision-reasoning.json
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.router import (
    BASE_FEATURES, EXT_NOTES, EXT_PSA, THR_ENS,
    build_feature_table, decide, fit_specialist, make_ensemble, rule_route
)
from src.reasoning import fit_policy, reasoning, route_text

INPUT_PATH = Path(os.environ.get("CHIMERA_INPUT", "/input"))
OUTPUT_PATH = Path(os.environ.get("CHIMERA_OUTPUT", "/output"))
DATA_PATH = Path(os.environ.get("CHIMERA_DATA", str(REPO_ROOT / "data")))


def load_json(name: str) -> dict:
    f = INPUT_PATH / name
    if f.exists():
        with open(f) as fp:
            return json.load(fp)
    return {}


def parse_case_features(structured: dict, clinical: dict) -> tuple[dict, pd.Series]:
    """Parses input JSONs into the required router feature dictionary."""
    age = float(structured.get("age", 65))
    psa = float(structured.get("psa", 5.0))
    pirads = float(structured.get("pirads", np.nan))
    psad = float(structured.get("psad", psa / 40.0 if psa else 0.15))
    vol = float(structured.get("vol", 40.0))
    cspca = float(structured.get("cspca", np.nan))
    dre = structured.get("dre", "Normal")
    months = float(structured.get("months", 0.0))
    bx = structured.get("bx", None)
    enc_type = structured.get("enc_type", "Follow-up - PSA + MRI")
    pmhx = structured.get("pmhx", [])
    n_comorb = len(pmhx) if isinstance(pmhx, list) else 0

    rad_report = clinical.get("radiology_report", "")
    prev_notes = clinical.get("previous_notes", "")
    psa_trend = clinical.get("psa_trend", [])

    dre_abn = 1.0 if str(dre).lower() in ["abnormal", "enlarged", "nodular", "nodus"] else 0.0
    bx_pos = 1 if str(bx).lower() == "positive" or "prior pca" in str(enc_type).lower() else 0
    bx_neg = 1 if str(bx).lower() == "negative" else 0
    bx_none = 1 if bx is None or pd.isna(bx) else 0

    # Parse lesion size
    lesion_mm = 0.0
    if rad_report and "mm" in rad_report:
        import re
        nums = re.findall(r"(\d+)\s*(?:x|×)\s*(\d+)", rad_report)
        if nums:
            lesion_mm = float(max(int(nums[0][0]), int(nums[0][1])))

    # PSA kinetics
    psa_slope = 0.0
    psa_ratio = 1.0
    if isinstance(psa_trend, list) and len(psa_trend) >= 2:
        vals = [p.get("psa", psa) for p in psa_trend if isinstance(p, dict) and "psa" in p]
        if len(vals) >= 2:
            psa_ratio = vals[-1] / max(vals[0], 0.1)
            psa_slope = (vals[-1] - vals[0]) / max(len(vals) - 1, 1)

    row = {
        "age": age,
        "months": months,
        "cspca": cspca,
        "pirads": pirads,
        "n_comorbidities": n_comorb,
        "dre_abnormal": dre_abn,
        "psa_log": np.log1p(psa),
        "psap_log": np.log1p(psa * 0.2),
        "psad_log": np.log1p(psad),
        "psav_log": np.log1p(abs(psa_slope)),
        "vol_log": np.log1p(vol),
        "lesion_mm": lesion_mm,
        "lesion_found": 1.0 if lesion_mm > 0 else 0.0,
        "enc_followup": 1.0 if "follow-up" in str(enc_type).lower() else 0.0,
        "enc_prior_pca": 1.0 if "prior pca" in str(enc_type).lower() else 0.0,
        "enc_prior_neg": 1.0 if "prior negative" in str(enc_type).lower() else 0.0,
        "bx_neg": float(bx_neg),
        "bx_none": float(bx_none),
        "bx_pos": float(bx_pos),
        "psa_first_log": np.log1p(psa),
        "psa_max_log": np.log1p(psa),
        "psa_slope_per_step_log": np.log1p(abs(psa_slope)),
        "psa_last_over_first_log": np.log1p(psa_ratio),
        "psa_trend_n": float(len(psa_trend) if isinstance(psa_trend, list) else 1),
        "prior_isup_grade": 0.0,
        "has_prior_isup": 0.0,
        "active_surveillance": 1.0 if bx_pos else 0.0,
        "n_previous_notes": 1.0 if prev_notes else 0.0,
    }
    s_dict = {"age": age, "psa": psa, "psad": psad, "vol": vol, "dre": dre, "cspca": cspca, "bx": bx}
    return s_dict, pd.Series(row)


def main():
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    structured = load_json("structured-prompt.json")
    clinical = load_json("prostate-biopsy-decision-clinical-data.json")

    # Train router on local reference data
    clean = pd.read_csv(DATA_PATH / "features_clean.csv")
    raw = pd.read_csv(DATA_PATH / "features.csv")
    T = build_feature_table(raw, clean)
    is_lab = raw["label"].notna().values
    TL = T[is_lab].reset_index(drop=True)
    y = raw.loc[is_lab, "label"].astype(int).values

    audit = pd.read_csv(DATA_PATH / "reasoning_audit.csv").set_index("case_id")
    ids = raw.loc[is_lab, "case_id"].values
    wrows = [audit.loc[c][[f"w_{v}" for v in ["psa", "psad", "vol", "pirads", "dre", "fh", "comorbidity", "cspca", "age", "bx"]]].rename(lambda x: x[2:]).to_dict() for c in ids]
    crows = [audit.loc[c, "confidence"] for c in ids]

    ens_final = make_ensemble().fit(TL[BASE_FEATURES], y)
    gm = TL.bx_pos.values == 1
    spec_final = fit_specialist(TL[gm].reset_index(drop=True), y[gm])
    policy_final = fit_policy(np.arange(len(y)), TL, wrows, crows)

    s_dict, row_feats = parse_case_features(structured, clinical)
    df_row = pd.DataFrame([row_feats])

    decisions = decide(df_row, ens_final, spec_final)
    dec, route, risk = decisions[0]
    conf, weights, reveals = reasoning(route, int(row_feats["bx_pos"]), policy_final, "route_honest")
    free_txt = route_text(route, dec, risk, spec_final, s_dict, dict(row_feats), row_feats["prior_isup_grade"])

    reasoning_payload = {
        "confidence": conf,
        "variable_weights": weights,
        "reveal_sequence": reveals,
        "free_text": free_txt,
    }

    with open(OUTPUT_PATH / "prostate-biopsy-decision.json", "w") as f:
        json.dump(dec, f, indent=2)

    with open(OUTPUT_PATH / "prostate-biopsy-decision-reasoning.json", "w") as f:
        json.dump(reasoning_payload, f, indent=2)

    print(f"Decision: {dec} (route: {route}, confidence: {conf})")


if __name__ == "__main__":
    main()
