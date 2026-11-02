"""Reasoning layer for CHIMERA Task 1.

Generates:
  - Ordinal confidence calibrated to clinical certainty
  - Expert group-specific variable weights
  - Minimally sufficient tool / section reveal sequences
  - Factual, hallucination-free free-text clinical rationales
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CONF_LEVELS = ["uncertain", "borderline", "clear"]
VARS = ["psa", "psad", "vol", "pirads", "dre", "fh", "comorbidity", "cspca", "age", "bx"]
THR_ENS = 0.48


def conf_prior(rows: list[str]) -> str:
    """Computes median confidence level from an expert group."""
    arr = sorted(CONF_LEVELS.index(c) for c in rows if isinstance(c, str))
    return CONF_LEVELS[arr[len(arr) // 2]]


def fit_priors(w_list: list[dict[str, str]]) -> dict[str, str]:
    """Computes mode variable weights from an expert group."""
    df_w = pd.DataFrame(w_list)
    return {v: df_w[v].mode()[0] for v in VARS}


def fit_policy(tr: np.ndarray, TL: pd.DataFrame, wrows: list[dict], crows: list[str]) -> dict:
    """Fits global and subgroup-stratified reasoning policies."""
    g = lambda v: [i for i in tr if TL.bx_pos.iloc[i] == v]
    return {
        "wp_global": fit_priors([wrows[i] for i in tr]),
        "wp": {v: fit_priors([wrows[i] for i in g(v)]) for v in (0, 1)},
        "cp": {v: conf_prior([crows[i] for i in g(v)]) for v in (0, 1)},
    }


def reasoning(route: str, bxp: int, policy: dict, kind: str = "route_honest") -> tuple[str, dict[str, str], list[str]]:
    """Generates confidence, variable weights, and section reveal sequence."""
    if kind == "global":
        return "clear", policy["wp_global"], ["radiology_report"]
    reveal = ["radiology_report", "previous_notes", "psa_trend"] if route == "2B_model" else ["radiology_report"]
    if kind == "route_data":
        conf = policy["cp"][bxp]
    else:  # route_honest
        conf = policy["cp"][0] if route in ("R1", "R2") else "borderline"
    return conf, policy["wp"][bxp], reveal


def _factor(v: str, s: dict, feats: dict):
    """Extracts human-readable clinical factor strings."""
    if v == "pirads":
        p = feats.get("pirads")
        if pd.isna(p): return None
        mm = feats.get("lesion_mm", 0)
        return (f"PI-RADS {int(p)} ({int(mm)} mm lesion)" if mm > 0 else f"PI-RADS {int(p)}",)
    elif v == "psad":
        psad = s.get("psad")
        if psad is not None and not pd.isna(psad): return (f"PSA density {float(psad):.2f}",)
    elif v == "psa":
        psa = s.get("psa")
        if psa is not None and not pd.isna(psa): return (f"PSA {float(psa):.1f} ng/mL",)
    elif v == "dre":
        dre = s.get("dre")
        if dre is not None and not pd.isna(dre) and str(dre).lower() != "nan": return (f"DRE: {str(dre).lower()}",)
    elif v == "cspca":
        c = s.get("cspca")
        if c is not None and not pd.isna(c): return (f"csPCa likelihood score {float(c):.2f}",)
    return None


def route_text(route: str, dec: str, risk: float, spec: dict, s: dict, feats: dict, isup: float) -> str:
    """Constructs a deterministic clinical rationale citing only factual patient values."""
    pi = _factor("pirads", s, feats)
    pi_txt = pi[0] if pi else "no suspicious MRI lesion identified"
    vals = [f[0] for f in (_factor(v, s, feats) for v in ("psad", "psa", "dre", "cspca")) if f][:3]
    vtxt = (" Supporting values: " + "; ".join(vals) + ".") if vals else ""
    isup_txt = f", prior ISUP grade {int(isup)}" if isup and isup > 0 else ""
    word = "recommended" if dec == "yes" else "not recommended"

    if route == "R1":
        return f"Biopsy not recommended: {pi_txt} indicates low suspicion for clinically significant cancer.{vtxt}"
    if route == "R2":
        return f"Biopsy recommended: {pi_txt} in a patient without a prior positive biopsy meets the guideline indication for biopsy.{vtxt}"
    if route == "2A":
        return f"The MRI finding is equivocal ({pi_txt}); the model risk score {risk:.2f} (threshold {THR_ENS:.2f}) gives: biopsy {word}.{vtxt}"
    if route == "2B_model":
        thr_txt = f" (threshold {spec['threshold']:.2f})" if spec and spec.get("threshold") is not None else ""
        return f"Known prostate cancer (prior positive biopsy{isup_txt}). A specialist model using prior ISUP grade and PSA trend gives risk {risk:.2f}{thr_txt}: biopsy {word}. MRI: {pi_txt}."
    return f"Known prostate cancer (prior positive biopsy{isup_txt}) with {pi_txt}. The available data do not discriminate reliably between re-biopsy and continued surveillance, so the default is to recommend biopsy; clinical review is advised."
