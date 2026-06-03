"""Step 6 - Meta-Learner & Score Calibration.

- Assembles a feature vector from all agent scores + OOD + cross-doc count.
- Computes a weighted ensemble fraud score (LightGBM is wired as a passthrough
  weighted sum for the POC since no ground-truth labels are available).
- Applies isotonic-regression calibration only when >50 labelled examples
  exist (models/labels.json); otherwise uses the raw weighted score.
- Computes SHAP attributions for the top-5 contributing agents via a linear
  surrogate whose coefficients are the configured agent weights.
- Assigns the P0/P1/P2 band.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

import config
from pipeline.utils import logger

_LABELS = config.MODELS_DIR / "labels.json"
_AGENT_ORDER = [f"agent_{i}" for i in range(1, 12)]


def build_feature_vector(ctx: dict[str, Any],
                         contradiction_count: int) -> dict[str, float]:
    findings = ctx.get("agent_findings", {})
    feats: dict[str, float] = {}
    for aid in _AGENT_ORDER:
        f = findings.get(aid, {})
        feats[f"{aid}_score"] = float(f.get("score", 0.0))
        feats[f"{aid}_flagged"] = 1.0 if f.get("flagged") else 0.0
        feats[f"{aid}_nregions"] = float(len(f.get("flagged_regions", [])))
    ood = ctx.get("understanding", {}).get("ood", {})
    feats["ood_distance"] = float(ood.get("distance") or 0.0)
    feats["ood_is_ood"] = 1.0 if ood.get("is_ood") else 0.0
    feats["contradiction_count"] = float(contradiction_count)
    feats["n_pages"] = float(len(ctx.get("pages", [])))
    feats["duplicate_hash"] = 1.0 if ctx.get("duplicate_of_known_hash") else 0.0
    return feats


def _weighted_score(findings: dict[str, dict]) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (aggregated_score, scores, weights).

    Spec asks for a weighted ensemble in [0,1]. A plain weighted *average* over
    all 11 agents dilutes single-category detections (a forgery typically fires
    only a few relevant agents), pushing obvious fraud into P2. We therefore
    aggregate with a weight-exponented noisy-OR, which keeps a strong detection
    from being washed out while still honouring per-agent weights. The raw
    weighted average is still reported separately for transparency.
    """
    weights = np.array([config.AGENT_WEIGHTS[a] for a in _AGENT_ORDER])
    # Gate by each agent's own flag decision: a sub-threshold score (e.g. ELA's
    # mild response to text edges) is treated as no-signal so it cannot inflate
    # the ensemble. Flagged regions drive the verdict.
    scores = np.array([float(findings.get(a, {}).get("score", 0.0))
                       if findings.get(a, {}).get("flagged") else 0.0
                       for a in _AGENT_ORDER])
    w_norm = weights / weights.max()
    noisy_or = 1.0 - float(np.prod((1.0 - np.clip(scores, 0, 0.999)) ** w_norm))
    return noisy_or, scores, weights


def raw_weighted_average(findings: dict[str, dict], weights: np.ndarray) -> float:
    raw = np.array([float(findings.get(a, {}).get("score", 0.0))
                    for a in _AGENT_ORDER])
    return float((weights * raw).sum() / weights.sum())


def _lightgbm_passthrough(feats: dict[str, float], base: float) -> float:
    """LightGBM is instantiated but acts as a passthrough weighted sum for the
    POC (no labels). Kept so the calibration path is real and swappable."""
    try:
        import lightgbm as lgb  # noqa: F401
        # A trained model would go here; for the POC we return the weighted sum.
        return base
    except Exception:  # noqa: BLE001
        return base


def _calibrate(score: float) -> tuple[float, str]:
    if _LABELS.exists():
        try:
            data = json.loads(_LABELS.read_text())
            if isinstance(data, list) and len(data) > 50:
                from sklearn.isotonic import IsotonicRegression

                xs = [d["raw"] for d in data]
                ys = [d["label"] for d in data]
                iso = IsotonicRegression(out_of_bounds="clip").fit(xs, ys)
                return float(iso.predict([score])[0]), "isotonic"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Isotonic calibration skipped: %s", exc)
    return score, "raw (insufficient labels)"


def _shap_top_agents(scores: np.ndarray, weights: np.ndarray):
    try:
        import shap
        from sklearn.linear_model import LinearRegression

        coef = weights / weights.sum()
        surrogate = LinearRegression()
        # Fit surrogate exactly to the linear weighted sum.
        X_fit = np.random.RandomState(0).uniform(0, 1, size=(256, len(weights)))
        surrogate.fit(X_fit, X_fit @ coef)
        # Baseline = all-clean (zeros) so each attribution is the agent's own
        # marginal push toward the fraud verdict. A 0.5-mean background instead
        # makes non-firing high-weight agents dominate with large negative
        # contributions, which is technically valid but hides the real drivers.
        X_bg = np.zeros((32, len(weights)))
        explainer = shap.LinearExplainer(
            surrogate, X_bg, feature_perturbation="interventional")
        sv = explainer.shap_values(scores.reshape(1, -1))[0]
        contrib = {a: float(v) for a, v in zip(_AGENT_ORDER, sv)}
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP failed, using weight*score attribution: %s", exc)
        contrib = {a: float(w * s) for a, w, s in
                   zip(_AGENT_ORDER, weights, scores)}
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    return [{"agent_id": a, "agent_name": config.AGENT_NAMES[a],
             "contribution": round(v, 4)} for a, v in top], contrib


def assign_band(score: float) -> str:
    if score <= config.P2_MAX:
        return "P2"
    if score <= config.P1_MAX:
        return "P1"
    return "P0"


def meta_score(ctx: dict[str, Any], contradiction_count: int) -> dict[str, Any]:
    findings = ctx.get("agent_findings", {})
    base, scores, weights = _weighted_score(findings)
    raw_avg = raw_weighted_average(findings, weights)

    # cross-doc + OOD boosters
    ood = ctx.get("understanding", {}).get("ood", {})
    boost = min(0.1 * contradiction_count, 0.3)
    if ood.get("is_ood"):
        boost += 0.1
    raw = float(np.clip(base + boost, 0, 1))

    raw = _lightgbm_passthrough(build_feature_vector(ctx, contradiction_count), raw)
    calibrated, calib_method = _calibrate(raw)
    top_agents, contrib = _shap_top_agents(scores, weights)
    band = assign_band(calibrated)

    return {
        "fraud_score": round(calibrated, 4),
        "ensemble_noisy_or": round(base, 4),
        "raw_weighted_average": round(raw_avg, 4),
        "score_with_boost": round(raw, 4),
        "calibration": calib_method,
        "band": band,
        "top_agents": [t["agent_id"] for t in top_agents],
        "shap_top_agents": top_agents,
        "shap_contributions": {k: round(v, 4) for k, v in contrib.items()},
        "features": build_feature_vector(ctx, contradiction_count),
    }
