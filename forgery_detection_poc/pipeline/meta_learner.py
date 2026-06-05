"""Step 6 - Meta-Learner & Score Calibration.

Uses a LightGBM binary classifier (trained by scripts/train_meta_learner.py)
to compute a calibrated fraud probability from agent scores + contradiction count.

Fallback: if no trained model exists (models/meta_learner.pkl absent), uses
the original weighted noisy-OR heuristic so the pipeline always produces a score.
The fallback is documented in SETUP.md §7.
"""
from __future__ import annotations

from typing import Any

import joblib
import numpy as np

import config
from pipeline.utils import logger

_MODEL = None
_CALIBRATOR = None
_META = None
_TRIED = False
_FEATURE_NAMES = [
    "agent_1", "agent_2", "agent_3", "agent_4", "agent_5", "agent_6",
    "agent_7", "agent_8", "agent_9", "agent_10", "agent_11", "agent_13",
    "n_contradictions",
]
# Feature names that map to a specific agent (i.e. all but n_contradictions).
_AGENT_FEATURES = [f for f in _FEATURE_NAMES if f != "n_contradictions"]


def _load_model() -> None:
    global _MODEL, _CALIBRATOR, _META, _TRIED
    if _TRIED:
        return
    _TRIED = True
    model_path = config.MODELS_DIR / "meta_learner.pkl"
    calib_path = config.MODELS_DIR / "meta_learner_calibrator.pkl"
    if model_path.exists() and calib_path.exists():
        try:
            _MODEL = joblib.load(model_path)
            _CALIBRATOR = joblib.load(calib_path)
            logger.info("Meta-learner: loaded LightGBM model from %s", model_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Meta-learner: failed to load model (%s); using "
                           "noisy-OR fallback", exc)
            _MODEL = None
    else:
        logger.info("Meta-learner: no trained model found; using noisy-OR "
                    "fallback. Run scripts/generate_synthetic_labels.py then "
                    "scripts/train_meta_learner.py.")


def _feature_vector(findings: dict, n_contradictions: int) -> np.ndarray:
    vec: list[float] = []
    for fname in _FEATURE_NAMES:
        if fname == "n_contradictions":
            vec.append(float(n_contradictions))
        else:
            vec.append(float(findings.get(fname, {}).get("score", 0.0)))
    return np.array(vec, dtype=np.float32).reshape(1, -1)


def _noisy_or_fallback(findings: dict, n_contradictions: int) -> float:
    """Original weighted heuristic - used when no trained model exists."""
    weights = config.AGENT_WEIGHTS
    total_w, total_score = 0.0, 0.0
    for aid, f in findings.items():
        if "error" in f:
            continue
        w = weights.get(aid, 0.5)
        s = float(f.get("score", 0.0))
        if f.get("flagged", False):
            total_score += w * s
        total_w += w
    raw = total_score / total_w if total_w > 0 else 0.0
    raw = min(1.0, raw + n_contradictions * 0.1)
    return round(raw, 3)


def assign_band(score: float) -> str:
    if score <= config.P2_MAX:
        return "P2"
    if score <= config.P1_MAX:
        return "P1"
    return "P0"


def _top_agents(findings: dict) -> list[str]:
    scored = [(aid, float(findings.get(aid, {}).get("score", 0.0)))
              for aid in _AGENT_FEATURES]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return [aid for aid, s in scored[:5] if s > 0.0]


def _format_contrib(contrib: dict[str, float]) -> list[dict[str, Any]]:
    top = sorted(contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
    return [{"agent_id": a, "agent_name": config.AGENT_NAMES.get(a, a),
             "contribution": round(float(v), 4)} for a, v in top]


def _compute_shap(findings: dict, n_contradictions: int) -> list[dict[str, Any]]:
    """SHAP attributions from the trained LightGBM model (TreeExplainer)."""
    try:
        import shap

        X = _feature_vector(findings, n_contradictions)
        explainer = shap.TreeExplainer(_MODEL)
        sv = explainer.shap_values(X)
        # Binary classifiers may return a list [class0, class1]; take class 1.
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv).reshape(-1)
        contrib = {name: float(sv[i]) for i, name in enumerate(_FEATURE_NAMES)
                   if name in _AGENT_FEATURES}
        return _format_contrib(contrib)
    except Exception as exc:  # noqa: BLE001
        logger.warning("SHAP (LightGBM) failed: %s; using weight*score", exc)
        return _fallback_shap(findings)


def _fallback_shap(findings: dict) -> list[dict[str, Any]]:
    weights = config.AGENT_WEIGHTS
    contrib = {aid: float(weights.get(aid, 0.5)
                          * findings.get(aid, {}).get("score", 0.0))
               for aid in _AGENT_FEATURES}
    return _format_contrib(contrib)


def meta_score(ctx: dict[str, Any], n_contradictions: int) -> dict[str, Any]:
    _load_model()
    findings = ctx.get("agent_findings", {})

    if _MODEL is not None and _CALIBRATOR is not None:
        try:
            X = _feature_vector(findings, n_contradictions)
            raw_prob = float(_MODEL.predict_proba(X)[0, 1])
            fraud_score = float(_CALIBRATOR.predict([raw_prob])[0])
            fraud_score = round(min(1.0, max(0.0, fraud_score)), 3)
            scorer = "lightgbm+isotonic"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Meta-learner inference failed (%s); using noisy-OR "
                           "fallback", exc)
            fraud_score = _noisy_or_fallback(findings, n_contradictions)
            scorer = "noisy_or_fallback"
    else:
        fraud_score = _noisy_or_fallback(findings, n_contradictions)
        scorer = "noisy_or_fallback"

    band = assign_band(fraud_score)
    shap_top = (_compute_shap(findings, n_contradictions)
                if _MODEL is not None else _fallback_shap(findings))

    return {
        "fraud_score": fraud_score,
        "band": band,
        "scorer": scorer,
        "top_agents": _top_agents(findings),
        "shap_top_agents": shap_top,
    }
