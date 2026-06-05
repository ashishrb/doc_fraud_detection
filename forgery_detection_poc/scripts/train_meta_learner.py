"""Train the LightGBM meta-learner on the synthetic labelled dataset.

Usage (from the forgery_detection_poc/ directory):

    python scripts/train_meta_learner.py

Requires: models/synthetic_labels.json (run generate_synthetic_labels.py first)
Output:   models/meta_learner.pkl              (LightGBM model)
          models/meta_learner_calibrator.pkl  (IsotonicRegression calibrator)
          models/meta_learner_meta.json       (feature names, train date, metrics)

This is a manual operator tool — it is NOT auto-run by the pipeline.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Make the package importable when run as `python scripts/train_meta_learner.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split  # noqa: E402

import config  # noqa: E402
from pipeline.utils import logger  # noqa: E402


def main() -> None:
    labels_path = config.MODELS_DIR / "synthetic_labels.json"
    if not labels_path.exists():
        raise SystemExit(
            f"{labels_path} not found. Run scripts/generate_synthetic_labels.py first.")

    data = json.loads(labels_path.read_text())
    feature_names = data["feature_names"]
    X = np.array(data["X"], dtype=np.float32)
    y = np.array(data["y"], dtype=np.int32)
    logger.info("Loaded %d samples with %d features", X.shape[0], X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Calibrate the test-set probabilities with isotonic regression.
    test_probs = model.predict_proba(X_test)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(test_probs, y_test)

    calibrated = calibrator.predict(test_probs)
    preds = (calibrated >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_test, calibrated))
    precision = float(precision_score(y_test, preds, zero_division=0))
    recall = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))

    logger.info("ROC-AUC=%.3f precision=%.3f recall=%.3f f1=%.3f",
                roc_auc, precision, recall, f1)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.META_LEARNER_PATH)
    joblib.dump(calibrator, config.META_LEARNER_CALIBRATOR_PATH)

    meta = {
        "feature_names": feature_names,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "roc_auc": round(roc_auc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "model_type": "LGBMClassifier",
        "calibrator": "IsotonicRegression",
        "label_source": "synthetic",
    }
    config.META_LEARNER_META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"Training complete. ROC-AUC={roc_auc:.3f}. "
          f"Model saved to {config.META_LEARNER_PATH}")


if __name__ == "__main__":
    main()
