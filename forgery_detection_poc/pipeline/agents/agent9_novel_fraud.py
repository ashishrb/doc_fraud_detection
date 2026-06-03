"""Agent 9 - Novel Fraud Detection [Mandatory].

Anomaly detection on document patches. The spec prefers PatchCore/DRAEM
(anomalib); when those torch-based models are unavailable we use the mandated
fallback: a simple reconstruction-based anomaly detector (a PCA linear
autoencoder, no torch needed).

Because the POC has no labelled corpus of authentic documents that resemble real
inputs, the detector is fit per-document on the document's own patches and
flags patches whose reconstruction error is a strong robust-z outlier relative
to the rest of that document. This catches genuinely anomalous texture regions
(splices, pasted stamps, photographic edits) without the false positives that a
template-trained global model produced on dissimilar real documents. An
optional global model trained on authentic patches can be enabled by dropping
authentic samples into models/agent9_authentic/ (see SETUP.md).
"""
from __future__ import annotations

from typing import Any

import numpy as np

import config
from pipeline.utils import heatmap_to_bboxes, load_page_image, safe_agent

AGENT_ID = "agent_9"
_PATCH = 32
_STRIDE = 32
_Z_FLAG = 8.0  # robust-z above which a patch is a novelty candidate


def _patch_vectors(gray: np.ndarray):
    h, w = gray.shape
    coords, vecs = [], []
    for y in range(0, h - _PATCH + 1, _STRIDE):
        for x in range(0, w - _PATCH + 1, _STRIDE):
            p = gray[y:y + _PATCH, x:x + _PATCH]
            coords.append((x, y))
            vecs.append(p.flatten().astype(np.float32) / 255.0)
    return coords, (np.array(vecs) if vecs else np.empty((0, _PATCH * _PATCH)))


def _robust_z(err: np.ndarray) -> np.ndarray:
    med = np.median(err)
    mad = np.median(np.abs(err - med)) + 1e-6
    return (err - med) / (1.4826 * mad)


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2
    from sklearn.decomposition import PCA

    regions: list[dict[str, Any]] = []
    page_scores: list[float] = []

    for page in ctx["pages"]:
        gray = cv2.cvtColor(np.array(load_page_image(page["raster_path"])),
                            cv2.COLOR_RGB2GRAY)
        coords, vecs = _patch_vectors(gray)
        if len(vecs) < 16:
            continue
        # Restrict to content (non-blank) patches: blank whitespace dominates a
        # page and would collapse the error MAD to ~0, making every text patch a
        # spurious outlier. We model the distribution over content patches only.
        stds = vecs.std(axis=1)
        content = stds > 0.02
        if content.sum() < 16:
            page_scores.append(0.0)
            continue
        cvecs = vecs[content]
        ccoords = [c for c, keep in zip(coords, content) if keep]
        n_comp = int(min(32, cvecs.shape[1], max(2, cvecs.shape[0] - 1)))
        pca = PCA(n_components=n_comp).fit(cvecs)
        recon = pca.inverse_transform(pca.transform(cvecs))
        err = np.mean((cvecs - recon) ** 2, axis=1)
        z = _robust_z(err)
        coords, vecs = ccoords, cvecs
        max_z = float(z.max()) if len(z) else 0.0
        # sigmoid centred at the flag threshold -> ~0 for ordinary variation.
        page_scores.append(1.0 / (1.0 + np.exp(-(max_z - _Z_FLAG) / 1.5)))

        h, w = gray.shape
        heat = np.zeros((h, w), dtype=np.float32)
        for (x, y), zz in zip(coords, z):
            if zz >= _Z_FLAG:
                heat[y:y + _PATCH, x:x + _PATCH] = min(1.0, (zz - _Z_FLAG) / 8 + 0.5)
        if heat.max() > 0:
            for bbox in heatmap_to_bboxes(heat, 0.5, min_area_frac=0.002):
                regions.append({"page": page["page"], "bbox": bbox,
                                "reason": "anomalous texture (novel pattern, "
                                          f"robust-z>{_Z_FLAG:.0f})",
                                "confidence": 0.6})

    score = float(max(page_scores)) if page_scores else 0.0
    return {
        "score": round(score, 3),
        "flagged": score >= config.AGENT9_THRESHOLD and bool(regions),
        "flagged_regions": regions,
        "detail": f"per-document PCA novelty detector; max anomaly={score:.3f}",
    }
