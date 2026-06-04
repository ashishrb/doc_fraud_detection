"""Agent 11 - Adversarial Robustness [Mandatory].

Maintains a local adversarial-pattern library (seeded with >=5 evasion
perturbations: JPEG double-compression, rotation, colour-channel shift, blur,
brightness shift). Applies each perturbation and re-runs Agent 2's ELA. If a
perturbation meaningfully changes the ELA heatmap (delta > AGENT11_DELTA_THRESHOLD)
the document is flagged as adversarially fragile / likely manipulated.
"""
from __future__ import annotations

import io
import json
from typing import Any

import numpy as np

import config
from pipeline.agents.agent2_image_forensics import _ela_heatmap
from pipeline.utils import load_page_image, safe_agent

AGENT_ID = "agent_11"
_MANIFEST = config.ADVERSARIAL_DIR / "patterns.json"

_SEED_PATTERNS = [
    {"name": "jpeg_double_compression", "param": 60},
    {"name": "rotation", "param": 1.5},
    {"name": "color_channel_shift", "param": 3},
    {"name": "gaussian_blur", "param": 1.2},
    {"name": "brightness_shift", "param": 1.15},
]


def _ensure_manifest() -> list[dict[str, Any]]:
    if _MANIFEST.exists():
        try:
            return json.loads(_MANIFEST.read_text())
        except Exception:  # noqa: BLE001
            pass
    _MANIFEST.write_text(json.dumps(_SEED_PATTERNS, indent=2))
    return _SEED_PATTERNS


def _apply(img, pattern: dict[str, Any]):
    from PIL import Image, ImageEnhance, ImageFilter

    name, param = pattern["name"], pattern["param"]
    if name == "jpeg_double_compression":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(param))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if name == "rotation":
        return img.rotate(float(param), expand=False, fillcolor=(255, 255, 255))
    if name == "color_channel_shift":
        arr = np.array(img).astype(np.int16)
        arr[..., 0] = np.clip(arr[..., 0] + int(param), 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] - int(param), 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
    if name == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=float(param)))
    if name == "brightness_shift":
        return ImageEnhance.Brightness(img).enhance(float(param))
    return img


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    patterns = _ensure_manifest()
    page = ctx["pages"][0]
    img = load_page_image(page["raster_path"])
    base = _ela_heatmap(img)

    deltas: dict[str, float] = {}
    max_delta = 0.0
    for pat in patterns:
        try:
            pert = _apply(img, pat)
            ela = _ela_heatmap(pert)
            if ela.shape != base.shape:
                import cv2

                ela = cv2.resize(ela, (base.shape[1], base.shape[0]))
            delta = float(np.mean(np.abs(base - ela)))
        except Exception:  # noqa: BLE001
            delta = 0.0
        deltas[pat["name"]] = round(delta, 4)
        max_delta = max(max_delta, delta)

    flagged = max_delta > config.AGENT11_DELTA_THRESHOLD
    score = float(np.clip(max_delta / max(config.AGENT11_DELTA_THRESHOLD, 1e-6)
                          * 0.6, 0, 1))
    return {
        "score": round(score, 3),
        "flagged": bool(flagged),
        "flagged_regions": [],  # document-level robustness signal
        "detail": f"max ELA delta under perturbation={max_delta:.3f} "
                  f"(threshold={config.AGENT11_DELTA_THRESHOLD}); per-pattern={deltas}",
    }
