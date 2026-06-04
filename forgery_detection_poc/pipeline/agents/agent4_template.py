"""Agent 4 - Template Matching [Mandatory].

Compare the incoming document raster against the closest authentic template for
its document type using OpenCV matchTemplate (global similarity) and SSIM
(local structural deviation). Regions with SSIM below AGENT4_SSIM_THRESHOLD are
flagged (e.g. mislocated logos, edited blocks).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import config
from pipeline.template_store import get_template_store
from pipeline.utils import heatmap_to_bboxes, load_page_image, safe_agent

AGENT_ID = "agent_4"


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    import cv2
    from skimage.metrics import structural_similarity as ssim

    doc_type = ctx.get("classified_type", ctx.get("doc_type", "other"))
    # TemplateStore hides the source (local dir or Azure Blob); switching is a
    # config-only change via TEMPLATE_SOURCE.
    templates = get_template_store().get_templates(doc_type)
    if not templates:
        return {"score": 0.0, "flagged": False, "flagged_regions": [],
                "detail": f"no authentic templates available for '{doc_type}'"}

    page = ctx["pages"][0]
    doc_img = cv2.cvtColor(np.array(load_page_image(page["raster_path"])),
                           cv2.COLOR_RGB2GRAY)

    best = {"score": -1.0, "path": None, "ssim_map": None, "ssim": 0.0}
    for tpath in templates:
        tmpl = cv2.imread(str(tpath), cv2.IMREAD_GRAYSCALE)
        if tmpl is None:
            continue
        tmpl_r = cv2.resize(tmpl, (doc_img.shape[1], doc_img.shape[0]))
        match = float(cv2.matchTemplate(doc_img, tmpl_r,
                                        cv2.TM_CCOEFF_NORMED).max())
        s, smap = ssim(doc_img, tmpl_r, full=True)
        if match > best["score"]:
            best = {"score": match, "path": str(tpath), "ssim_map": smap,
                    "ssim": float(s)}

    if best["ssim_map"] is None:
        return {"score": 0.0, "flagged": False, "flagged_regions": [],
                "detail": "templates unreadable"}

    deviation = 1.0 - best["ssim_map"]  # high where structure differs
    deviation = np.clip(deviation, 0, 1)
    flag_level = 1.0 - config.AGENT4_SSIM_THRESHOLD
    regions = []
    for bbox in heatmap_to_bboxes(deviation, flag_level, min_area_frac=0.004):
        regions.append({
            "page": page["page"], "bbox": bbox,
            "reason": f"structural deviation from template "
                      f"(SSIM<{config.AGENT4_SSIM_THRESHOLD})",
            "confidence": 0.5,
        })

    doc_score = float(np.clip(1.0 - best["ssim"], 0, 1))
    flagged = best["ssim"] < config.AGENT4_SSIM_THRESHOLD
    return {
        "score": round(doc_score, 3),
        "flagged": bool(flagged and regions),
        "flagged_regions": regions,
        "detail": f"best template '{Path(best['path']).name}' "
                  f"SSIM={best['ssim']:.3f} matchTemplate={best['score']:.3f}",
    }
