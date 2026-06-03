"""Agent 10 - Cross-OCR Disagreement [Mandatory].

Consumes the Step 2 disagreement_vector, weights each field by criticality
(designation/employer 1.0, date/amount 0.9, name 0.7, other 0.3), and flags
fields whose weighted disagreement exceeds AGENT10_THRESHOLD. Disagreeing
fields are mapped back to bounding boxes from the Step 3 field extractor.
"""
from __future__ import annotations

from typing import Any

import config
from pipeline.utils import locate_text, safe_agent

AGENT_ID = "agent_10"


def _bbox_for_field(field: str, ctx: dict[str, Any], value: str | None):
    for f in ctx.get("understanding", {}).get("fields", []):
        if f["field"] == field and f.get("bbox"):
            return f["bbox"], f.get("page", 1)
    if value:
        return locate_text(value, ctx.get("words", []))
    return None, 1


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    vector = ctx.get("disagreement_vector", [])
    if not vector:
        return {"score": 0.0, "flagged": False, "flagged_regions": [],
                "detail": "no disagreement vector (need >=2 OCR engines)"}

    n_engines = max((v["n_engines"] for v in vector), default=0)
    if n_engines < 2:
        return {"score": 0.0, "flagged": False, "flagged_regions": [],
                "detail": f"only {n_engines} OCR engine(s) available; "
                          "cross-OCR disagreement needs >=2"}

    regions: list[dict[str, Any]] = []
    reasons: list[str] = []
    weighted_scores: list[float] = []

    for entry in vector:
        weighted = entry["disagreement"] * entry["criticality"]
        weighted_scores.append(weighted)
        if weighted > config.AGENT10_THRESHOLD:
            field = entry["field"]
            some_value = next(iter(entry.get("values", {}).values()), None)
            bbox, page = _bbox_for_field(field, ctx, some_value)
            reasons.append(
                f"field '{field}' disagrees across engines "
                f"(weighted={weighted:.2f}, values={entry.get('values')})")
            if bbox:
                regions.append({"page": page, "bbox": bbox,
                                "reason": f"OCR engines disagree on '{field}'",
                                "confidence": round(min(weighted, 1.0), 3)})

    score = float(max(weighted_scores)) if weighted_scores else 0.0
    return {
        "score": round(min(score, 1.0), 3),
        "flagged": score > config.AGENT10_THRESHOLD,
        "flagged_regions": regions,
        "detail": "; ".join(reasons) if reasons
                  else f"OCR engines agree across {len(vector)} field(s)",
    }
