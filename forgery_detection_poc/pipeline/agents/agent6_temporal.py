"""Agent 6 - Temporal Consistency [Mandatory].

Extract dates (regex + spaCy DATE entities), then validate:
  - no date inversions (a start date later than an end date),
  - no employment gaps beyond 6 months between consecutive dates,
  - no future-dated documents.
Each violation is anchored to the offending date field's bbox.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from pipeline.utils import locate_text, logger, safe_agent
from pipeline.nlp import get_spacy

AGENT_ID = "agent_6"

_DATE_RE = re.compile(
    r"\b(\d{1,2}[/-][A-Za-z0-9]{2,9}[/-]\d{2,4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|[A-Za-z]{3,9}\.?\s+\d{4}"
    r"|\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})\b")

_GAP_DAYS = 183  # ~6 months


def _parse(text: str) -> datetime | None:
    from dateutil import parser as dparser

    try:
        return dparser.parse(text, fuzzy=True, default=datetime(2000, 1, 1))
    except Exception:  # noqa: BLE001
        return None


def _collect_dates(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    text = ctx.get("text", "")
    found: dict[str, datetime] = {}
    for m in _DATE_RE.finditer(text):
        dt = _parse(m.group(0))
        if dt:
            found[m.group(0)] = dt
    nlp = get_spacy()
    if nlp is not None:
        try:
            for ent in nlp(text[:100000]).ents:
                if ent.label_ == "DATE":
                    dt = _parse(ent.text)
                    if dt:
                        found.setdefault(ent.text, dt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("spaCy date NER failed: %s", exc)
    out = []
    for raw, dt in found.items():
        bbox, page = locate_text(raw, ctx.get("words", []))
        out.append({"raw": raw, "dt": dt, "bbox": bbox, "page": page})
    return out


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    dates = _collect_dates(ctx)
    regions: list[dict[str, Any]] = []
    reasons: list[str] = []
    score = 0.0
    now = datetime.now()

    # Future-dated documents.
    for d in dates:
        if d["dt"] > now:
            score = max(score, 0.7)
            reasons.append(f"future-dated value '{d['raw']}'")
            if d["bbox"]:
                regions.append({"page": d["page"], "bbox": d["bbox"],
                                "reason": f"future date '{d['raw']}'",
                                "confidence": 0.7})

    # Inversions + gaps across consecutive parsed dates.
    ordered = [d for d in dates if d["dt"].year > 1900]
    seq = sorted(ordered, key=lambda d: d["dt"])
    raw_seq = ordered  # document order proxy: list order
    for i in range(len(raw_seq) - 1):
        a, b = raw_seq[i], raw_seq[i + 1]
        if a["dt"] > b["dt"]:
            delta = (a["dt"] - b["dt"]).days
            if delta > 1:
                score = max(score, 0.6)
                reasons.append(f"date inversion '{a['raw']}' > '{b['raw']}'")
                for d in (a, b):
                    if d["bbox"]:
                        regions.append({"page": d["page"], "bbox": d["bbox"],
                                        "reason": f"date inversion near '{d['raw']}'",
                                        "confidence": 0.55})
    for i in range(len(seq) - 1):
        gap = (seq[i + 1]["dt"] - seq[i]["dt"]).days
        if gap > _GAP_DAYS:
            score = max(score, 0.45)
            reasons.append(
                f"gap of {gap} days between '{seq[i]['raw']}' and "
                f"'{seq[i + 1]['raw']}'")

    return {
        "score": round(score, 3),
        "flagged": bool(regions) or score >= 0.45,
        "flagged_regions": regions,
        "detail": "; ".join(reasons) if reasons
                  else f"{len(dates)} date(s) consistent",
    }
