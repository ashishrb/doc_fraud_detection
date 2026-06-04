"""Agent 1 - Metadata Analysis [Mandatory].

Inspect PDF metadata (PyMuPDF + exiftool, gathered in Step 2). Flags:
  - producer/creator matches a known PDF editor,
  - author field equals the candidate name,
  - modification date after creation/issuance date.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import config
from pipeline.utils import safe_agent

AGENT_ID = "agent_1"


def _parse_pdf_date(val: str) -> datetime | None:
    if not val:
        return None
    m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", val)
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hh = int(m.group(4) or 0)
    mm = int(m.group(5) or 0)
    ss = int(m.group(6) or 0)
    try:
        return datetime(y, mo, d, hh, mm, ss)
    except ValueError:
        return None


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    meta = ctx.get("pdf_metadata", {}) or {}
    pmeta = meta.get("pymupdf", {}) or {}
    emeta = meta.get("exiftool", {}) or {}

    reasons: list[str] = []
    score = 0.0

    producer = " ".join(str(v) for v in [
        pmeta.get("producer", ""), pmeta.get("creator", ""),
        emeta.get("PDF:Producer", ""), emeta.get("PDF:Creator", ""),
        emeta.get("XMP:CreatorTool", ""),
    ]).lower()
    for editor in config.KNOWN_PDF_EDITORS:
        if editor in producer:
            reasons.append(f"producer/creator matches known PDF editor: '{editor}'")
            score = max(score, 0.6)
            break

    author = (pmeta.get("author", "") or emeta.get("PDF:Author", "")).strip().lower()
    cand = (ctx.get("candidate_id", "") or "").strip().lower()
    if author and cand and (author == cand or author in cand or cand in author):
        reasons.append(f"author field ('{author}') matches candidate identity")
        score = max(score, 0.7)

    created = _parse_pdf_date(pmeta.get("creationDate", "")
                             or emeta.get("PDF:CreateDate", ""))
    modified = _parse_pdf_date(pmeta.get("modDate", "")
                              or emeta.get("PDF:ModifyDate", ""))
    if created and modified and modified > created:
        delta = (modified - created).total_seconds()
        if delta > 60:
            reasons.append(
                f"modification date ({modified.isoformat()}) is after creation "
                f"date ({created.isoformat()})")
            score = max(score, 0.55)

    if not ctx.get("is_pdf"):
        return {"score": 0.0, "flagged": False,
                "detail": "non-PDF: metadata agent limited to image EXIF",
                "flagged_regions": []}

    flagged = score >= 0.5
    return {
        "score": round(score, 3),
        "flagged": flagged,
        "flagged_regions": [],  # metadata findings are document-level
        "detail": "; ".join(reasons) if reasons else "no metadata anomalies",
        "evidence": {"producer": producer.strip(), "author": author,
                     "created": str(created), "modified": str(modified)},
    }
