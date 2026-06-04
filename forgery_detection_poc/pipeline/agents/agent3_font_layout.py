"""Agent 3 - Font & Layout Analysis [Mandatory].

Extract font names/sizes per text line with pdfminer.six. Flags:
  - font name or size changes within a single line / field,
  - a line whose font family or size differs from the document's dominant
    font/size while that font is rare (classic field-edit / font-swap fraud),
  - text baseline (x0) deviating from the document's dominant left margin.

Bounding boxes are converted from pdfminer points (origin bottom-left) to
150-DPI pixel coordinates (origin top-left) to match the rest of the pipeline.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import config
from pipeline.utils import safe_agent

AGENT_ID = "agent_3"
_SCALE = config.RASTER_DPI / 72.0


def _base_family(fontname: str) -> str:
    name = re.sub(r"^[A-Z]{6}\+", "", fontname or "")  # strip subset prefix
    return name.split("-")[0].split(",")[0].lower()


def _line_fonts(line):
    from pdfminer.layout import LTChar

    fonts, families, sizes, n = set(), set(), set(), 0
    for ch in line:
        if isinstance(ch, LTChar) and ch.get_text().strip():
            fonts.add(ch.fontname)
            families.add(_base_family(ch.fontname))
            sizes.add(round(ch.size, 1))
            n += 1
    return fonts, families, sizes, n


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    if not ctx.get("is_pdf"):
        return {"score": 0.0, "flagged": False, "flagged_regions": [],
                "detail": "non-PDF: font/layout analysis requires a PDF"}

    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTTextLineHorizontal

    regions: list[dict[str, Any]] = []
    left_margins: list[float] = []
    line_records: list[dict[str, Any]] = []
    family_counter: Counter = Counter()
    size_counter: Counter = Counter()

    for pidx, page_layout in enumerate(extract_pages(ctx["path"]), start=1):
        page_h = page_layout.height
        for element in page_layout:
            if not isinstance(element, LTTextContainer):
                continue
            for line in element:
                if not isinstance(line, LTTextLineHorizontal):
                    continue
                text = line.get_text().strip()
                if not text:
                    continue
                fonts, families, sizes, nchar = _line_fonts(line)
                if nchar == 0:
                    continue
                x0, y0, x1, y1 = line.bbox
                bbox = [x0 * _SCALE, (page_h - y1) * _SCALE,
                        x1 * _SCALE, (page_h - y0) * _SCALE]
                left_margins.append(round(x0, 0))
                for fam in families:
                    family_counter[fam] += nchar
                for sz in sizes:
                    size_counter[sz] += nchar
                line_records.append({
                    "page": pidx, "text": text, "fonts": fonts,
                    "families": families, "sizes": sizes, "bbox": bbox, "x0": x0,
                })

    if not line_records:
        return {"score": 0.0, "flagged": False, "flagged_regions": [],
                "detail": "no extractable text lines (likely scanned image PDF)"}

    dominant_margin = Counter(left_margins).most_common(1)[0][0]
    dominant_family = family_counter.most_common(1)[0][0]
    dominant_size = size_counter.most_common(1)[0][0]
    total_chars = sum(family_counter.values())
    score = 0.0

    for rec in line_records:
        reasons: list[str] = []
        conf = 0.0
        if len(rec["fonts"]) > 1:
            reasons.append(f"multiple fonts in one line: {sorted(rec['fonts'])}")
            score = max(score, 0.6); conf = max(conf, 0.6)
        if len(rec["sizes"]) > 1:
            reasons.append(f"multiple font sizes in one line: {sorted(rec['sizes'])}")
            score = max(score, 0.55); conf = max(conf, 0.55)
        # font family differs from dominant AND is rare -> likely swapped field
        odd_fams = [f for f in rec["families"] if f != dominant_family]
        for fam in odd_fams:
            frac = family_counter.get(fam, 0) / max(total_chars, 1)
            if frac < 0.2:
                reasons.append(f"font '{fam}' differs from dominant "
                               f"'{dominant_family}' (used in {frac:.0%} of text)")
                score = max(score, 0.65); conf = max(conf, 0.65)
        # alignment deviation
        dev = abs(rec["x0"] - dominant_margin)
        if 6 < dev < 120 and len(rec["text"]) > 10 and not reasons:
            reasons.append(f"left margin deviates {dev:.0f}pt from dominant")
            score = max(score, 0.4); conf = max(conf, 0.4)
        if reasons:
            regions.append({
                "page": rec["page"], "bbox": rec["bbox"],
                "reason": "; ".join(reasons), "text": rec["text"][:80],
                "confidence": round(conf, 3),
            })

    return {
        "score": round(score, 3),
        "flagged": bool(regions),
        "flagged_regions": regions,
        "detail": f"{len(regions)} anomalous line(s); dominant font "
                  f"'{dominant_family}' size {dominant_size} margin "
                  f"{dominant_margin}pt",
    }
