"""Agent 12 - PDF Layer Analysis [Mandatory].

Catches native PDF post-issuance edits - text, images, or form fields added on
top of a locked or signed PDF after the original creation date. Agent 1 catches
metadata signals; Agent 12 catches the structural layer evidence Agent 1 misses.

Four checks (each produces a flagged_region):
  1. Incremental update layers   - extra cross-reference sections (pikepdf / raw).
  2. Annotation layers           - annotations modified >24h after creation (PyMuPDF).
  3. Overlapping text layers     - text blocks overlapping >30% (PyMuPDF).
  4. Form field post-fill        - AcroForm fields filled after creation (pikepdf).

Runs on every document. Non-PDF inputs return a neutral non-flagging result
immediately. Any error returns the standard error stub and never crashes the
pipeline.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import config
from pipeline.utils import safe_agent

AGENT_ID = "agent_12"

# Raster scale: agents express bboxes in RASTER_DPI px; PDF geometry is in points.
_ZOOM = config.RASTER_DPI / 72.0
_DAY_SECONDS = 24 * 3600


def _parse_pdf_date(val: str) -> datetime | None:
    if not val:
        return None
    m = re.search(r"(\d{4})(\d{2})(\d{2})(\d{2})?(\d{2})?(\d{2})?", str(val))
    if not m:
        return None
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    hh, mm, ss = int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0)
    try:
        return datetime(y, mo, d, hh, mm, ss)
    except ValueError:
        return None


def _doc_creation_date(ctx: dict[str, Any]) -> datetime | None:
    meta = ctx.get("pdf_metadata", {}) or {}
    pmeta = meta.get("pymupdf", {}) or {}
    emeta = meta.get("exiftool", {}) or {}
    return _parse_pdf_date(pmeta.get("creationDate", "")
                           or emeta.get("PDF:CreateDate", ""))


def _scaled_bbox(rect, page_w_px: float, page_h_px: float) -> list[float]:
    """Convert a PDF-point rect (x0, y0, x1, y1) to RASTER_DPI px, clamped."""
    x0, y0, x1, y1 = (float(rect[0]) * _ZOOM, float(rect[1]) * _ZOOM,
                      float(rect[2]) * _ZOOM, float(rect[3]) * _ZOOM)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return [max(0.0, min(x0, page_w_px)), max(0.0, min(y0, page_h_px)),
            max(0.0, min(x1, page_w_px)), max(0.0, min(y1, page_h_px))]


def _overlap_fraction(a: list[float], b: list[float]) -> float:
    """Intersection area as a fraction of the smaller block's area."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    smaller = min(area_a, area_b)
    return inter / smaller if smaller > 0 else 0.0


def _page_px_dims(ctx: dict[str, Any]) -> dict[int, tuple[float, float]]:
    return {p["page"]: (float(p["width"]), float(p["height"]))
            for p in ctx.get("pages", [])}


# --------------------------- Check 1 --------------------------- #
def _check_incremental_layers(path: str, ctx: dict[str, Any], regions: list) -> float:
    import pikepdf  # validates the PDF; raises on a corrupt file

    with pikepdf.open(path):
        pass
    with open(path, "rb") as fh:
        raw = fh.read()
    # Each cross-reference section ends with a `startxref` pointer; a legitimate
    # single-pass PDF has exactly one. Extra sections = incremental updates.
    count = raw.count(b"startxref")
    if count >= 2:
        dims = _page_px_dims(ctx)
        w, h = dims.get(1, (10_000.0, 10_000.0))
        regions.append({
            "page": 1,
            "bbox": [0.0, 0.0, w, h],
            "reason": f"PDF has {count} incremental update layers - "
                      f"post-issuance edits detected",
            "confidence": 0.7,
        })
    return min(1.0, (count - 1) * 0.3) if count >= 2 else 0.0


# --------------------------- Check 2 --------------------------- #
def _check_annotations(doc, ctx: dict[str, Any], regions: list) -> float:
    created = _doc_creation_date(ctx)
    dims = _page_px_dims(ctx)
    n_flagged = 0
    for page in doc:
        try:
            annots = page.annots()
        except Exception:  # noqa: BLE001
            annots = None
        if not annots:
            continue
        w, h = dims.get(page.number + 1, (10_000.0, 10_000.0))
        for annot in annots:
            info = annot.info or {}
            mod = _parse_pdf_date(info.get("modDate", ""))
            cre = _parse_pdf_date(info.get("creationDate", ""))
            ref = created or cre
            stamp = mod or cre
            if ref is None or stamp is None:
                continue
            delta = (stamp - ref).total_seconds()
            if delta > _DAY_SECONDS:
                n_flagged += 1
                delta_days = int(delta // _DAY_SECONDS)
                regions.append({
                    "page": page.number + 1,
                    "bbox": _scaled_bbox(annot.rect, w, h),
                    "reason": f"Annotation added/modified {delta_days} days "
                              f"after document creation",
                    "confidence": 0.75,
                })
    return min(1.0, n_flagged * 0.5)


# --------------------------- Check 3 --------------------------- #
def _check_text_overlaps(doc, ctx: dict[str, Any], regions: list) -> float:
    dims = _page_px_dims(ctx)
    n_overlaps = 0
    for page in doc:
        try:
            blocks = page.get_text("dict").get("blocks", [])
        except Exception:  # noqa: BLE001
            continue
        boxes = [b["bbox"] for b in blocks if b.get("type") == 0 and b.get("bbox")]
        w, h = dims.get(page.number + 1, (10_000.0, 10_000.0))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _overlap_fraction(list(boxes[i]), list(boxes[j])) > 0.30:
                    n_overlaps += 1
                    ix0 = max(boxes[i][0], boxes[j][0])
                    iy0 = max(boxes[i][1], boxes[j][1])
                    ix1 = min(boxes[i][2], boxes[j][2])
                    iy1 = min(boxes[i][3], boxes[j][3])
                    regions.append({
                        "page": page.number + 1,
                        "bbox": _scaled_bbox((ix0, iy0, ix1, iy1), w, h),
                        "reason": "Overlapping text blocks detected - possible "
                                  "field value substitution",
                        "confidence": 0.8,
                    })
    return min(1.0, n_overlaps * 0.5)


# --------------------------- Check 4 --------------------------- #
def _check_form_fields(path: str, ctx: dict[str, Any], regions: list) -> float:
    import pikepdf

    created = _doc_creation_date(ctx)
    dims = _page_px_dims(ctx)
    n_flagged = 0
    with pikepdf.open(path) as pdf:
        root = pdf.Root
        if "/AcroForm" not in root:
            return 0.0
        acro = root.AcroForm
        if "/Fields" not in acro:
            return 0.0
        for field in acro.Fields:
            try:
                has_value = "/V" in field and str(field.V).strip() not in ("", "/")
                mod = _parse_pdf_date(str(field.M)) if "/M" in field else None
                if not (has_value and mod):
                    continue
                ref = created
                if ref is not None and (mod - ref).total_seconds() <= _DAY_SECONDS:
                    continue
                name = str(field.T) if "/T" in field else "<unnamed>"
                page_no, bbox = 1, [0.0, 0.0, 10_000.0, 10_000.0]
                if "/Rect" in field:
                    w, h = dims.get(1, (10_000.0, 10_000.0))
                    bbox = _scaled_bbox([float(x) for x in field.Rect], w, h)
                n_flagged += 1
                regions.append({
                    "page": page_no,
                    "bbox": bbox,
                    "reason": f"Form field '{name}' filled after document creation",
                    "confidence": 0.85,
                })
            except Exception:  # noqa: BLE001
                continue
    return min(1.0, n_flagged * 0.5)


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    if not ctx.get("is_pdf"):
        return {"score": 0.0, "flagged": False, "flagged_regions": [],
                "detail": "skipped - not a native PDF"}

    path = ctx.get("path")
    if not path:
        return {"score": 0.0, "flagged": False, "flagged_regions": [],
                "detail": "skipped - no document path"}

    import fitz  # PyMuPDF

    regions: list[dict[str, Any]] = []

    incremental_score = _check_incremental_layers(path, ctx, regions)
    n_layers_regions = len(regions)

    doc = fitz.open(path)
    try:
        annotation_score = _check_annotations(doc, ctx, regions)
        n_annot = len(regions) - n_layers_regions
        overlap_score = _check_text_overlaps(doc, ctx, regions)
        n_overlaps = len(regions) - n_layers_regions - n_annot
    finally:
        doc.close()

    formfield_score = _check_form_fields(path, ctx, regions)
    n_fields = len(regions) - n_layers_regions - n_annot - n_overlaps

    score = min(1.0, (
        (incremental_score * 0.3)
        + (annotation_score * 0.3)
        + (overlap_score * 0.25)
        + (formfield_score * 0.15)
    ))
    flagged = score >= config.AGENT12_THRESHOLD and bool(regions)

    n_xref_layers = sum(1 for r in regions if "incremental update layers" in r["reason"])
    return {
        "score": round(score, 3),
        "flagged": flagged,
        "flagged_regions": regions,
        "detail": (f"PDF layer analysis: {n_xref_layers} XRef layers, "
                   f"{n_annot} post-creation annotations, {n_overlaps} text "
                   f"overlaps, {n_fields} late-filled fields"),
    }
