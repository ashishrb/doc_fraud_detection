"""Shared helpers: rasterisation, bbox math, safe agent execution, logging.

All page rasters are produced at config.RASTER_DPI (150 DPI by default) and all
agent bounding boxes are expressed in pixel coordinates relative to that raster,
as required by the HTML UI.
"""
from __future__ import annotations

import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("forgery_poc")


def render_to_rasters(src_path: Path, work_dir: Path) -> list[dict[str, Any]]:
    """Render a PDF/image to per-page PNG rasters at RASTER_DPI.

    Returns a list of page dicts: {page, width, height, raster_path}.
    PIL images are loaded lazily by callers via load_page_image().
    """
    import fitz  # PyMuPDF
    from PIL import Image

    work_dir.mkdir(parents=True, exist_ok=True)
    ext = src_path.suffix.lower()
    pages: list[dict[str, Any]] = []

    if ext == ".pdf":
        doc = fitz.open(src_path)
        zoom = config.RASTER_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out = work_dir / f"page_{i + 1}.png"
            pix.save(out)
            pages.append(
                {"page": i + 1, "width": pix.width, "height": pix.height,
                 "raster_path": str(out)}
            )
        doc.close()
    elif ext == ".docx":
        # Convert DOCX -> PDF via LibreOffice if available, else render text.
        pdf_path = _docx_to_pdf(src_path, work_dir)
        if pdf_path is not None:
            return render_to_rasters(pdf_path, work_dir)
        # Fallback: render extracted text onto a single raster.
        img = _docx_text_to_image(src_path)
        out = work_dir / "page_1.png"
        img.save(out)
        pages.append({"page": 1, "width": img.width, "height": img.height,
                      "raster_path": str(out)})
    else:  # raster image formats
        img = Image.open(src_path).convert("RGB")
        out = work_dir / "page_1.png"
        img.save(out)
        pages.append({"page": 1, "width": img.width, "height": img.height,
                      "raster_path": str(out)})

    return pages


def _docx_to_pdf(src_path: Path, work_dir: Path) -> Path | None:
    import shutil
    import subprocess

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        return None
    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir",
             str(work_dir), str(src_path)],
            check=True, capture_output=True, timeout=120,
        )
        cand = work_dir / (src_path.stem + ".pdf")
        return cand if cand.exists() else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("DOCX->PDF conversion failed: %s", exc)
        return None


def _docx_text_to_image(src_path: Path):
    from docx import Document  # type: ignore
    from PIL import Image, ImageDraw

    try:
        d = Document(str(src_path))
        text = "\n".join(p.text for p in d.paragraphs)
    except Exception:  # noqa: BLE001
        text = "[unable to extract DOCX text]"
    img = Image.new("RGB", (1240, 1754), "white")  # ~150 DPI A4
    draw = ImageDraw.Draw(img)
    y = 40
    for line in text.splitlines()[:80]:
        draw.text((40, y), line[:120], fill="black")
        y += 20
    return img


def load_page_image(raster_path: str):
    from PIL import Image

    return Image.open(raster_path).convert("RGB")


def clamp_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = bbox
    x0, x1 = sorted((max(0, min(x0, width)), max(0, min(x1, width))))
    y0, y1 = sorted((max(0, min(y0, height)), max(0, min(y1, height))))
    return [float(x0), float(y0), float(x1), float(y1)]


def heatmap_to_bboxes(heat: np.ndarray, threshold: float,
                      min_area_frac: float = 0.0008) -> list[list[float]]:
    """Convert a normalised [0,1] heatmap into connected-component bboxes."""
    import cv2

    mask = (heat >= threshold).astype(np.uint8) * 255
    if mask.sum() == 0:
        return []
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = heat.shape[:2]
    min_area = min_area_frac * h * w
    boxes: list[list[float]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area:
            continue
        boxes.append([float(x), float(y), float(x + bw), float(y + bh)])
    return boxes


def safe_agent(agent_id: str) -> Callable:
    """Decorator: never let an agent crash the pipeline (Rule 3 — graceful
    degradation). On exception returns the mandated error stub."""

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> dict[str, Any]:
            try:
                result = fn(*args, **kwargs)
                result.setdefault("agent_id", agent_id)
                result.setdefault("score", 0.0)
                result.setdefault("flagged", False)
                result.setdefault("flagged_regions", [])
                result.setdefault("detail", "")
                return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s failed: %s\n%s", agent_id, exc,
                               traceback.format_exc())
                return {
                    "agent_id": agent_id,
                    "score": 0.0,
                    "flagged": False,
                    "flagged_regions": [],
                    "error": str(exc),
                    "detail": f"{agent_id} unavailable: {exc}",
                }

        return wrapper

    return deco


def locate_text(value: str, words: list[dict[str, Any]]):
    """Find the bbox/page spanning the OCR words that make up `value`.

    Returns (bbox|None, page). Used by several agents to anchor a flagged
    string to a region in the 150-DPI raster.
    """
    import re

    tokens = [re.sub(r"[^a-z0-9]", "", t) for t in str(value).lower().split()]
    tokens = [t for t in tokens if t]
    if not tokens or not words:
        return None, 1
    norm = [(w, re.sub(r"[^a-z0-9]", "", w["text"].lower())) for w in words]
    first = tokens[0]
    for i, (w, wn) in enumerate(norm):
        if wn and (wn == first or first in wn):
            span = [w]
            for off in range(1, min(len(tokens), len(norm) - i)):
                span.append(norm[i + off][0])
            xs0 = [s["bbox"][0] for s in span]
            ys0 = [s["bbox"][1] for s in span]
            xs1 = [s["bbox"][2] for s in span]
            ys1 = [s["bbox"][3] for s in span]
            return [min(xs0), min(ys0), max(xs1), max(ys1)], span[0]["page"]
    return None, 1


def to_jsonable(obj: Any) -> Any:
    """Recursively coerce numpy/Path types into JSON-serialisable values."""
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj
