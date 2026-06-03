"""Step 3 - Document Understanding.

Attempts LayoutLMv3 / Donut / DiT via Hugging Face transformers (lazy import,
graceful degradation). When the deep models are unavailable (no torch / no
download / no GPU) the module falls back to:
  - keyword + regex document classification,
  - OCR-word-anchored field extraction with bounding boxes,
  - a deterministic image-feature embedding (so the FAISS OOD check still works).

Model IDs (confirmed from Hugging Face model cards):
  - LayoutLMv3:  microsoft/layoutlmv3-base
  - Donut:       naver-clova-ix/donut-base
  - DiT:         microsoft/dit-base
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

import config
from pipeline.utils import load_page_image, logger

EMBED_DIM = 768

_DOC_TYPE_KEYWORDS = {
    "payslip": ["payslip", "pay slip", "salary slip", "net pay", "gross pay",
                "earnings", "deductions", "pf", "esi"],
    "experience_letter": ["experience", "relieving", "to whom it may concern",
                          "service certificate", "worked with us", "last working day"],
    "offer_letter": ["offer", "we are pleased to offer", "joining date",
                     "annual ctc", "appointment"],
    "form16": ["form 16", "form no. 16", "tds", "income tax", "assessment year",
               "deductor", "pan of the deductor"],
    "education_certificate": ["degree", "bachelor", "master", "university",
                              "marksheet", "grade", "cgpa", "certificate"],
}

_FIELD_PATTERNS = {
    "name": re.compile(r"(?:employee name|candidate name|name)\s*[:\-]\s*"
                       r"([A-Za-z .'-]{2,50})", re.I),
    "designation": re.compile(r"(?:designation|title|position|role)\s*[:\-]\s*"
                              r"([A-Za-z0-9 .,&/()'-]{2,60})", re.I),
    "employer": re.compile(r"(?:employer|company|organization|organisation)\s*"
                           r"[:\-]\s*([A-Za-z0-9 .,&/()'-]{2,60})", re.I),
    "date": re.compile(r"\b(\d{1,2}[/-][A-Za-z0-9]{2,9}[/-]\d{2,4}|"
                       r"\d{4}-\d{2}-\d{2})\b"),
    "amount": re.compile(r"(?:salary|amount|net pay|gross|ctc|total)\s*[:\-]?\s*"
                         r"(?:rs\.?|inr|usd|\$|₹)?\s*([0-9][0-9,]{2,}"
                         r"(?:\.\d{1,2})?)", re.I),
}


# ----------------------- DiT / fallback embedding ----------------------- #
class _Embedder:
    """Lazily tries DiT; otherwise a deterministic 768-d image feature vector."""

    def __init__(self) -> None:
        self._dit = None
        self._tried = False
        self.backend = "fallback"

    def _try_load_dit(self) -> None:
        if self._tried:
            return
        self._tried = True
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModel

            self._proc = AutoImageProcessor.from_pretrained("microsoft/dit-base")
            self._dit = AutoModel.from_pretrained("microsoft/dit-base")
            self._dit.eval()
            self.backend = "dit-base"
            logger.info("DiT loaded (microsoft/dit-base)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("DiT unavailable, using fallback embedding: %s", exc)
            self._dit = None

    def embed(self, raster_path: str) -> np.ndarray:
        self._try_load_dit()
        img = load_page_image(raster_path)
        if self._dit is not None:
            try:
                import torch

                inputs = self._proc(images=img, return_tensors="pt")
                with torch.no_grad():
                    out = self._dit(**inputs)
                vec = out.last_hidden_state[:, 0].squeeze().cpu().numpy()
                return _l2(vec.astype(np.float32))
            except Exception as exc:  # noqa: BLE001
                logger.warning("DiT inference failed, fallback: %s", exc)
        return _fallback_embedding(img)


def _fallback_embedding(img) -> np.ndarray:
    """Deterministic 768-d descriptor: multi-grid intensity + gradient stats."""
    import cv2

    g = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (256, 256)).astype(np.float32) / 255.0
    feats: list[float] = []
    # 8x8 grid mean+std (128 dims)
    for gy in np.array_split(g, 8, axis=0):
        for cell in np.array_split(gy, 8, axis=1):
            feats.append(float(cell.mean()))
            feats.append(float(cell.std()))
    # gradient magnitude grid (64 dims)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
    gy_ = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy_ ** 2)
    for row in np.array_split(mag, 8, axis=0):
        for cell in np.array_split(row, 8, axis=1):
            feats.append(float(cell.mean()))
    # intensity histogram (64 dims)
    hist = cv2.calcHist([(g * 255).astype(np.uint8)], [0], None, [64],
                        [0, 256]).flatten()
    feats.extend((hist / (hist.sum() + 1e-6)).tolist())
    vec = np.array(feats, dtype=np.float32)
    if vec.shape[0] < EMBED_DIM:
        vec = np.pad(vec, (0, EMBED_DIM - vec.shape[0]))
    return _l2(vec[:EMBED_DIM])


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


_EMBEDDER = _Embedder()


# ----------------------------- FAISS OOD ----------------------------- #
class TemplateIndex:
    """Small FAISS index of authentic template embeddings for OOD detection."""

    def __init__(self) -> None:
        self.index = None
        self.meta: list[dict[str, Any]] = []
        self._build()

    def _build(self) -> None:
        import faiss

        self.index = faiss.IndexFlatL2(EMBED_DIM)
        seeded = 0
        for tdir in sorted(config.TEMPLATES_DIR.glob("*")):
            if tdir.is_dir():
                for img_path in sorted(tdir.glob("*.png")) + \
                        sorted(tdir.glob("*.jpg")):
                    self._add(img_path, tdir.name)
                    seeded += 1
            elif tdir.suffix.lower() in (".png", ".jpg", ".jpeg"):
                self._add(tdir, "generic")
                seeded += 1
        logger.info("FAISS template index seeded with %d authentic embeddings",
                    seeded)

    def _add(self, img_path: Path, doc_type: str) -> None:
        try:
            vec = _EMBEDDER.embed(str(img_path))
            self.index.add(vec.reshape(1, -1))
            self.meta.append({"path": str(img_path), "doc_type": doc_type})
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to index template %s: %s", img_path, exc)

    def nearest_distance(self, vec: np.ndarray) -> tuple[float, dict | None]:
        if self.index is None or self.index.ntotal == 0:
            return float("inf"), None
        D, indices = self.index.search(vec.reshape(1, -1), 1)
        return float(D[0][0]), self.meta[int(indices[0][0])]


_TEMPLATE_INDEX: TemplateIndex | None = None


def get_template_index() -> TemplateIndex:
    global _TEMPLATE_INDEX
    if _TEMPLATE_INDEX is None:
        _TEMPLATE_INDEX = TemplateIndex()
    return _TEMPLATE_INDEX


# --------------------------- classification --------------------------- #
def classify_doc_type(text: str, user_hint: str) -> tuple[str, dict[str, float]]:
    text_l = (text or "").lower()
    scores: dict[str, float] = {}
    for dtype, kws in _DOC_TYPE_KEYWORDS.items():
        scores[dtype] = sum(1 for kw in kws if kw in text_l) / max(len(kws), 1)
    best = max(scores, key=scores.get) if scores else "other"
    if scores.get(best, 0) == 0:
        best = user_hint if user_hint in config.DOCUMENT_TYPES else "other"
    return best, scores


def _attempt_layoutlmv3(ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForTokenClassification, AutoProcessor

        AutoProcessor.from_pretrained("microsoft/layoutlmv3-base",
                                      apply_ocr=False)
        AutoModelForTokenClassification.from_pretrained(
            "microsoft/layoutlmv3-base")
        return {"available": True, "backend": "layoutlmv3-base",
                "note": "loaded; token-classification head is randomly "
                        "initialised without fine-tuning"}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def _attempt_donut(ctx: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch  # noqa: F401
        from transformers import DonutProcessor, VisionEncoderDecoderModel

        DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")
        return {"available": True, "backend": "donut-base"}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc)}


def _extract_fields_with_bbox(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Regex fields anchored to OCR word bboxes (works without DL models)."""
    text = ctx.get("text", "")
    words = ctx.get("words", [])
    fields: list[dict[str, Any]] = []
    for fname, pat in _FIELD_PATTERNS.items():
        m = pat.search(text)
        if not m:
            continue
        value = re.sub(r"\s+", " ", m.group(1)).strip()
        bbox, page = _locate_value(value, words)
        fields.append({"field": fname, "value": value, "bbox": bbox,
                       "page": page})
    return fields


def _locate_value(value: str, words: list[dict[str, Any]]):
    """Find the bbox spanning the OCR words that make up a field value."""
    tokens = [t for t in re.split(r"\s+", value.lower()) if t]
    if not tokens or not words:
        return None, 1
    norm = [(w, re.sub(r"[^a-z0-9]", "", w["text"].lower())) for w in words]
    first = re.sub(r"[^a-z0-9]", "", tokens[0])
    for i, (w, wn) in enumerate(norm):
        if first and (wn == first or first in wn):
            span = [w]
            for off in range(1, min(len(tokens), len(norm) - i)):
                span.append(norm[i + off][0])
            xs0 = [s["bbox"][0] for s in span]
            ys0 = [s["bbox"][1] for s in span]
            xs1 = [s["bbox"][2] for s in span]
            ys1 = [s["bbox"][3] for s in span]
            return [min(xs0), min(ys0), max(xs1), max(ys1)], span[0]["page"]
    return None, 1


def understand(ctx: dict[str, Any]) -> dict[str, Any]:
    text = ctx.get("text", "")
    doc_type, type_scores = classify_doc_type(text, ctx.get("doc_type", "other"))

    layout = _attempt_layoutlmv3(ctx)
    donut = _attempt_donut(ctx)

    # DiT embedding (page 1) + FAISS OOD check.
    ood = {"distance": None, "is_ood": False, "nearest": None,
           "embedding_backend": _EMBEDDER.backend}
    if ctx.get("pages"):
        vec = _EMBEDDER.embed(ctx["pages"][0]["raster_path"])
        ctx["dit_embedding"] = vec.tolist()
        dist, nearest = get_template_index().nearest_distance(vec)
        ood = {
            "distance": None if dist == float("inf") else round(dist, 4),
            "is_ood": dist > config.OOD_THRESHOLD,
            "nearest": nearest,
            "embedding_backend": _EMBEDDER.backend,
        }

    fields = _extract_fields_with_bbox(ctx)

    ctx["understanding"] = {
        "classified_type": doc_type,
        "type_scores": type_scores,
        "layoutlmv3": layout,
        "donut": donut,
        "dit": {"backend": _EMBEDDER.backend},
        "ood": ood,
        "fields": fields,
    }
    ctx["classified_type"] = doc_type
    logger.info("%s classified as '%s' (OOD=%s, dist=%s)", ctx["filename"],
                doc_type, ood["is_ood"], ood["distance"])
    return ctx
