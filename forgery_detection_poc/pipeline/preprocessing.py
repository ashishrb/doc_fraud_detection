"""Step 2 - Pre-processing & Triple OCR.

- Extract PDF metadata (PyMuPDF + exiftool) BEFORE OCR to preserve byte layout.
- Run up to three OCR engines in parallel: Azure Document Intelligence
  (optional), PaddleOCR (optional), Tesseract (required baseline).
- Compute a per-field consensus & disagreement vector.

Azure / PaddleOCR degrade gracefully when unavailable; the run continues with
whatever engines are present and the gap is recorded.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import config
from pipeline.utils import load_page_image, logger

# --- Field extraction patterns (shared across engines for disagreement) ---
_FIELD_PATTERNS: dict[str, re.Pattern] = {
    "designation": re.compile(
        r"(?:designation|title|position|role)\s*[:\-]\s*([A-Za-z0-9 .,&/()'-]{2,60})",
        re.I),
    "employer": re.compile(
        r"(?:employer|company|organization|organisation|firm)\s*[:\-]\s*"
        r"([A-Za-z0-9 .,&/()'-]{2,60})", re.I),
    "name": re.compile(
        r"(?:employee name|candidate name|name)\s*[:\-]\s*"
        r"([A-Za-z .'-]{2,50})", re.I),
    "date": re.compile(
        r"\b(\d{1,2}[/-][A-Za-z0-9]{2,9}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b"),
    "amount": re.compile(
        r"(?:salary|amount|net pay|gross|ctc|total)\s*[:\-]?\s*"
        r"(?:rs\.?|inr|usd|\$|₹)?\s*([0-9][0-9,]{2,}(?:\.\d{1,2})?)", re.I),
}


def extract_pdf_metadata(ctx: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata via PyMuPDF and (if present) exiftool."""
    meta: dict[str, Any] = {"pymupdf": {}, "exiftool": {}}
    if ctx.get("is_pdf"):
        try:
            import fitz

            doc = fitz.open(ctx["path"])
            meta["pymupdf"] = dict(doc.metadata or {})
            meta["pymupdf"]["page_count"] = doc.page_count
            # XML metadata stream (XMP), if any.
            try:
                xmp = doc.xref_xml_metadata()
                meta["pymupdf"]["xmp_present"] = bool(xmp)
            except Exception:  # noqa: BLE001
                meta["pymupdf"]["xmp_present"] = False
            doc.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("PyMuPDF metadata failed: %s", exc)

    if shutil.which("exiftool"):
        try:
            out = subprocess.run(
                ["exiftool", "-s", "-G", ctx["path"]],
                capture_output=True, text=True, timeout=30,
            )
            parsed: dict[str, str] = {}
            for line in out.stdout.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    parsed[k.strip()] = v.strip()
            meta["exiftool"] = parsed
        except Exception as exc:  # noqa: BLE001
            logger.warning("exiftool failed: %s", exc)
    return meta


# --------------------------- OCR engines --------------------------- #
def _ocr_tesseract(ctx: dict[str, Any]) -> dict[str, Any]:
    import pytesseract
    from pytesseract import Output

    words: list[dict[str, Any]] = []
    full_text_parts: list[str] = []
    for page in ctx["pages"]:
        img = load_page_image(page["raster_path"])
        data = pytesseract.image_to_data(img, output_type=Output.DICT)
        n = len(data["text"])
        for i in range(n):
            txt = data["text"][i].strip()
            conf = float(data["conf"][i]) if data["conf"][i] not in ("-1", "") else -1
            if not txt:
                continue
            x, y, w, h = (data["left"][i], data["top"][i],
                          data["width"][i], data["height"][i])
            words.append({
                "page": page["page"], "text": txt,
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "conf": conf,
            })
        full_text_parts.append(pytesseract.image_to_string(img))
    return {"engine": "tesseract", "available": True,
            "text": "\n".join(full_text_parts), "words": words}


def _ocr_paddle(ctx: dict[str, Any]) -> dict[str, Any]:
    if not config.ENABLE_PADDLEOCR:
        return {"engine": "paddleocr", "available": False, "text": "",
                "words": [], "error": "PaddleOCR disabled (set ENABLE_PADDLEOCR=1)"}
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {"engine": "paddleocr", "available": False, "text": "",
                "words": [], "error": f"paddleocr not installed: {exc}"}
    try:
        import numpy as np

        ocr = _get_paddle(PaddleOCR)
        words: list[dict[str, Any]] = []
        parts: list[str] = []
        for page in ctx["pages"]:
            img = np.array(load_page_image(page["raster_path"]))
            for txt, conf, box in _paddle_predict(ocr, img):
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                words.append({
                    "page": page["page"], "text": txt,
                    "bbox": [float(min(xs)), float(min(ys)),
                             float(max(xs)), float(max(ys))],
                    "conf": float(conf),
                })
                parts.append(txt)
        return {"engine": "paddleocr", "available": True,
                "text": "\n".join(parts), "words": words}
    except Exception as exc:  # noqa: BLE001
        return {"engine": "paddleocr", "available": False, "text": "",
                "words": [], "error": str(exc)}


_PADDLE_SINGLETON = None


def _get_paddle(PaddleOCR):
    """Instantiate PaddleOCR once, tolerant of 2.x vs 3.x constructor args."""
    global _PADDLE_SINGLETON
    if _PADDLE_SINGLETON is not None:
        return _PADDLE_SINGLETON
    try:  # PaddleOCR 3.x
        _PADDLE_SINGLETON = PaddleOCR(
            lang="en", use_doc_orientation_classify=False,
            use_doc_unwarping=False, use_textline_orientation=False)
    except Exception:  # noqa: BLE001  # fall back to 2.x signature
        _PADDLE_SINGLETON = PaddleOCR(use_angle_cls=True, lang="en")
    return _PADDLE_SINGLETON


def _paddle_predict(ocr, img):
    """Yield (text, conf, box[4 pts]) tolerant of PaddleOCR 2.x/3.x outputs."""
    # 3.x: predict() -> list of result dicts with rec_texts/rec_scores/rec_polys
    if hasattr(ocr, "predict"):
        try:
            for res in ocr.predict(img) or []:
                data = res.get("res", res) if isinstance(res, dict) else res
                texts = data.get("rec_texts", [])
                scores = data.get("rec_scores", [])
                polys = data.get("rec_polys", data.get("dt_polys", []))
                for i, txt in enumerate(texts):
                    conf = scores[i] if i < len(scores) else 0.0
                    box = polys[i] if i < len(polys) else [[0, 0]] * 4
                    yield txt, conf, [list(p) for p in box]
            return
        except Exception:  # noqa: BLE001  # fall through to legacy api
            pass
    # 2.x: ocr() -> [[ [box, (txt, conf)], ... ]]
    res = ocr.ocr(img)
    for line in (res[0] or []) if res else []:
        box, (txt, conf) = line[0], line[1]
        yield txt, conf, [list(p) for p in box]


def _ocr_azure(ctx: dict[str, Any]) -> dict[str, Any]:
    if not (config.AZURE_DOC_INTELLIGENCE_ENDPOINT
            and config.AZURE_DOC_INTELLIGENCE_KEY):
        return {"engine": "azure", "available": False, "text": "", "words": [],
                "error": "Azure DI key/endpoint not configured"}
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        client = DocumentIntelligenceClient(
            endpoint=config.AZURE_DOC_INTELLIGENCE_ENDPOINT,
            credential=AzureKeyCredential(config.AZURE_DOC_INTELLIGENCE_KEY),
        )
        with open(ctx["path"], "rb") as fh:
            poller = client.begin_analyze_document("prebuilt-read", body=fh)
        result = poller.result()
        words: list[dict[str, Any]] = []
        scale = config.RASTER_DPI  # Azure polygons are in inches for PDFs
        for pidx, page in enumerate(result.pages or [], start=1):
            for word in getattr(page, "words", []) or []:
                poly = word.polygon or []
                xs = poly[0::2]
                ys = poly[1::2]
                if not xs:
                    continue
                words.append({
                    "page": pidx, "text": word.content,
                    "bbox": [min(xs) * scale, min(ys) * scale,
                             max(xs) * scale, max(ys) * scale],
                    "conf": float(getattr(word, "confidence", 0.0) or 0.0),
                })
        return {"engine": "azure", "available": True,
                "text": result.content or "", "words": words}
    except Exception as exc:  # noqa: BLE001
        return {"engine": "azure", "available": False, "text": "", "words": [],
                "error": str(exc)}


def _extract_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for fname, pat in _FIELD_PATTERNS.items():
        m = pat.search(text or "")
        if m:
            fields[fname] = re.sub(r"\s+", " ", m.group(1)).strip().lower()
    return fields


def _normalise(val: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (val or "").lower())


def compute_disagreement_vector(ocr_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare extracted field values across engines; build disagreement vec."""
    available = [r for r in ocr_results if r.get("available")]
    per_engine_fields = {r["engine"]: _extract_fields(r["text"]) for r in available}
    all_fields = set().union(*[set(f) for f in per_engine_fields.values()]) \
        if per_engine_fields else set()

    vector: list[dict[str, Any]] = []
    for field in sorted(all_fields):
        values = {eng: f.get(field) for eng, f in per_engine_fields.items()
                  if f.get(field)}
        norm_vals = {_normalise(v) for v in values.values()}
        n_engines = len(values)
        crit = config.FIELD_CRITICALITY.get(field, config.DEFAULT_FIELD_CRITICALITY)
        if n_engines <= 1:
            disagreement = 0.0
        else:
            # fraction of engines NOT matching the majority value.
            from collections import Counter

            counts = Counter(_normalise(v) for v in values.values())
            majority = counts.most_common(1)[0][1]
            disagreement = 1.0 - (majority / n_engines)
        vector.append({
            "field": field,
            "criticality": crit,
            "values": values,
            "n_engines": n_engines,
            "disagreement": round(disagreement, 3),
            "distinct_values": len(norm_vals),
        })
    return vector


def preprocess(ctx: dict[str, Any]) -> dict[str, Any]:
    """Run Step 2 for a single document context (mutates and returns ctx)."""
    ctx["pdf_metadata"] = extract_pdf_metadata(ctx)

    engines = [_ocr_azure, _ocr_paddle, _ocr_tesseract]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(fn, ctx): fn.__name__ for fn in engines}
        for fut in futures:
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR engine %s crashed: %s", futures[fut], exc)

    ctx["ocr"] = {r["engine"]: r for r in results}
    ctx["disagreement_vector"] = compute_disagreement_vector(results)

    primary = (ctx["ocr"].get("azure") if ctx["ocr"].get("azure", {}).get("available")
               else ctx["ocr"].get("tesseract", {}))
    ctx["text"] = primary.get("text", "")
    ctx["words"] = primary.get("text") and primary.get("words", []) or \
        ctx["ocr"].get("tesseract", {}).get("words", [])
    ctx["ocr_engines_available"] = [e for e, r in ctx["ocr"].items()
                                    if r.get("available")]
    logger.info("OCR engines available for %s: %s", ctx["filename"],
                ctx["ocr_engines_available"])
    return ctx
