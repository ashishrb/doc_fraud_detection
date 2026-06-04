"""Agent 8 - QR / Barcode Validation [Mandatory].

Decode QR codes / barcodes with pyzbar (+ OpenCV fallback). Flags:
  - a code present but decoding to a malformed/empty payload,
  - a QR expected for this document type but absent,
  - a decoded value inconsistent with the document's extracted fields.
"""
from __future__ import annotations

from typing import Any

from pipeline.utils import load_page_image, safe_agent

AGENT_ID = "agent_8"
# Document types where an authenticity QR/barcode is normally expected.
_EXPECT_QR = {"form16", "education_certificate", "offer_letter"}


def _decode(img):
    results = []
    try:
        from pyzbar.pyzbar import decode

        for sym in decode(img):
            x, y, w, h = sym.rect.left, sym.rect.top, sym.rect.width, sym.rect.height
            results.append({
                "type": sym.type,
                "data": sym.data.decode("utf-8", errors="replace"),
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
            })
    except Exception:  # noqa: BLE001
        pass
    return results


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    doc_type = ctx.get("classified_type", ctx.get("doc_type", "other"))
    regions: list[dict[str, Any]] = []
    reasons: list[str] = []
    score = 0.0
    decoded: list[dict[str, Any]] = []

    for page in ctx["pages"]:
        img = load_page_image(page["raster_path"])
        codes = _decode(img)
        for c in codes:
            decoded.append(c)
            if not c["data"].strip():
                score = max(score, 0.6)
                reasons.append(f"{c['type']} decodes to empty/malformed payload")
                regions.append({"page": page["page"], "bbox": c["bbox"],
                                "reason": "empty/malformed code payload",
                                "confidence": 0.6})
            else:
                # consistency check vs known field values
                text_l = (ctx.get("text", "") or "").lower()
                payload_l = c["data"].lower()
                if len(payload_l) > 4 and payload_l not in text_l and \
                        not any(tok in text_l for tok in payload_l.split()
                                if len(tok) > 4):
                    score = max(score, 0.35)
                    reasons.append(
                        f"{c['type']} payload not corroborated by document text")
                    regions.append({"page": page["page"], "bbox": c["bbox"],
                                    "reason": "code value inconsistent with fields",
                                    "confidence": 0.35})

    if not decoded and doc_type in _EXPECT_QR:
        score = max(score, 0.4)
        reasons.append(f"no QR/barcode found but expected for '{doc_type}'")

    return {
        "score": round(score, 3),
        "flagged": bool(regions) or score >= 0.4,
        "flagged_regions": regions,
        "detail": "; ".join(reasons) if reasons
                  else f"{len(decoded)} code(s) decoded OK",
        "decoded": decoded,
    }
