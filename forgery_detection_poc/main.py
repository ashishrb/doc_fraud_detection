"""FastAPI entry point for the document forgery detection POC.

Endpoints:
  GET  /                      -> single-page HTML UI
  POST /analyze               -> multipart (files + candidate_id + doc_types)
  GET  /raster/{doc_id}/{n}   -> page raster PNG (for the canvas overlay)
  GET  /health                -> liveness
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
from pipeline.orchestrator import analyze_candidate
from pipeline.seed import ensure_seed_assets
from pipeline.utils import logger

app = FastAPI(title="Document Forgery Detection POC", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    ensure_seed_assets()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/raster/{doc_id}/{page}")
def raster(doc_id: str, page: int):
    safe_id = Path(doc_id).name
    path = config.UPLOADS_DIR / safe_id / f"page_{page}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="raster not found")
    return FileResponse(path, media_type="image/png")


@app.post("/analyze")
async def analyze(
    candidate_id: str = Form(...),
    files: list[UploadFile] = File(...),
    doc_types: list[str] = Form(default=[]),
):
    if not files:
        raise HTTPException(status_code=400, detail="no files uploaded")
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="max 5 documents per candidate")

    payload = []
    for i, uf in enumerate(files):
        data = await uf.read()
        dtype = doc_types[i] if i < len(doc_types) else "other"
        payload.append({"filename": uf.filename, "data": data, "doc_type": dtype})

    logger.info("ANALYZE candidate=%s files=%d", candidate_id, len(payload))
    try:
        verdict = analyze_candidate(payload, candidate_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("analyze failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(verdict)


# Static UI mounted last so /analyze etc. take precedence.
app.mount("/", StaticFiles(directory=str(config.STATIC_DIR), html=True),
          name="static")
