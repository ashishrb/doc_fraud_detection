"""Step 1 - Document Intake.

Validate uploads, compute SHA-256, store in uploads/, build the per-document
context object that the rest of the pipeline mutates.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from pipeline.utils import logger, render_to_rasters

_HASH_LOG = config.UPLOADS_DIR / "processed_hashes.json"


def _load_hash_log() -> dict[str, str]:
    if _HASH_LOG.exists():
        try:
            return json.loads(_HASH_LOG.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_hash_log(data: dict[str, str]) -> None:
    _HASH_LOG.write_text(json.dumps(data, indent=2))


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate(filename: str, data: bytes) -> tuple[bool, str]:
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return False, f"unsupported extension '{ext}'"
    if len(data) == 0:
        return False, "empty file"
    if len(data) > config.MAX_FILE_BYTES:
        return False, f"file too large ({len(data)} bytes)"
    # Minimal magic-byte sanity check.
    if ext == ".pdf" and not data[:5].startswith(b"%PDF"):
        return False, "corrupt PDF (missing %PDF header)"
    return True, "ok"


def intake_document(filename: str, data: bytes, candidate_id: str,
                    doc_type: str) -> dict[str, Any]:
    """Validate + persist a single uploaded document, returning its context."""
    ok, reason = validate(filename, data)
    digest = sha256_of(data)
    doc_id = uuid.uuid4().hex[:12]
    work_dir = config.UPLOADS_DIR / doc_id
    work_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(filename).suffix.lower()
    saved_path = work_dir / f"original{ext}"

    ctx: dict[str, Any] = {
        "doc_id": doc_id,
        "filename": filename,
        "doc_type": doc_type if doc_type in config.DOCUMENT_TYPES else "other",
        "candidate_id": candidate_id,
        "sha256": digest,
        "ext": ext,
        "is_pdf": ext == ".pdf",
        "valid": ok,
        "validation_reason": reason,
        "path": str(saved_path),
        "work_dir": str(work_dir),
        "pages": [],
        "duplicate_of_known_hash": False,
        # Consent is validated by main.py before intake runs (DPDP/GDPR).
        "consent_given": True,
        "consent_logged_at": datetime.utcnow().isoformat() + "Z",
    }

    if not ok:
        logger.warning("Intake rejected %s: %s", filename, reason)
        return ctx

    saved_path.write_bytes(data)

    hash_log = _load_hash_log()
    if digest in hash_log:
        ctx["duplicate_of_known_hash"] = True
        ctx["previous_filename"] = hash_log[digest]
        logger.info("SHA-256 %s already processed (%s)", digest[:12],
                    hash_log[digest])
    hash_log[digest] = filename
    _save_hash_log(hash_log)

    try:
        ctx["pages"] = render_to_rasters(saved_path, work_dir)
    except Exception as exc:  # noqa: BLE001
        ctx["valid"] = False
        ctx["validation_reason"] = f"render failed: {exc}"
        logger.warning("Render failed for %s: %s", filename, exc)

    return ctx
