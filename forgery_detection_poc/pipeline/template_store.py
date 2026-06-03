"""TemplateStore abstraction (Agent 4 + Step 3 seeding).

Provides authentic reference templates for a given document type behind a single
interface, so the source can be switched from the local ``templates/`` directory
to Azure Blob Storage with a config-only change (``TEMPLATE_SOURCE``):

  - ``local``      (POC default): reads ``templates/<doc_type>/*`` from disk.
  - ``azure_blob`` (future):      lists/downloads ``<container>/<doc_type>/*``
                                  from Blob Storage, caches them under
                                  ``templates/cache/<doc_type>/`` and returns
                                  them in the same form as the local backend.

Callers (Agent 4, the indexing scripts) only ever call
``get_template_store().get_templates(doc_type)`` and receive a list of local
image ``Path`` objects — they never touch disk paths or Blob directly.

The azure_blob backend degrades gracefully: any connection/SDK error is logged
and the store falls back to the local backend so the pipeline never crashes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import config
from pipeline.utils import logger

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")
_RASTERISABLE = (".pdf", ".docx", ".tif", ".tiff")


def _local_templates(doc_type: str) -> list[Path]:
    """Authentic template images for ``doc_type`` from the local directory."""
    paths: list[Path] = []
    tdir = config.TEMPLATES_DIR / doc_type
    if tdir.is_dir():
        for suf in _IMAGE_SUFFIXES:
            paths += sorted(tdir.glob(f"*{suf}"))
    if not paths:  # fall back to any available template
        for d in sorted(config.TEMPLATES_DIR.glob("*")):
            if d.is_dir() and d.name != "cache":
                for suf in _IMAGE_SUFFIXES:
                    paths += sorted(d.glob(f"*{suf}"))
    return paths


def _as_image(path: Path, doc_type: str) -> Path | None:
    """Return an image Path: pass images through, rasterise PDFs/DOCX first."""
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        return path
    if path.suffix.lower() in _RASTERISABLE:
        from pipeline.utils import render_to_rasters

        work = config.TEMPLATE_CACHE_DIR / doc_type / f"{path.stem}_raster"
        try:
            pages = render_to_rasters(path, work)
            if pages:
                return Path(pages[0]["raster_path"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("template rasterise failed for %s: %s", path, exc)
    return None


class TemplateStore:
    """Switchable template provider (local / azure_blob)."""

    def __init__(self, source: str | None = None) -> None:
        self.source = (source or config.TEMPLATE_SOURCE or "local").lower()

    # ---------------------------- public API ---------------------------- #
    def get_templates(self, doc_type: str) -> list[Path]:
        if self.source == "azure_blob":
            blob = self._azure_blob_templates(doc_type)
            if blob:
                return blob
            logger.warning("azure_blob returned no templates for '%s'; "
                           "falling back to local", doc_type)
        return _local_templates(doc_type)

    # ------------------------- azure_blob backend ----------------------- #
    def _azure_blob_templates(self, doc_type: str) -> list[Path]:
        """List + download ``<container>/<doc_type>/*`` and cache locally.

        Fully working stub: connects with AZURE_BLOB_CONNECTION_STRING, lists
        blobs under the ``<doc_type>/`` prefix, downloads each (skipping ones
        already cached), rasterises non-image files, and returns local image
        paths. Activated purely by ``TEMPLATE_SOURCE=azure_blob`` + a
        connection string — no code changes required.
        """
        if not config.AZURE_BLOB_CONNECTION_STRING:
            logger.warning("TEMPLATE_SOURCE=azure_blob but "
                           "AZURE_BLOB_CONNECTION_STRING is empty")
            return []
        try:
            from azure.storage.blob import ContainerClient
        except Exception as exc:  # noqa: BLE001
            logger.warning("azure-storage-blob not installed: %s", exc)
            return []

        cache_dir = config.TEMPLATE_CACHE_DIR / doc_type
        cache_dir.mkdir(parents=True, exist_ok=True)
        out: list[Path] = []
        try:
            container = ContainerClient.from_connection_string(
                config.AZURE_BLOB_CONNECTION_STRING,
                config.AZURE_BLOB_TEMPLATE_CONTAINER,
            )
            prefix = f"{doc_type}/"
            for blob in container.list_blobs(name_starts_with=prefix):
                name = Path(blob.name).name
                if not name:
                    continue
                local = cache_dir / name
                if not local.exists():
                    data = container.download_blob(blob.name).readall()
                    local.write_bytes(data)
                    logger.info("cached blob %s -> %s", blob.name, local)
                img = _as_image(local, doc_type)
                if img is not None:
                    out.append(img)
        except Exception as exc:  # noqa: BLE001
            logger.warning("azure_blob template fetch failed: %s", exc)
            return []
        return out


_STORE: TemplateStore | None = None


def get_template_store() -> TemplateStore:
    global _STORE
    if _STORE is None:
        _STORE = TemplateStore()
    return _STORE
