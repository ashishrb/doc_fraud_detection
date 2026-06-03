"""TemplateEmbeddingIndex - persisted FAISS index for Step 3 OOD detection.

Wraps a local FAISS ``IndexFlatL2`` of authentic-template embeddings with
load / upsert / query / persist semantics. The embedding function is injected
(Step 3 supplies the DiT-or-fallback embedder) so this module has no dependency
on the heavy model stack and can be reused by the operator script
``scripts/index_templates.py``.

Persistence:
  - vectors  -> ``faiss_index/template_embeddings.index`` (config.TEMPLATE_INDEX_PATH)
  - metadata -> ``faiss_index/template_embeddings.meta.json``

Each entry's metadata records a ``key`` (``"<doc_type>/<filename>"``) so upserts
are incremental — a file already present (matched by key) is skipped, which lets
the index be re-run cheaply as new authentic documents arrive.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

import config
from pipeline.utils import logger

EmbedFn = Callable[[str], np.ndarray]


class TemplateEmbeddingIndex:
    def __init__(self, embed_fn: EmbedFn, dim: int = 768,
                 index_path: Path | None = None,
                 meta_path: Path | None = None) -> None:
        self.embed_fn = embed_fn
        self.dim = dim
        self.index_path = index_path or config.TEMPLATE_INDEX_PATH
        self.meta_path = meta_path or config.TEMPLATE_INDEX_META_PATH
        self.index = None
        self.meta: list[dict[str, Any]] = []
        self._load_or_init()

    # ----------------------------- lifecycle ---------------------------- #
    def _load_or_init(self) -> None:
        import faiss

        if self.index_path.exists() and self.meta_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))
                self.meta = json.loads(self.meta_path.read_text())
                if self.index.d == self.dim and self.index.ntotal == len(self.meta):
                    logger.info("Loaded template index (%d vectors) from %s",
                                self.index.ntotal, self.index_path)
                    return
                logger.warning("Template index mismatch (d=%s n=%s vs meta=%s); "
                               "rebuilding", self.index.d, self.index.ntotal,
                               len(self.meta))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to load template index, rebuilding: %s", exc)
        self.index = faiss.IndexFlatL2(self.dim)
        self.meta = []

    def save(self) -> None:
        import faiss

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.meta, indent=2))
        logger.info("Persisted template index (%d vectors) to %s",
                    self.index.ntotal, self.index_path)

    # ------------------------------- ops -------------------------------- #
    @property
    def count(self) -> int:
        return 0 if self.index is None else int(self.index.ntotal)

    def _keys(self) -> set[str]:
        return {m.get("key") for m in self.meta}

    def upsert(self, file_path: str | Path, doc_type: str) -> bool:
        """Embed + add one document if not already indexed. Returns True if added."""
        file_path = Path(file_path)
        key = f"{doc_type}/{file_path.name}"
        if key in self._keys():
            return False
        try:
            vec = self.embed_fn(str(file_path)).astype(np.float32)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to embed template %s: %s", file_path, exc)
            return False
        self.index.add(vec.reshape(1, -1))
        self.meta.append({"key": key, "path": str(file_path),
                          "doc_type": doc_type})
        return True

    def seed_from_templates(self, store=None) -> int:
        """Seed from the TemplateStore (all known doc types). Returns # added."""
        if store is None:
            from pipeline.template_store import get_template_store
            store = get_template_store()
        added = 0
        for doc_type in config.DOCUMENT_TYPES:
            for tpath in store.get_templates(doc_type):
                if self.upsert(tpath, doc_type):
                    added += 1
        if added:
            self.save()
        logger.info("Template index seeded with %d new authentic embeddings "
                    "(total=%d)", added, self.count)
        return added

    def nearest_distance(self, vec: np.ndarray) -> tuple[float, dict | None]:
        if self.index is None or self.index.ntotal == 0:
            return float("inf"), None
        D, indices = self.index.search(vec.reshape(1, -1).astype(np.float32), 1)
        return float(D[0][0]), self.meta[int(indices[0][0])]
