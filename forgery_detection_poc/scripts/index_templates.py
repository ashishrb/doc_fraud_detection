"""Operator tool: ingest authentic templates into the Step-3 OOD FAISS index.

This is the entry point for improving Agent 4 template matching and Step-3
out-of-distribution detection once a corpus of authentic documents is available.
It is NOT run automatically by the pipeline.

Usage (from the forgery_detection_poc/ directory):

    python scripts/index_templates.py --source local
    python scripts/index_templates.py --source azure_blob

  - ``--source local``      reads from the templates/ directory.
  - ``--source azure_blob`` reads from the Azure Blob container defined by
    AZURE_BLOB_CONNECTION_STRING + AZURE_BLOB_TEMPLATE_CONTAINER, organised as
    <container>/<doc_type>/<file>.

Embeddings (DiT or deterministic fallback) are generated for every document
found at the source and upserted into faiss_index/template_embeddings.index
with metadata. The operation is incremental — files already indexed (matched by
"<doc_type>/<filename>") are skipped — so it can be re-run cheaply.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Make the project root importable when run as `python scripts/index_templates.py`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline.document_understanding import EMBED_DIM, get_embedder  # noqa: E402
from pipeline.template_embedding_index import TemplateEmbeddingIndex  # noqa: E402
from pipeline.template_store import TemplateStore  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["local", "azure_blob"],
                        default=config.TEMPLATE_SOURCE,
                        help="where to read authentic templates from")
    args = parser.parse_args()

    store = TemplateStore(source=args.source)
    index = TemplateEmbeddingIndex(embed_fn=get_embedder().embed, dim=EMBED_DIM)

    print(f"[index_templates] source={args.source} "
          f"existing_vectors={index.count}")

    added = 0
    seen = 0
    for doc_type in config.DOCUMENT_TYPES:
        templates = store.get_templates(doc_type)
        for tpath in templates:
            seen += 1
            if index.upsert(tpath, doc_type):
                added += 1
                print(f"  + indexed [{doc_type}] {tpath.name}")
    if added:
        index.save()

    print(f"[index_templates] done: scanned={seen} added={added} "
          f"total_vectors={index.count}")
    print(f"[index_templates] index file: {config.TEMPLATE_INDEX_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
