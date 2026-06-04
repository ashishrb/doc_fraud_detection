"""Operator tool: (re)train Agent 9 (Novel Fraud) on an authentic corpus.

Run this when a corpus of authentic BGV documents becomes available to tighten
the authentic distribution boundary and reduce false positives on legitimate but
unusual documents. It is NOT run automatically by the pipeline.

Usage (from the forgery_detection_poc/ directory):

    python scripts/finetune_agent9.py --source local
    python scripts/finetune_agent9.py --source azure_blob

  - ``--source local``      reads authentic document patches from templates/.
  - ``--source azure_blob`` reads from the Azure Blob container defined by
    AZURE_BLOB_CONNECTION_STRING + AZURE_BLOB_TEMPLATE_CONTAINER.

The mandated fallback model is a PCA linear autoencoder (no torch needed). It is
fit on content patches drawn from every authentic document found at the source;
the reconstruction-error median/MAD over that corpus are stored so the live
agent can compute robust-z novelty scores against the authentic distribution.
Weights are written to models/agent9_weights/agent9_autoencoder.npz, overwriting
the previous checkpoint. Agent 9 loads them automatically on the next run.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# Make the project root importable when run as `python scripts/finetune_agent9.py`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline.agents.agent9_novel_fraud import _patch_vectors  # noqa: E402
from pipeline.template_store import TemplateStore  # noqa: E402
from pipeline.utils import load_page_image  # noqa: E402


def _content_patches(image_path: pathlib.Path) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(np.array(load_page_image(str(image_path))),
                        cv2.COLOR_RGB2GRAY)
    _, vecs = _patch_vectors(gray)
    if len(vecs) == 0:
        return vecs
    return vecs[vecs.std(axis=1) > 0.02]


def main() -> int:
    from sklearn.decomposition import PCA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["local", "azure_blob"],
                        default=config.TEMPLATE_SOURCE,
                        help="where to read authentic documents from")
    parser.add_argument("--components", type=int, default=32,
                        help="max PCA components (default 32)")
    args = parser.parse_args()

    store = TemplateStore(source=args.source)
    all_patches: list[np.ndarray] = []
    doc_count = 0
    for doc_type in config.DOCUMENT_TYPES:
        for tpath in store.get_templates(doc_type):
            patches = _content_patches(pathlib.Path(tpath))
            if len(patches):
                all_patches.append(patches)
                doc_count += 1

    if not all_patches:
        print(f"[finetune_agent9] no authentic documents found for "
              f"source={args.source}; nothing to train.")
        return 1

    X = np.vstack(all_patches).astype(np.float32)
    n_comp = int(min(args.components, X.shape[1], max(2, X.shape[0] - 1)))
    print(f"[finetune_agent9] source={args.source} docs={doc_count} "
          f"patches={X.shape[0]} dim={X.shape[1]} components={n_comp}")

    pca = PCA(n_components=n_comp).fit(X)
    recon = pca.inverse_transform(pca.transform(X))
    err = np.mean((X - recon) ** 2, axis=1)
    err_median = float(np.median(err))
    err_mad = float(np.median(np.abs(err - err_median)))

    config.AGENT9_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.AGENT9_WEIGHTS_DIR / "agent9_autoencoder.npz"
    np.savez(
        out,
        components=pca.components_.astype(np.float32),
        mean=pca.mean_.astype(np.float32),
        err_median=np.float32(err_median),
        err_mad=np.float32(err_mad),
        patch=np.int32(32),
    )

    print(f"[finetune_agent9] train loss (mean recon err)={float(err.mean()):.6f}")
    print(f"[finetune_agent9] err_median={err_median:.6f} err_mad={err_mad:.6f}")
    print(f"[finetune_agent9] saved weights -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
