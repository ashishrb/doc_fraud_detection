"""Agent 5 - Duplicate / Similarity Detection [Mandatory].

  - pHash (imagehash) of the page raster,
  - MinHash (datasketch) of the extracted text,
  - a persistent FAISS binary index (Hamming) of previously processed pHashes.

Flags exact SHA-256 matches and near-duplicates (pHash Hamming distance below
AGENT5_PHASH_THRESHOLD, or MinHash Jaccard above 0.8).
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np

import config
from pipeline.utils import load_page_image, safe_agent

AGENT_ID = "agent_5"
_REGISTRY = config.FAISS_DIR / "dedup_registry.json"
_NUM_PERM = 128


def _load_registry() -> list[dict[str, Any]]:
    if _REGISTRY.exists():
        try:
            return json.loads(_REGISTRY.read_text())
        except Exception:  # noqa: BLE001
            return []
    return []


def _save_registry(reg: list[dict[str, Any]]) -> None:
    _REGISTRY.write_text(json.dumps(reg, indent=2))


def _phash_hex(ctx: dict[str, Any]) -> str:
    import imagehash

    img = load_page_image(ctx["pages"][0]["raster_path"])
    return str(imagehash.phash(img))


def _minhash(text: str):
    from datasketch import MinHash

    m = MinHash(num_perm=_NUM_PERM)
    tokens = set(re.findall(r"\w+", (text or "").lower()))
    for tok in tokens:
        m.update(tok.encode("utf-8"))
    return m


def _hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _build_faiss_binary(reg: list[dict[str, Any]]):
    """Demonstrative FAISS binary (Hamming) index over 64-bit pHashes."""
    import faiss

    index = faiss.IndexBinaryFlat(64)
    if reg:
        vecs = np.array([_hex_to_bytes(e["phash_hex"]) for e in reg],
                        dtype=np.uint8)
        index.add(vecs)
    return index


def _hex_to_bytes(h: str) -> list[int]:
    val = int(h, 16)
    return list(val.to_bytes(8, "big"))


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    reg = _load_registry()
    cur_phash = _phash_hex(ctx)
    cur_minhash = _minhash(ctx.get("text", ""))
    cur_sha = ctx.get("sha256", "")

    reasons: list[str] = []
    score = 0.0
    matches: list[dict[str, Any]] = []

    # FAISS binary index search (demonstration) + exact registry comparison.
    if reg:
        index = _build_faiss_binary(reg)
        q = np.array([_hex_to_bytes(cur_phash)], dtype=np.uint8)
        D, indices = index.search(q, min(3, len(reg)))
        for dist, idx in zip(D[0], indices[0]):
            if idx < 0:
                continue
            entry = reg[int(idx)]
            if entry.get("doc_id") == ctx.get("doc_id"):
                continue
            jacc = cur_minhash.jaccard(_restore_minhash(entry.get("minhash", [])))
            if entry.get("sha256") == cur_sha:
                score = max(score, 1.0)
                reasons.append(f"exact SHA-256 duplicate of '{entry['filename']}'")
                matches.append({"filename": entry["filename"], "type": "exact_sha",
                                "phash_distance": int(dist)})
            elif int(dist) < config.AGENT5_PHASH_THRESHOLD:
                score = max(score, 0.85)
                reasons.append(
                    f"near-duplicate image of '{entry['filename']}' "
                    f"(pHash dist={int(dist)})")
                matches.append({"filename": entry["filename"],
                                "type": "near_phash", "phash_distance": int(dist)})
            elif jacc > 0.8:
                score = max(score, 0.8)
                reasons.append(f"near-duplicate text of '{entry['filename']}' "
                               f"(Jaccard={jacc:.2f})")
                matches.append({"filename": entry["filename"],
                                "type": "near_text", "jaccard": round(jacc, 3)})

    # Register current doc (dedup by doc_id).
    if not any(e.get("doc_id") == ctx.get("doc_id") for e in reg):
        reg.append({
            "doc_id": ctx.get("doc_id"),
            "filename": ctx.get("filename"),
            "sha256": cur_sha,
            "phash_hex": cur_phash,
            "minhash": cur_minhash.hashvalues.tolist(),
        })
        _save_registry(reg)

    return {
        "score": round(score, 3),
        "flagged": score > 0.0,
        "flagged_regions": [],  # duplicate findings are document-level
        "detail": "; ".join(reasons) if reasons else "no duplicates found",
        "matches": matches,
    }


def _restore_minhash(values: list[int]):
    from datasketch import MinHash

    if not values:
        return MinHash(num_perm=_NUM_PERM)
    return MinHash(num_perm=_NUM_PERM, hashvalues=np.array(values, dtype=np.uint64))
