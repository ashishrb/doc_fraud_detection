"""HITL (Human-in-the-Loop) review data model and persistence.

Stores reviewer decisions for P0/P1 documents. Each decision is appended to
models/hitl_decisions.jsonl (one JSON object per line, newline-delimited).
This file is the source of truth for future meta-learner retraining.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any

import config
from pipeline.utils import logger

_VALID_DECISIONS = {"confirmed_fraud", "cleared", "escalated"}
# Directories scanned for candidate/document verdict JSONs awaiting review.
_VERDICT_DIRS = [config.BASE_DIR / "dry_run_output", config.UPLOADS_DIR]


def _decisions_path():
    return config.HITL_DECISIONS_PATH


def get_decisions() -> list[dict[str, Any]]:
    """Return all reviewer decisions from hitl_decisions.jsonl."""
    path = _decisions_path()
    if not path.exists():
        return []
    decisions: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            decisions.append(json.loads(line))
        except Exception:  # noqa: BLE001
            logger.warning("HITL: skipping malformed decision line")
    return decisions


def get_decision_count() -> int:
    """Number of decisions logged (used to decide when to retrain)."""
    return len(get_decisions())


def _agent_summary(doc: dict[str, Any]) -> dict[str, float]:
    """Top contributing agents (agent_id -> score) for a flagged document."""
    findings = doc.get("agent_findings", {}) or {}
    scored = [(aid, float(f.get("score", 0.0)))
              for aid, f in findings.items() if "error" not in f]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return {aid: round(score, 3) for aid, score in scored[:5] if score > 0.0}


def _iter_verdict_files():
    for d in _VERDICT_DIRS:
        if not d.exists():
            continue
        yield from d.rglob("*.json")


def get_review_queue() -> list[dict[str, Any]]:
    """Return P0/P1 documents from verdict JSONs that have no decision yet."""
    decided_doc_ids = {d.get("doc_id") for d in get_decisions()}
    queue: list[dict[str, Any]] = []
    seen: set[str] = set()

    for fpath in _iter_verdict_files():
        try:
            data = json.loads(fpath.read_text())
        except Exception:  # noqa: BLE001
            continue
        candidate_id = data.get("candidate_id", "unknown")
        # Candidate-level verdicts hold a "documents" list; document-level
        # verdicts are a single dict. Normalise to a list of documents.
        docs = data.get("documents")
        if not isinstance(docs, list):
            docs = [data] if "band" in data else []

        for doc in docs:
            band = doc.get("band")
            doc_id = doc.get("doc_id")
            if band not in ("P0", "P1") or not doc_id:
                continue
            if doc_id in decided_doc_ids or doc_id in seen:
                continue
            seen.add(doc_id)
            summary = _agent_summary(doc)
            queue.append({
                "candidate_id": candidate_id,
                "doc_id": doc_id,
                "filename": doc.get("filename"),
                "system_band": band,
                "system_fraud_score": doc.get("fraud_score"),
                "top_agents": [
                    {"agent_id": aid,
                     "agent_name": config.AGENT_NAMES.get(aid, aid),
                     "score": score}
                    for aid, score in list(summary.items())[:3]
                ],
                "flagged_regions": [
                    {"page": r.get("page"), "reason": r.get("reason"),
                     "confidence": r.get("confidence"),
                     "agent_name": r.get("agent_name")}
                    for r in doc.get("flagged_regions", [])
                ],
                "agent_findings_summary": summary,
                "agent_findings": doc.get("agent_findings", {}),
                "source": fpath.name,
            })
    return queue


def submit_decision(decision: dict[str, Any]) -> dict[str, Any]:
    """Validate and append a reviewer decision; returns status + decision_id."""
    reviewer_decision = decision.get("reviewer_decision")
    if reviewer_decision not in _VALID_DECISIONS:
        return {"status": "error",
                "message": f"reviewer_decision must be one of "
                           f"{sorted(_VALID_DECISIONS)}"}
    if not decision.get("doc_id"):
        return {"status": "error", "message": "doc_id is required"}

    record = {
        "decision_id": uuid.uuid4().hex,
        "candidate_id": decision.get("candidate_id"),
        "doc_id": decision.get("doc_id"),
        "filename": decision.get("filename"),
        "system_band": decision.get("system_band"),
        "system_fraud_score": decision.get("system_fraud_score"),
        "reviewer_decision": reviewer_decision,
        "reviewer_comment": (decision.get("reviewer_comment") or "")[:500],
        "reviewed_at": datetime.utcnow().isoformat() + "Z",
        "agent_findings_summary": decision.get("agent_findings_summary", {}),
    }

    path = _decisions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    logger.info("HITL: logged decision %s (%s) for doc %s",
                record["decision_id"], reviewer_decision, record["doc_id"])
    return {"status": "ok", "decision_id": record["decision_id"]}
