"""End-to-end orchestration: Steps 1-7 for one candidate's document set."""
from __future__ import annotations

from typing import Any

from pipeline import (cross_doc_llm, document_understanding, ensemble, intake,
                      preprocessing, verdict)
from pipeline.utils import logger, to_jsonable


def analyze_document(filename: str, data: bytes, candidate_id: str,
                     doc_type: str) -> dict[str, Any]:
    """Run Steps 1-4 for a single document, returning its mutated context."""
    ctx = intake.intake_document(filename, data, candidate_id, doc_type)
    if not ctx.get("valid"):
        ctx["agent_findings"] = {}
        return ctx
    preprocessing.preprocess(ctx)          # Step 2
    document_understanding.understand(ctx)  # Step 3
    ensemble.run_agents(ctx)               # Step 4
    return ctx


def analyze_candidate(files: list[dict[str, Any]], candidate_id: str
                      ) -> dict[str, Any]:
    """files: list of {filename, data (bytes), doc_type}."""
    contexts: list[dict[str, Any]] = []
    for f in files:
        logger.info("Analyzing %s (candidate=%s)", f["filename"], candidate_id)
        contexts.append(analyze_document(f["filename"], f["data"],
                                         candidate_id, f.get("doc_type", "other")))

    valid_ctx = [c for c in contexts if c.get("valid")]
    cross = cross_doc_llm.analyze_cross_document(valid_ctx)  # Step 5

    doc_verdicts = []
    for c in contexts:
        if c.get("valid"):
            doc_verdicts.append(
                verdict.build_document_verdict(c, cross["contradictions"]))
        else:
            doc_verdicts.append({
                "doc_id": c.get("doc_id"), "filename": c.get("filename"),
                "band": "P2", "fraud_score": 0.0, "flagged_regions": [],
                "error": c.get("validation_reason"), "pages": [],
                "top_agents": [], "shap_top_agents": [], "escalation_flags": {},
                "agent_findings": {},
            })

    candidate_verdict = verdict.build_candidate_verdict(
        candidate_id, doc_verdicts, cross)
    return to_jsonable(candidate_verdict)
