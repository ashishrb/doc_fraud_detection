"""Step 7 - Verdict & Routing.

Builds the per-document and candidate-level verdict, applying the 3 escalation
rules on top of the band:
  Rule 1 (Confidence): high agent-score std dev -> mandatory human review.
  Rule 2 (Disagreement): 2+ agents strongly disagree w/ ensemble -> override.
  Rule 3 (Novelty): Agent 9 high but specialists low -> promote band to P1.
"""
from __future__ import annotations

from typing import Any

import numpy as np

import config
from pipeline.agents import agent14_adjudicator
from pipeline.meta_learner import meta_score
from pipeline.utils import clamp_bbox

_BAND_RANK = {"P2": 0, "P1": 1, "P0": 2}
_RANK_BAND = {v: k for k, v in _BAND_RANK.items()}


def _collect_regions(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    page_dims = {p["page"]: (p["width"], p["height"]) for p in ctx["pages"]}
    for aid, f in ctx.get("agent_findings", {}).items():
        if not f.get("flagged"):
            continue  # only surface regions from agents that actually flagged
        for r in f.get("flagged_regions", []):
            page = r.get("page", 1)
            w, h = page_dims.get(page, (10_000, 10_000))
            conf = float(r.get("confidence", f.get("score", 0.5)))
            regions.append({
                "agent_id": aid,
                "agent_name": config.AGENT_NAMES.get(aid, aid),
                "page": page,
                "bbox": clamp_bbox([float(x) for x in r["bbox"]], w, h),
                "reason": r.get("reason", ""),
                "confidence": round(conf, 3),
            })
    return regions


def _escalation_rules(ctx: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    findings = ctx.get("agent_findings", {})
    scores = np.array([f.get("score", 0.0) for f in findings.values()])
    std = float(scores.std()) if len(scores) else 0.0

    flags: dict[str, Any] = {}

    # Rule 1 - confidence / uncertainty
    flags["rule1_uncertainty_std"] = round(std, 3)
    flags["human_review_required"] = std > config.RULE1_UNCERTAINTY_THRESHOLD

    # Rule 2 - disagreement with ensemble
    ens = meta["fraud_score"]
    strong_disagree = [aid for aid, f in findings.items()
                       if abs(f.get("score", 0.0) - ens)
                       > config.RULE2_DISAGREEMENT_DELTA]
    flags["disagreeing_agents"] = strong_disagree
    flags["disagreement_override"] = len(strong_disagree) >= config.RULE2_MIN_AGENTS

    # Rule 3 - novelty promotion
    a9 = findings.get("agent_9", {}).get("score", 0.0)
    specialist = [f.get("score", 0.0) for aid, f in findings.items()
                  if aid != "agent_9"]
    spec_max = max(specialist) if specialist else 0.0
    band = meta["band"]
    if a9 > config.AGENT9_NOVELTY_HIGH and spec_max < 0.3 and \
            _BAND_RANK[band] < _BAND_RANK["P1"]:
        band = "P1"
        flags["novelty_promotion"] = True
    else:
        flags["novelty_promotion"] = False

    return {"band": band, "flags": flags}


def build_document_verdict(ctx: dict[str, Any],
                           contradictions: list[dict[str, Any]]) -> dict[str, Any]:
    involved = [c for c in contradictions
                if ctx["filename"] in c.get("documents_involved", [])]
    meta = meta_score(ctx, len(involved))
    esc = _escalation_rules(ctx, meta)
    regions = _collect_regions(ctx)

    # Agent 14 (Cross-Agent Adjudicator) - runs only when Rule 2 fires, AFTER
    # the ensemble. It can override the ensemble fraud score with a reasoned
    # consensus verdict; on any error the ensemble verdict stands unchanged.
    fraud_score = meta["fraud_score"]
    adjudication: dict[str, Any] = {}
    if esc["flags"].get("disagreement_override"):
        result = agent14_adjudicator.run(ctx.get("agent_findings", {}),
                                         meta["fraud_score"])
        if "error" in result:
            adjudication = {"adjudication_applied": False,
                            "adjudication_error": result["error"]}
        else:
            fraud_score = result["score"]
            adjudication = {"adjudication_applied": True}

    return {
        "doc_id": ctx["doc_id"],
        "filename": ctx["filename"],
        "document_type": ctx.get("classified_type", ctx.get("doc_type")),
        "sha256": ctx.get("sha256"),
        "pages": [{"page": p["page"], "width": p["width"], "height": p["height"],
                   "raster_url": f"/raster/{ctx['doc_id']}/{p['page']}"}
                  for p in ctx["pages"]],
        "fraud_score": fraud_score,
        **adjudication,
        "band": esc["band"],
        "top_agents": meta["top_agents"],
        "shap_top_agents": meta["shap_top_agents"],
        "escalation_flags": esc["flags"],
        "flagged_regions": regions,
        "agent_findings": ctx.get("agent_findings", {}),
        "ood": ctx.get("understanding", {}).get("ood", {}),
        "ocr_engines_available": ctx.get("ocr_engines_available", []),
    }


def build_candidate_verdict(candidate_id: str, doc_verdicts: list[dict[str, Any]],
                            cross_doc: dict[str, Any]) -> dict[str, Any]:
    if doc_verdicts:
        overall_rank = max(_BAND_RANK[d["band"]] for d in doc_verdicts)
        overall_band = _RANK_BAND[overall_rank]
        overall_score = max(d["fraud_score"] for d in doc_verdicts)
    else:
        overall_band, overall_score = "P2", 0.0

    return {
        "candidate_id": candidate_id,
        "band": overall_band,
        "fraud_score": overall_score,
        "documents": doc_verdicts,
        "cross_doc_contradictions": cross_doc.get("contradictions", []),
        "cross_doc_backend": cross_doc.get("backend"),
        "summary": {
            "n_documents": len(doc_verdicts),
            "n_contradictions": len(cross_doc.get("contradictions", [])),
            "bands": {b: sum(1 for d in doc_verdicts if d["band"] == b)
                      for b in ("P0", "P1", "P2")},
        },
    }
