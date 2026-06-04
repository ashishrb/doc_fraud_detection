"""Agent 7 - NER / Semantic Analysis [Mandatory].

spaCy NER for ORG/PERSON/MONEY/DATE entities, then SymSpell-based near-miss
detection of known company names (e.g. 'lnfosys' vs 'Infosys'). Entities within
edit distance <= 2 of a legitimate name (but not an exact match) are flagged as
typo-squatting / impersonation.
"""
from __future__ import annotations

from typing import Any

import config
from pipeline.nlp import get_spacy
from pipeline.utils import locate_text, safe_agent

AGENT_ID = "agent_7"
_MAX_EDIT = 2


def _levenshtein(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _build_symspell():
    try:
        from symspellpy import SymSpell

        sym = SymSpell(max_dictionary_edit_distance=_MAX_EDIT)
        for ent in config.KNOWN_ENTITIES:
            sym.create_dictionary_entry(ent.lower(), 1)
            for tok in ent.lower().split():
                if len(tok) > 3:
                    sym.create_dictionary_entry(tok, 1)
        return sym
    except Exception:  # noqa: BLE001
        return None


@safe_agent(AGENT_ID)
def run(ctx: dict[str, Any]) -> dict[str, Any]:
    nlp = get_spacy()
    if nlp is None:
        return {"score": 0.0, "flagged": False, "flagged_regions": [],
                "detail": "spaCy unavailable"}

    doc = nlp((ctx.get("text", "") or "")[:100000])
    entities = {"ORG": [], "PERSON": [], "MONEY": [], "DATE": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)

    sym = _build_symspell()
    regions: list[dict[str, Any]] = []
    reasons: list[str] = []
    score = 0.0
    known_lower = {e.lower() for e in config.KNOWN_ENTITIES}

    for org in set(entities["ORG"]):
        org_l = org.lower().strip()
        if not org_l or org_l in known_lower:
            continue
        # nearest known entity by Levenshtein
        best_name, best_d = None, 99
        for known in config.KNOWN_ENTITIES:
            d = _levenshtein(org_l, known.lower())
            if d < best_d:
                best_name, best_d = known, d
        sym_hit = None
        if sym is not None:
            from symspellpy import Verbosity

            sug = sym.lookup(org_l, Verbosity.CLOSEST, max_edit_distance=_MAX_EDIT)
            if sug and sug[0].term != org_l:
                sym_hit = sug[0].term
        if best_name and 0 < best_d <= _MAX_EDIT:
            score = max(score, 0.75)
            reasons.append(f"'{org}' is edit-distance {best_d} from "
                           f"legitimate '{best_name}' (possible typo-squat)")
            bbox, page = locate_text(org, ctx.get("words", []))
            if bbox:
                regions.append({"page": page, "bbox": bbox,
                                "reason": f"near-miss of '{best_name}' (dist={best_d})",
                                "confidence": 0.75})
        elif sym_hit:
            score = max(score, 0.5)
            reasons.append(f"'{org}' close to known term '{sym_hit}'")

    return {
        "score": round(score, 3),
        "flagged": bool(regions),
        "flagged_regions": regions,
        "detail": "; ".join(reasons) if reasons
                  else f"entities OK (ORG={len(set(entities['ORG']))})",
        "entities": {k: sorted(set(v)) for k, v in entities.items()},
    }
