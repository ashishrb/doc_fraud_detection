"""Lazy singleton loader for spaCy (shared by Agents 6 and 7)."""
from __future__ import annotations

from pipeline.utils import logger

_NLP = None
_TRIED = False


def get_spacy():
    global _NLP, _TRIED
    if _TRIED:
        return _NLP
    _TRIED = True
    try:
        import spacy

        _NLP = spacy.load("en_core_web_sm")
        logger.info("spaCy en_core_web_sm loaded")
    except Exception as exc:  # noqa: BLE001
        logger.warning("spaCy unavailable: %s", exc)
        _NLP = None
    return _NLP
