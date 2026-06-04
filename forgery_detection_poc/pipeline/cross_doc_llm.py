"""Step 5 - Cross-Document LLM Reasoning.

Bundles extracted text + field JSONs from all of a candidate's documents and
asks an LLM to enumerate inter-document contradictions. Backend preference:
Azure OpenAI (Azure AI Foundry) -> OpenAI -> Anthropic. When no API key/SDK is
available, a deterministic rule-based fallback produces the same contradiction
schema so the pipeline still surfaces cross-doc fraud.
"""
from __future__ import annotations

import json
import re
from typing import Any

import config
from pipeline.utils import logger

SYSTEM_PROMPT = """You are a document fraud detection expert. You will receive extracted text and structured fields from multiple candidate documents (payslips, experience letters, offer letters, Form 16, certificates). 

Your task: identify ALL inter-document contradictions. For each contradiction, produce a JSON object with:
- "contradiction_type": one of [designation_mismatch, date_overlap, entity_mismatch, salary_break, date_inversion]
- "documents_involved": [list of document names]
- "doc1_quote": exact extracted text from document 1
- "doc2_quote": exact extracted text from document 2  
- "confidence": float 0.0-1.0
- "explanation": one sentence

Respond with ONLY a JSON array of contradiction objects. No preamble, no markdown fences."""


def _build_user_payload(docs: list[dict[str, Any]]) -> str:
    bundle = []
    for d in docs:
        bundle.append({
            "document_name": d["filename"],
            "document_type": d.get("classified_type", d.get("doc_type")),
            "fields": {f["field"]: f["value"]
                       for f in d.get("understanding", {}).get("fields", [])},
            "text_excerpt": (d.get("text", "") or "")[:3000],
        })
    return json.dumps(bundle, indent=2)


def _azure_openai_configured() -> bool:
    return bool(config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_API_KEY
                and config.AZURE_OPENAI_DEPLOYMENT)


def _call_azure_openai(payload: str) -> list[dict[str, Any]]:
    from openai import AzureOpenAI

    client = AzureOpenAI(
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )
    # On Azure OpenAI the model is addressed by the deployment name.
    resp = client.chat.completions.create(
        model=config.AZURE_OPENAI_DEPLOYMENT,
        temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": payload}],
    )
    return _parse_json_array(resp.choices[0].message.content)


def _call_openai(payload: str) -> list[dict[str, Any]]:
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=config.CROSS_DOC_MODEL,
        temperature=0.1,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": payload}],
    )
    return _parse_json_array(resp.choices[0].message.content)


def _call_anthropic(payload: str) -> list[dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.CROSS_DOC_MODEL,
        max_tokens=2000,
        temperature=0.1,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": payload}],
    )
    return _parse_json_array(resp.content[0].text)


def _parse_json_array(text: str) -> list[dict[str, Any]]:
    if not text:
        return []
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse LLM contradiction JSON: %s", exc)
        return []


def _fields_map(doc: dict[str, Any]) -> dict[str, str]:
    return {f["field"]: f["value"].lower().strip()
            for f in doc.get("understanding", {}).get("fields", [])}


def _rule_based(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic contradiction finder used when no LLM key is configured."""
    out: list[dict[str, Any]] = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            fa, fb = _fields_map(docs[i]), _fields_map(docs[j])
            pair = [docs[i]["filename"], docs[j]["filename"]]
            for field, ctype in (("designation", "designation_mismatch"),
                                 ("employer", "entity_mismatch")):
                if field in fa and field in fb and fa[field] != fb[field]:
                    out.append({
                        "contradiction_type": ctype,
                        "documents_involved": pair,
                        "doc1_quote": fa[field],
                        "doc2_quote": fb[field],
                        "confidence": 0.7,
                        "explanation": f"{field} differs between the two documents.",
                    })
            if "amount" in fa and "amount" in fb:
                na = re.sub(r"[^0-9.]", "", fa["amount"])
                nb = re.sub(r"[^0-9.]", "", fb["amount"])
                if na and nb and na != nb:
                    out.append({
                        "contradiction_type": "salary_break",
                        "documents_involved": pair,
                        "doc1_quote": fa["amount"],
                        "doc2_quote": fb["amount"],
                        "confidence": 0.6,
                        "explanation": "Salary/amount differs between documents.",
                    })
    return out


def analyze_cross_document(docs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(docs) < 2:
        return {"contradictions": [], "backend": "skipped (single document)"}

    payload = _build_user_payload(docs)
    backend = "rule_based_fallback"
    contradictions: list[dict[str, Any]] = []

    try:
        if _azure_openai_configured():
            contradictions = _call_azure_openai(payload)
            backend = f"azure_openai:{config.AZURE_OPENAI_DEPLOYMENT}"
        elif config.OPENAI_API_KEY:
            contradictions = _call_openai(payload)
            backend = f"openai:{config.CROSS_DOC_MODEL}"
        elif config.ANTHROPIC_API_KEY:
            contradictions = _call_anthropic(payload)
            backend = f"anthropic:{config.CROSS_DOC_MODEL}"
        else:
            contradictions = _rule_based(docs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM cross-doc failed (%s); using rule-based fallback", exc)
        contradictions = _rule_based(docs)
        backend = f"rule_based_fallback (LLM error: {exc})"

    logger.info("Cross-doc reasoning [%s]: %d contradiction(s)", backend,
                len(contradictions))
    return {"contradictions": contradictions, "backend": backend}
