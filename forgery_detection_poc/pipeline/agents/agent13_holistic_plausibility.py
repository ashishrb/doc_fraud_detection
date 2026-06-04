"""Agent 13 - Holistic Plausibility [Good-to-have].

Catches Category 7 fraud (document-level implausibility) - the only agent that
does. Uses Claude Opus (config.AGENT13_MODEL) to read the full document text +
extracted fields and judge whether the document plausibly exists in the real
world (implausible designation/issuer, anachronisms, wrong tax slabs, etc.).

Runs inside the parallel ensemble batch (pipeline/ensemble.py). It never raises:
on any error (missing API key, network failure, JSON parse failure) it returns a
neutral, non-flagging stub so the pipeline continues uninterrupted.
"""
from __future__ import annotations

import json
import re
from typing import Any

import config
from pipeline.utils import logger

AGENT_ID = "agent_13"

SYSTEM_PROMPT = """You are a document fraud expert with deep knowledge of Indian corporate structures,
HR practices, tax regulations, and government-issued documents.
You will receive the full text and extracted fields of a single BGV document.
Judge whether this document plausibly exists in the real world.

Check specifically for:
- Implausible designation for the claimed issuer size (e.g. Senior Vice President at a 12-person sole proprietorship)
- Anachronistic content (references to events, regulations, or structures that post-date the document's claimed date)
- Incorrect tax slabs or assessment year references that do not match Indian tax law for that period
- Issuers that could not plausibly issue this document type (no HR department, no payroll registration, no tax registration)

Respond ONLY with a JSON object — no preamble, no markdown fences:
{
  "plausibility_score": <float 0.0–1.0, where 1.0 = fully plausible and 0.0 = implausible>,
  "flagged": <true|false>,
  "red_flags": ["<finding 1>", "<finding 2>"],
  "rationale": "<one sentence>"
}"""


def _error(reason: str) -> dict[str, Any]:
    return {"agent_id": AGENT_ID, "score": 0.0, "flagged": False,
            "error": reason}


def _parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("empty response")
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        text = m.group(0)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def _build_user_payload(ctx: dict[str, Any]) -> str:
    fields = ctx.get("understanding", {}).get("fields", [])
    bundle = {
        "document_type": ctx.get("classified_type", ctx.get("doc_type")),
        "extracted_fields": {f["field"]: f["value"] for f in fields},
        "document_text": (ctx.get("text", "") or "")[:6000],
    }
    return json.dumps(bundle, indent=2)


def run(ctx: dict[str, Any]) -> dict[str, Any]:
    if not config.ANTHROPIC_API_KEY:
        return _error("ANTHROPIC_API_KEY not configured")
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=config.AGENT13_MODEL,
            max_tokens=500,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_payload(ctx)}],
        )
        data = _parse_json_object(resp.content[0].text)
        plausibility = float(data.get("plausibility_score", 1.0))
        red_flags = data.get("red_flags", []) or []
        rationale = data.get("rationale", "")
        detail = rationale
        if red_flags:
            detail = f"{rationale} | Red flags: {', '.join(red_flags)}"
        return {
            "agent_id": AGENT_ID,
            "score": round(1.0 - plausibility, 3),
            "flagged": bool(data.get("flagged", False)),
            "flagged_regions": [],
            "detail": detail,
        }
    except Exception as exc:  # noqa: BLE001  (never crash the pipeline)
        logger.warning("%s failed: %s", AGENT_ID, exc)
        return _error(str(exc))
