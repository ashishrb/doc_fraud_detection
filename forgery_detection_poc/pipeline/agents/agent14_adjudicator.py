"""Agent 14 - Cross-Agent Adjudicator [Good-to-have].

Runs AFTER all other agents complete - it is NOT part of the parallel ensemble
batch. It fires only when Rule 2 (disagreement) triggers in pipeline/verdict.py
(2+ agents disagree strongly with the ensemble consensus). Using Claude Opus
(config.AGENT14_MODEL) it weighs all agent evidence and produces a reasoned
consensus verdict so ambiguous cases do not automatically escalate to human
review.

It never raises: on any error it returns an error stub and the caller leaves the
original ensemble score unchanged.
"""
from __future__ import annotations

import json
import re
from typing import Any

import config
from pipeline.utils import logger

AGENT_ID = "agent_14"

SYSTEM_PROMPT = """You are a fraud adjudicator reviewing evidence from specialist document detection agents.
You will receive structured findings from multiple agents that have each examined the same document.
Some agents flagged the document as fraudulent; others did not.
Agent credibility guide (use to weight evidence):
- High credibility (weight ≥0.85): agent_3 (font/layout), agent_9 (novelty), agent_12 (PDF layers), agent_13 (plausibility), agent_10 (cross-OCR — only when ENABLE_PADDLEOCR=1, otherwise absent from findings; do not treat its absence as negative evidence)
- Medium credibility (weight 0.7–0.84): agent_1 (metadata), agent_4 (template), agent_6 (temporal), agent_7 (NER)
- Lower credibility (weight ≤0.69): agent_2 (image — no deep models provisioned), agent_5 (duplicate), agent_8 (QR), agent_11 (adversarial)
Weight high-credibility agent findings more heavily when they contradict lower-credibility agents. If an agent is absent from the findings list, it is unconfigured or errored — treat its absence as missing evidence only, never as a clean signal.
Your job is to read all the evidence, weigh it, and produce a reasoned consensus verdict.

Respond ONLY with a JSON object — no preamble, no markdown fences:
{
  "adjudicated_score": <float 0.0–1.0 fraud probability>,
  "reasoning": "<two sentences maximum>",
  "dominant_agents": ["<agent_id>", ...],
  "overridden_agents": ["<agent_id>", ...]
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


def run(agent_findings: dict[str, Any],
        ensemble_score: float) -> dict[str, Any]:
    if not config.ANTHROPIC_API_KEY:
        return _error("ANTHROPIC_API_KEY not configured")
    try:
        import anthropic

        payload = json.dumps({
            "ensemble_score": float(ensemble_score),
            "note": "Only non-errored agent findings are included. Absent "
                    "agents (API key missing, model unavailable) are excluded "
                    "and must not be treated as negative evidence.",
            "agent_findings": [
                f for f in agent_findings.values()
                if "error" not in f
            ],
        }, indent=2, default=str)

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=config.AGENT14_MODEL,
            max_tokens=600,
            temperature=0.1,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": payload}],
        )
        data = _parse_json_object(resp.content[0].text)
        adjudicated = float(data.get("adjudicated_score", ensemble_score))
        return {
            "agent_id": AGENT_ID,
            "score": round(adjudicated, 3),
            "flagged": adjudicated >= config.P1_MAX,
            "flagged_regions": [],
            "detail": data.get("reasoning", ""),
        }
    except Exception as exc:  # noqa: BLE001  (never crash the pipeline)
        logger.warning("%s failed: %s", AGENT_ID, exc)
        return _error(str(exc))
