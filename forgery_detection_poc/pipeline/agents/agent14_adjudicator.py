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
            "agent_findings": list(agent_findings.values()),
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
