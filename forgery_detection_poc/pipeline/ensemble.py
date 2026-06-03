"""Step 4 - Specialist Agent Ensemble orchestration.

Runs all 11 mandatory agents in parallel via ThreadPoolExecutor. Each agent
returns a structured finding; failures degrade to error stubs (handled by the
@safe_agent decorator) so one agent can never crash the pipeline.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from pipeline.agents import (agent1_metadata, agent2_image_forensics,
                             agent3_font_layout, agent4_template,
                             agent5_duplicate, agent6_temporal,
                             agent7_ner_semantic, agent8_qr_barcode,
                             agent9_novel_fraud, agent10_cross_ocr,
                             agent11_adversarial)
from pipeline.utils import logger

AGENTS: list[Callable[[dict], dict]] = [
    agent1_metadata.run,
    agent2_image_forensics.run,
    agent3_font_layout.run,
    agent4_template.run,
    agent5_duplicate.run,
    agent6_temporal.run,
    agent7_ner_semantic.run,
    agent8_qr_barcode.run,
    agent9_novel_fraud.run,
    agent10_cross_ocr.run,
    agent11_adversarial.run,
]


def run_agents(ctx: dict[str, Any]) -> dict[str, dict[str, Any]]:
    findings: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(AGENTS)) as ex:
        futures = {ex.submit(fn, ctx): fn for fn in AGENTS}
        for fut, fn in futures.items():
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001  (defensive; agents are wrapped)
                logger.warning("Agent %s raised: %s", fn, exc)
                continue
            findings[res["agent_id"]] = res
    ctx["agent_findings"] = findings
    logger.info("Agents complete for %s: %s", ctx["filename"],
                {k: round(v.get("score", 0), 2) for k, v in findings.items()})
    return findings
