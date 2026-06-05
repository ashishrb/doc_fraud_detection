"""Operator tool: generate a labeled training dataset for the meta-learner.

Produces ~500 document feature vectors (250 clean label=0 + 250 forged label=1)
using the same fpdf2 synthetic-PDF approach as dry_run.py. It does NOT call the
full pipeline (no orchestrator / verdict / cross-doc). For each synthetic
document it builds a minimal context, runs only the lightweight specialist
agents directly, and assembles the exact feature vector that
pipeline/meta_learner.py feeds the LightGBM model at inference time.

Forged documents inject one or more known fraud signals programmatically:
  - font swap on the Designation field (Agent 3)
  - iLovePDF producer string in the PDF metadata (Agent 1)
  - post-creation modification date (Agent 1)
  - near-duplicate pHash (a copy of a clean doc) (Agent 5)
  - an overlapping text block (Agent 12 / layout)

Image-forensics agents that require raster processing (2, 4, 9) and the LLM
agent (13) are set to 0.0 to keep generation fast, matching the prompt spec.

Usage (from the forgery_detection_poc/ directory):

    python scripts/generate_synthetic_labels.py

Output: models/synthetic_labels.json (feature_names, X, y, generated_at, n_samples)
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
import tempfile
import uuid
from datetime import datetime

import fitz  # PyMuPDF
from fpdf import FPDF

# Make the project root importable when run as `python scripts/generate_...py`.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline.agents import (agent1_metadata, agent3_font_layout,  # noqa: E402
                             agent5_duplicate, agent6_temporal,
                             agent7_ner_semantic, agent8_qr_barcode,
                             agent10_cross_ocr, agent11_adversarial)
from pipeline.meta_learner import _FEATURE_NAMES, _feature_vector  # noqa: E402
from pipeline.preprocessing import preprocess  # noqa: E402
from pipeline.utils import logger, render_to_rasters  # noqa: E402

# Lightweight agents run directly (image agents 2/4/9 + LLM 13 -> 0.0).
_LIGHTWEIGHT_AGENTS = [
    agent1_metadata.run,
    agent3_font_layout.run,
    agent5_duplicate.run,
    agent6_temporal.run,
    agent7_ner_semantic.run,
    agent8_qr_barcode.run,
    agent10_cross_ocr.run,
    agent11_adversarial.run,
]

_NAMES = ["Rahul Sharma", "Priya Singh", "Amit Patel", "Sneha Gupta",
          "Vikram Rao", "Anjali Mehta", "Karan Nair", "Divya Iyer",
          "Rohit Verma", "Pooja Reddy"]
_DESIGNATIONS = ["Software Engineer", "Senior Engineer", "Analyst", "Consultant",
                 "Team Lead", "Project Manager", "Associate", "Specialist"]
_EMPLOYERS = ["Infosys", "Wipro", "Cognizant", "Accenture", "TCS", "Capgemini"]
_DEPARTMENTS = ["Engineering", "Finance", "Operations", "Sales", "HR"]
_LOCATIONS = ["Bengaluru", "Pune", "Hyderabad", "Chennai", "Mumbai"]


def _reset_dedup_registry() -> None:
    reg = config.FAISS_DIR / "dedup_registry.json"
    if reg.exists():
        reg.unlink()


def _make_payslip(path: pathlib.Path, fields: dict, designation: str) -> None:
    pdf = FPDF(format="A4")
    pdf.set_creation_date(datetime(2024, 5, 1, 9, 0, 0))
    pdf.set_title("Salary Slip")
    pdf.set_author("ACME Corp Payroll")
    pdf.set_producer("ACME-Payroll-System")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "ACME CORP - SALARY SLIP", ln=1)
    pdf.ln(4)
    rows = [
        ("Employee Name:", fields["name"]),
        ("Designation:", designation),
        ("Employer:", fields["employer"]),
        ("Date:", fields["date"]),
        ("Net Pay Amount:", fields["amount"]),
        ("Department:", fields["department"]),
        ("Location:", fields["location"]),
        ("Reference:", fields["reference"]),
    ]
    for label, value in rows:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(60, 10, label)
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, value, ln=1)
    pdf.ln(6)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, "This is a system generated payslip for the salary "
                         "disbursed for the stated month.")
    pdf.ln(20)
    pdf.cell(0, 10, "Authorised Signatory", ln=1)
    pdf.output(str(path))


def _forge(src: pathlib.Path, dst: pathlib.Path, designation: str,
           signals: set[str]) -> None:
    """Apply the chosen fraud signals to a clean payslip copy."""
    doc = fitz.open(src)
    page = doc[0]
    if "font_swap" in signals:
        rects = page.search_for(designation)
        if rects:
            r = rects[0]
            page.add_redact_annot(r, fill=(1, 1, 1))
            page.apply_redactions()
            page.insert_text((r.x0, r.y1 - 2), random.choice(_DESIGNATIONS),
                             fontname="tiro", fontsize=14, color=(0, 0, 0))
    if "overlap" in signals:
        # Insert an overlapping text block over an existing line.
        page.insert_text((70, 120), "REVISED", fontname="tiro", fontsize=20,
                         color=(0, 0, 0), overlay=True)
    meta = dict(doc.metadata or {})
    if "producer" in signals:
        meta["producer"] = "iLovePDF"
    if "moddate" in signals:
        meta["modDate"] = "D:20260601120000Z"
    doc.set_metadata(meta)
    doc.save(str(dst))
    doc.close()


def _random_fields() -> dict:
    return {
        "name": random.choice(_NAMES),
        "employer": random.choice(_EMPLOYERS),
        "date": f"{random.randint(1, 28):02d}-Apr-2024",
        "amount": f"Rs. {random.randint(40, 150)},000",
        "department": random.choice(_DEPARTMENTS),
        "location": random.choice(_LOCATIONS),
        "reference": f"PAY-2024-04-{random.randint(1000, 9999)}",
    }


def _build_ctx(pdf_path: pathlib.Path, work_dir: pathlib.Path) -> dict:
    doc_id = uuid.uuid4().hex[:12]
    data = pdf_path.read_bytes()
    import hashlib
    ctx = {
        "doc_id": doc_id,
        "filename": pdf_path.name,
        "candidate_id": "SYN",
        "doc_type": "payslip",
        "is_pdf": True,
        "path": str(pdf_path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "pages": render_to_rasters(pdf_path, work_dir / doc_id),
    }
    preprocess(ctx)
    return ctx


def _run_agents(ctx: dict) -> dict:
    findings: dict[str, dict] = {}
    for fn in _LIGHTWEIGHT_AGENTS:
        res = fn(ctx)
        findings[res["agent_id"]] = res
    return findings


def main() -> None:
    random.seed(42)
    n_clean = n_forged = 250
    X: list[list[float]] = []
    y: list[int] = []

    tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="synlabels_"))
    logger.info("Generating synthetic labels in %s", tmp_root)

    # --- Clean documents (label 0) ---
    for i in range(n_clean):
        _reset_dedup_registry()
        fields = _random_fields()
        designation = random.choice(_DESIGNATIONS)
        clean_pdf = tmp_root / f"clean_{i}.pdf"
        _make_payslip(clean_pdf, fields, designation)
        ctx = _build_ctx(clean_pdf, tmp_root)
        findings = _run_agents(ctx)
        X.append(_feature_vector(findings, 0).flatten().tolist())
        y.append(0)
        if (i + 1) % 50 == 0:
            logger.info("  clean %d/%d", i + 1, n_clean)

    # --- Forged documents (label 1) ---
    all_signals = ["font_swap", "producer", "moddate", "overlap", "near_dup"]
    for i in range(n_forged):
        _reset_dedup_registry()
        fields = _random_fields()
        designation = random.choice(_DESIGNATIONS)
        base_pdf = tmp_root / f"fbase_{i}.pdf"
        _make_payslip(base_pdf, fields, designation)

        k = random.randint(1, 3)
        signals = set(random.sample(all_signals, k))
        # Guarantee at least one metadata/font signal so the doc is detectable.
        if not signals & {"font_swap", "producer", "moddate"}:
            signals.add(random.choice(["font_swap", "producer", "moddate"]))

        if "near_dup" in signals:
            # Register the clean base so Agent 5 sees the forged copy as a dup.
            base_ctx = _build_ctx(base_pdf, tmp_root)
            agent5_duplicate.run(base_ctx)

        forged_pdf = tmp_root / f"forged_{i}.pdf"
        _forge(base_pdf, forged_pdf, designation, signals)
        ctx = _build_ctx(forged_pdf, tmp_root)
        findings = _run_agents(ctx)
        X.append(_feature_vector(findings, 0).flatten().tolist())
        y.append(1)
        if (i + 1) % 50 == 0:
            logger.info("  forged %d/%d", i + 1, n_forged)

    _reset_dedup_registry()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.MODELS_DIR / "synthetic_labels.json"
    out.write_text(json.dumps({
        "feature_names": list(_FEATURE_NAMES),
        "X": X,
        "y": y,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "n_samples": len(y),
    }, indent=2))

    print(f"Generated {n_clean} clean + {n_forged} forged = {len(y)} samples. "
          f"Saved to models/synthetic_labels.json.")


if __name__ == "__main__":
    main()
