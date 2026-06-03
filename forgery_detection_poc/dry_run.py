"""Dry run for the forgery detection POC.

  - Test Case A (clean): a synthetic payslip generated with fpdf2.
  - Test Case B (forged): a copy of A whose 'Designation' field is altered with
    PyMuPDF (different font + size) to simulate a text-level manipulation.

Runs A alone, then the A+B pair (to exercise cross-document reasoning), prints a
structured report and writes verdict JSONs to dry_run_output/.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from fpdf import FPDF

import config
from pipeline.orchestrator import analyze_candidate
from pipeline.seed import ensure_seed_assets

OUT = config.BASE_DIR / "dry_run_output"
OUT.mkdir(exist_ok=True)

ORIG_DESIGNATION = "Software Engineer"
FORGED_DESIGNATION = "Senior Architect"


def make_clean_payslip(path: Path) -> None:
    pdf = FPDF(format="A4")
    pdf.set_creation_date(datetime(2024, 5, 1, 9, 0, 0))  # metadata creation
    pdf.set_title("Salary Slip")
    pdf.set_author("ACME Corp Payroll")
    pdf.set_producer("ACME-Payroll-System")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "ACME CORP - SALARY SLIP", ln=1)
    pdf.ln(4)
    pdf.set_font("Helvetica", size=12)
    rows = [
        ("Employee Name:", "Rahul Sharma"),
        ("Designation:", ORIG_DESIGNATION),
        ("Employer:", "Infosys"),
        ("Date:", "30-Apr-2024"),
        ("Net Pay Amount:", "Rs. 85,000"),
        ("Department:", "Engineering"),
        ("Location:", "Bengaluru"),
        ("Reference:", "PAY-2024-04-1187"),
    ]
    for label, value in rows:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(60, 10, label)
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, value, ln=1)
    pdf.ln(6)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, "This is a system generated payslip for the salary "
                         "disbursed for the stated month. Earnings include basic, "
                         "HRA and special allowance; deductions include PF and "
                         "professional tax.")
    pdf.ln(20)
    pdf.cell(0, 10, "Authorised Signatory", ln=1)
    pdf.output(str(path))


def make_forged(src: Path, dst: Path) -> bool:
    """Alter the designation field with a different font + size (text-level
    manipulation). Returns True if the field was located and edited."""
    doc = fitz.open(src)
    page = doc[0]
    rects = page.search_for(ORIG_DESIGNATION)
    if not rects:
        doc.close()
        return False
    r = rects[0]
    page.add_redact_annot(r, fill=(1, 1, 1))
    page.apply_redactions()
    # Insert replacement with a DIFFERENT font (Times) and size (14 vs 12).
    page.insert_text((r.x0, r.y1 - 2), FORGED_DESIGNATION,
                     fontname="tiro", fontsize=14, color=(0, 0, 0))
    # Simulate the doc having been edited in a consumer PDF tool: a tell-tale
    # editor 'producer' string and a modification date after issuance.
    meta = dict(doc.metadata or {})
    meta["producer"] = "iLovePDF"
    meta["modDate"] = "D:20260601120000Z"
    doc.set_metadata(meta)
    doc.save(str(dst))
    doc.close()
    return True


def _b(path: Path) -> bytes:
    return path.read_bytes()


def summarise(tag: str, verdict: dict) -> None:
    print(f"\n===== {tag} =====")
    print(f"candidate band: {verdict['band']}  fraud_score: {verdict['fraud_score']}")
    for d in verdict["documents"]:
        if d.get("error"):
            print(f"  - {d['filename']}: SKIPPED ({d['error']})")
            continue
        print(f"  - {d['filename']}  band={d['band']} score={d['fraud_score']} "
              f"type={d.get('document_type')}")
        print(f"      OCR engines: {d.get('ocr_engines_available')}")
        fired = {a: round(f.get('score', 0), 3)
                 for a, f in d.get('agent_findings', {}).items()
                 if f.get('flagged')}
        print(f"      agents flagged: {fired or 'none'}")
        errs = {a: f.get('error') for a, f in d.get('agent_findings', {}).items()
                if f.get('error')}
        if errs:
            print(f"      agents errored/skipped: {errs}")
        print(f"      flagged regions: {len(d.get('flagged_regions', []))}")
        print(f"      escalation: {d.get('escalation_flags')}")
    if verdict.get("cross_doc_contradictions"):
        print(f"  cross-doc [{verdict.get('cross_doc_backend')}]:")
        for c in verdict["cross_doc_contradictions"]:
            print(f"    * {c.get('contradiction_type')} "
                  f"{c.get('documents_involved')}: "
                  f"'{c.get('doc1_quote')}' vs '{c.get('doc2_quote')}'")


def _reset_state() -> None:
    """Clear stateful registries so the dry run is reproducible."""
    for p in [config.UPLOADS_DIR / "processed_hashes.json",
              config.FAISS_DIR / "dedup_registry.json"]:
        if p.exists():
            p.unlink()


def main() -> None:
    ensure_seed_assets()
    _reset_state()
    a_path = OUT / "testA_clean_payslip.pdf"
    b_path = OUT / "testB_forged_payslip.pdf"
    make_clean_payslip(a_path)
    edited = make_forged(a_path, b_path)
    print(f"Generated Test Case A: {a_path}")
    print(f"Generated Test Case B (designation edited={edited}): {b_path}")

    # Test Case A alone (expect P2)
    _reset_state()
    vA = analyze_candidate(
        [{"filename": a_path.name, "data": _b(a_path), "doc_type": "payslip"}],
        "CAND-A")
    summarise("TEST CASE A (clean, single doc)", vA)
    (OUT / "verdict_A.json").write_text(json.dumps(vA, indent=2))

    # Test Case B as candidate pair A+B (expect B flagged + contradiction)
    _reset_state()
    vB = analyze_candidate([
        {"filename": a_path.name, "data": _b(a_path), "doc_type": "payslip"},
        {"filename": b_path.name, "data": _b(b_path), "doc_type": "payslip"},
    ], "CAND-B")
    summarise("TEST CASE B (clean + forged pair)", vB)
    (OUT / "verdict_B.json").write_text(json.dumps(vB, indent=2))

    print("\nVerdict JSONs written to", OUT)


if __name__ == "__main__":
    main()
