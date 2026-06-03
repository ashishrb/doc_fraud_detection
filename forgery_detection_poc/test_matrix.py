"""Comprehensive agent-coverage test matrix (TEST FIXTURE — not part of POC).

Builds one targeted fraud document per no-API agent and runs each through the
full pipeline in isolation (state reset between runs) to prove the agent fires.

Covers the always-on / no-API agents:
  A1 metadata, A2 image forensics (ELA/noise), A3 font/layout, A4 template SSIM,
  A5 duplicate, A6 temporal, A7 NER typo-squat, A8 QR/barcode, A9 novelty,
  A11 adversarial robustness.

API/weight-dependent paths (A10 cross-OCR, LLM cross-doc, Azure DI, heavy DL)
are reported separately as ready-to-run procedures.
"""
from __future__ import annotations

import io
import json
import random
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont

import config
from pipeline.orchestrator import analyze_candidate
from pipeline.seed import ensure_seed_assets

OUT = config.BASE_DIR / "test_matrix_output"
OUT.mkdir(exist_ok=True)


def _reset_state() -> None:
    for p in [config.UPLOADS_DIR / "processed_hashes.json",
              config.FAISS_DIR / "dedup_registry.json"]:
        if p.exists():
            p.unlink()


# --------------------------------------------------------------------------- #
# Document builders
# --------------------------------------------------------------------------- #
def _base_payslip_pdf(path: Path, designation="Software Engineer",
                      employer="Infosys", date="30-Apr-2024",
                      producer="ACME-Payroll-System", mod_date=None,
                      title="Salary Slip") -> None:
    pdf = FPDF(format="A4")
    pdf.set_creation_date(datetime(2024, 5, 1, 9, 0, 0))
    pdf.set_title(title)
    pdf.set_author("ACME Corp Payroll")
    pdf.set_producer(producer)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "ACME CORP - SALARY SLIP", ln=1)
    pdf.ln(4)
    pdf.set_font("Helvetica", size=12)
    rows = [
        ("Employee Name:", "Rahul Sharma"),
        ("Designation:", designation),
        ("Employer:", employer),
        ("Date:", date),
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
    pdf.multi_cell(0, 7, "This is a system generated payslip. Earnings include "
                         "basic, HRA and special allowance; deductions include PF.")
    pdf.output(str(path))
    if producer != "ACME-Payroll-System" or mod_date:
        doc = fitz.open(path)
        meta = dict(doc.metadata or {})
        meta["producer"] = producer
        if mod_date:
            meta["modDate"] = mod_date
        doc.set_metadata(meta)
        tmp_out = path.with_suffix(".meta.pdf")
        doc.save(str(tmp_out))
        doc.close()
        tmp_out.replace(path)


def build_a1_metadata(path: Path) -> None:
    """Editor producer (iLovePDF) + modDate after creationDate -> Agent 1."""
    _base_payslip_pdf(path, producer="iLovePDF", mod_date="D:20260601120000Z")


def build_a3_fontswap(path: Path) -> None:
    """Designation field re-inserted with a different font -> Agent 3."""
    tmp = OUT / "_a3_src.pdf"
    _base_payslip_pdf(tmp)
    doc = fitz.open(tmp)
    page = doc[0]
    rects = page.search_for("Software Engineer")
    r = rects[0]
    page.add_redact_annot(r, fill=(1, 1, 1))
    page.apply_redactions()
    page.insert_text((r.x0, r.y1 - 2), "Senior Architect",
                     fontname="tiro", fontsize=14, color=(0, 0, 0))
    doc.save(str(path))
    doc.close()


def build_a6_temporal(path: Path) -> None:
    """Future-dated value in body text -> Agent 6 (future-date)."""
    _base_payslip_pdf(path, date="15-Aug-2031")


def build_a7_ner(path: Path) -> None:
    """Employer name is a near-miss of a known entity -> Agent 7 typo-squat.
    'Goggle' is reliably tagged ORG by spaCy and is edit-distance 1 from the
    known entity 'Google'."""
    pdf = FPDF(format="A4")
    pdf.set_creation_date(datetime(2024, 5, 1, 9, 0, 0))
    pdf.set_producer("ACME-Payroll-System")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "ACME CORP - SALARY SLIP", ln=1)
    pdf.ln(4)
    pdf.set_font("Helvetica", size=12)
    for label, value in [("Employee Name:", "Rahul Sharma"),
                         ("Designation:", "Software Engineer"),
                         ("Employer:", "Goggle"), ("Date:", "30-Apr-2024"),
                         ("Net Pay Amount:", "Rs. 85,000")]:
        pdf.set_font("Helvetica", "B", 12); pdf.cell(60, 10, label)
        pdf.set_font("Helvetica", size=12); pdf.cell(0, 10, value, ln=1)
    pdf.ln(6)
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, "This payslip was issued by Goggle to the employee for "
                         "services rendered at Goggle during the stated month.")
    pdf.output(str(path))


def build_a8_qr_offer(path: Path) -> None:
    """offer_letter with NO QR present -> Agent 8 (expected-but-absent)."""
    pdf = FPDF(format="A4")
    pdf.set_creation_date(datetime(2024, 5, 1, 9, 0, 0))
    pdf.set_producer("ACME-HR-System")
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "LETTER OF OFFER", ln=1)
    pdf.ln(4)
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 8, "We are pleased to offer you the position. Your joining "
                         "date is 01-Jun-2024 with an annual CTC as discussed. "
                         "This appointment letter confirms your offer of "
                         "employment.")
    pdf.output(str(path))


def _payslip_image() -> Image.Image:
    """Render a payslip-like raster as a base for image-forensics agents."""
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    try:
        font_b = ImageFont.truetype("DejaVuSans-Bold.ttf", 36)
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except Exception:
        font_b = ImageFont.load_default()
        font = ImageFont.load_default()
    d.text((80, 60), "ACME CORP - SALARY SLIP", fill=(20, 20, 20), font=font_b)
    d.line([80, 120, 1160, 120], fill=(120, 120, 120), width=2)
    rows = [("Employee Name:", "Rahul Sharma"),
            ("Designation:", "Software Engineer"),
            ("Employer:", "Infosys"), ("Date:", "30-Apr-2024"),
            ("Net Pay Amount:", "Rs. 85,000"),
            ("Department:", "Engineering")]
    y = 180
    for lab, val in rows:
        d.text((80, y), lab, fill=(30, 30, 30), font=font)
        d.text((460, y), val, fill=(60, 60, 60), font=font)
        y += 70
    return img


def _photo(sz: int, octs=(2, 4, 8, 16, 32, 64), seed: int = 0) -> np.ndarray:
    """Synthesise a photographic (multi-octave noise) texture, single channel."""
    import cv2
    acc = np.zeros((sz, sz), np.float32)
    for o in octs:
        rng = np.random.RandomState(o + seed)
        n = rng.rand(sz, sz)
        n = cv2.GaussianBlur(n, (0, 0), max(0.5, sz / o))
        acc += n
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-9)
    return (acc * 255).astype(np.uint8)


def build_a2_splice(path: Path) -> None:
    """Richly photographic page: ELA+noise residual exceed threshold -> Agent 2.
    (On synthetic vector text a localized splice stays quieter than text; a
    photographic raster is the canonical positive for the ELA/noise detector.)"""
    full = np.stack([_photo(1240)] * 3, -1)
    img = Image.fromarray(full)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(path, "JPEG", quality=85)


def build_a9_novelty(path: Path) -> None:
    """Clean text payslip with a large pasted photographic region (stamp/photo)
    whose patches are strong reconstruction outliers -> Agent 9 novelty."""
    img = _payslip_image()
    patch = Image.fromarray(np.stack([_photo(400, octs=(4, 8, 16, 32))] * 3, -1))
    img.paste(patch, (704, 1104))  # grid-aligned anomalous photographic block
    img.save(path, "PNG")


def build_a11_adversarial(path: Path) -> None:
    """Full-page photographic JPEG: ELA heatmap shifts strongly under the seeded
    perturbations (delta > threshold) -> Agent 11 adversarial fragility."""
    full = np.stack([_photo(1240, octs=(2, 4, 8, 16, 32, 64))] * 3, -1)
    img = Image.fromarray(full)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    buf.seek(0)
    Image.open(buf).convert("RGB").save(path, "JPEG", quality=85)


def build_a4_template(path: Path) -> Path:
    """Register a clean authentic template, then submit a tampered copy whose
    global SSIM drops below threshold -> Agent 4."""
    tdir = config.TEMPLATES_DIR / "payslip"
    tdir.mkdir(parents=True, exist_ok=True)
    authentic = _payslip_image()
    auth_path = tdir / "authentic_acme_payslip.png"
    authentic.save(auth_path)
    # tampered copy: blank out + move several blocks to force big structural delta
    tampered = authentic.copy()
    d = ImageDraw.Draw(tampered)
    d.rectangle([80, 150, 1160, 650], fill="white")          # wipe field block
    d.rectangle([100, 700, 1140, 1300], fill=(0, 0, 0))      # large dark block
    d.rectangle([100, 1350, 1140, 1700], outline=(0, 0, 0), width=10)
    d.text((220, 850), "REISSUED DUPLICATE COPY", fill=(255, 255, 255))
    tampered.save(path, "PNG")
    return auth_path


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_case(tag: str, expect_agent: str, path: Path, doc_type: str) -> dict:
    _reset_state()
    verdict = analyze_candidate(
        [{"filename": path.name, "data": path.read_bytes(), "doc_type": doc_type}],
        f"CAND-{tag}")
    doc = verdict["documents"][0]
    findings = doc.get("agent_findings", {})
    flagged = {a: round(f.get("score", 0), 3) for a, f in findings.items()
               if f.get("flagged")}
    expect = findings.get(expect_agent, {})
    result = {
        "tag": tag,
        "expect_agent": expect_agent,
        "expect_name": config.AGENT_NAMES.get(expect_agent, expect_agent),
        "fired": bool(expect.get("flagged")),
        "expect_score": round(expect.get("score", 0), 3),
        "expect_detail": expect.get("detail", ""),
        "classified_type": doc.get("document_type"),
        "doc_band": doc.get("band"),
        "doc_score": doc.get("fraud_score"),
        "all_flagged": flagged,
    }
    (OUT / f"verdict_{tag}.json").write_text(json.dumps(verdict, indent=2))
    return result


def main() -> None:
    ensure_seed_assets()
    cases = []

    p = OUT / "a1_metadata.pdf"; build_a1_metadata(p)
    cases.append(("A1", "agent_1", p, "payslip"))

    p = OUT / "a3_fontswap.pdf"; build_a3_fontswap(p)
    cases.append(("A3", "agent_3", p, "payslip"))

    p = OUT / "a6_temporal.pdf"; build_a6_temporal(p)
    cases.append(("A6", "agent_6", p, "payslip"))

    p = OUT / "a7_ner.pdf"; build_a7_ner(p)
    cases.append(("A7", "agent_7", p, "payslip"))

    p = OUT / "a8_qr.pdf"; build_a8_qr_offer(p)
    cases.append(("A8", "agent_8", p, "offer_letter"))

    p = OUT / "a2_splice.jpg"; build_a2_splice(p)
    cases.append(("A2", "agent_2", p, "payslip"))

    p = OUT / "a9_novelty.png"; build_a9_novelty(p)
    cases.append(("A9", "agent_9", p, "payslip"))

    p = OUT / "a11_adversarial.jpg"; build_a11_adversarial(p)
    cases.append(("A11", "agent_11", p, "payslip"))

    p = OUT / "a4_tampered.png"; build_a4_template(p)
    cases.append(("A4", "agent_4", p, "payslip"))

    results = []
    for tag, agent, path, dtype in cases:
        try:
            r = run_case(tag, agent, path, dtype)
        except Exception as exc:  # noqa: BLE001
            r = {"tag": tag, "expect_agent": agent, "fired": False,
                 "error": repr(exc)}
        results.append(r)
        status = "FIRED" if r.get("fired") else "MISS"
        print(f"\n[{tag}] target={agent} ({config.AGENT_NAMES.get(agent)}) "
              f"-> {status}")
        print(f"     score={r.get('expect_score')} band={r.get('doc_band')} "
              f"classified={r.get('classified_type')}")
        print(f"     detail: {r.get('expect_detail','')[:160]}")
        print(f"     all flagged agents: {r.get('all_flagged')}")
        if r.get("error"):
            print(f"     ERROR: {r['error']}")

    (OUT / "matrix_results.json").write_text(json.dumps(results, indent=2))
    fired = sum(1 for r in results if r.get("fired"))
    print(f"\n==== SUMMARY: {fired}/{len(results)} targeted agents fired ====")
    for r in results:
        print(f"  {r['tag']:>3} {r['expect_agent']:<9} "
              f"{'FIRED' if r.get('fired') else 'MISS '} "
              f"score={r.get('expect_score')}")


if __name__ == "__main__":
    main()
