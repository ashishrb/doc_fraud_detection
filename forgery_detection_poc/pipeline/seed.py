"""Project-init asset seeding.

Creates placeholder authentic template images (2-3 per document type) used by
Agent 4 (template matching), Agent 9 (novelty training) and the Step 3 FAISS
OOD index, plus the Agent 11 adversarial-pattern manifest. Idempotent.
"""
from __future__ import annotations

import json

import config
from pipeline.utils import logger

_W, _H = 1240, 1754  # ~A4 at 150 DPI

_TYPE_TITLES = {
    "payslip": ["ACME CORP - SALARY SLIP", "Payslip for the month",
                "Employee Net Pay Statement"],
    "experience_letter": ["EXPERIENCE CERTIFICATE", "To Whom It May Concern",
                          "Service & Relieving Letter"],
    "offer_letter": ["LETTER OF OFFER", "Offer of Employment",
                    "Appointment Letter"],
    "form16": ["FORM NO. 16", "Certificate under Section 203",
              "TDS Certificate - Form 16"],
    "education_certificate": ["DEGREE CERTIFICATE", "Statement of Marks",
                             "University Provisional Certificate"],
}


def _make_template(title: str, seed: int):
    import random

    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    img = Image.new("RGB", (_W, _H), "white")
    d = ImageDraw.Draw(img)
    # logo box (top-left)
    d.rectangle([60, 50, 240, 150], outline=(40, 70, 140), width=4)
    d.text((80, 90), "LOGO", fill=(40, 70, 140))
    # title
    d.text((300, 90), title, fill=(20, 20, 20))
    d.line([60, 180, _W - 60, 180], fill=(120, 120, 120), width=2)
    # field rows
    labels = ["Name:", "Designation:", "Employer:", "Date:", "Amount:",
              "Reference:", "Department:", "Location:"]
    y = 240
    for lab in labels:
        d.text((80, y), lab, fill=(30, 30, 30))
        d.line([360, y + 18, 900, y + 18], fill=(180, 180, 180), width=1)
        d.text((380, y), "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ ")
                                 for _ in range(12)), fill=(60, 60, 60))
        y += 70
    # body block
    for i in range(10):
        d.line([80, y, _W - 80, y], fill=(210, 210, 210), width=1)
        y += 26
    # footer / signature
    d.text((80, _H - 160), "Authorised Signatory", fill=(30, 30, 30))
    d.rectangle([_W - 360, _H - 200, _W - 80, _H - 120],
                outline=(150, 150, 150), width=2)
    return img


def ensure_seed_assets() -> None:
    created = 0
    for dtype, titles in _TYPE_TITLES.items():
        tdir = config.TEMPLATES_DIR / dtype
        tdir.mkdir(parents=True, exist_ok=True)
        existing = list(tdir.glob("*.png"))
        if existing:
            continue
        for i, title in enumerate(titles):
            img = _make_template(title, seed=hash((dtype, i)) & 0xFFFF)
            img.save(tdir / f"tmpl_{i + 1}.png")
            created += 1
    if created:
        logger.info("Seeded %d placeholder template images", created)

    manifest = config.ADVERSARIAL_DIR / "patterns.json"
    if not manifest.exists():
        manifest.write_text(json.dumps([
            {"name": "jpeg_double_compression", "param": 60},
            {"name": "rotation", "param": 1.5},
            {"name": "color_channel_shift", "param": 3},
            {"name": "gaussian_blur", "param": 1.2},
            {"name": "brightness_shift", "param": 1.15},
        ], indent=2))
        logger.info("Seeded adversarial pattern manifest")


if __name__ == "__main__":
    ensure_seed_assets()
    print("seed assets ready")
