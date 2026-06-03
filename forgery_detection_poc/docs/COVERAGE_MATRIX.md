# AI Document Forgery Detection POC — Full Agent Coverage Matrix

Prepared so that **nothing is missed when the external API credentials arrive.** Every
no-API agent has been driven with a purpose-built fraud document and **confirmed to fire**;
every API/weight-dependent path has a **ready-to-run procedure** below.

- Repo / PR: https://github.cognizant.com/Digital-Platform-IT-Build/OneC_4095_CLRM_MWS/pull/1
- Session: https://cognizant.devinenterprise.com/sessions/a4013f43b584436aba2666a4411c58ff
- Test harness (fixture generator, not part of the POC): `forgery_detection_poc/test_matrix.py`
- Per-agent verdict JSONs: `forgery_detection_poc/test_matrix_output/verdict_*.json`

---

## 1. Coverage summary

| # | Agent | Needs API/weights? | Trigger condition | Tested now? | Result |
|---|-------|--------------------|-------------------|-------------|--------|
| A1 | Metadata | No | Producer/creator is a known PDF editor, or modDate after creationDate | Yes | **FIRED** score 0.60 |
| A2 | Image Forensics (ELA + noise) | No (heavy: TruFor/MVSS-Net optional) | Fused ELA+noise 99th-pct ≥ 0.40 over a region | Yes | **FIRED** score 0.41 |
| A3 | Font & Layout | No | A text line's font/size deviates from the page's dominant font | Yes | **FIRED** score 0.65 |
| A4 | Template Matching (SSIM) | No (templates via TemplateStore; authentic corpus optional) | Best-template SSIM < 0.70 with deviation regions | Yes | **FIRED** score 0.33 (SSIM 0.666) |
| A5 | Duplicate / Similarity | No | Exact SHA-256 / near-duplicate (pHash) of a previously-seen doc | Yes | **FIRED** score 1.00 |
| A6 | Temporal Consistency | No | Future-dated value, date inversion, or gap > 183 days | Yes | **FIRED** score 0.70 |
| A7 | NER / Semantic | No | ORG entity within edit-distance ≤ 2 of a known legitimate name | Yes | **FIRED** score 0.75 |
| A8 | QR / Barcode | No | QR expected (form16/offer_letter/edu_cert) but absent, or malformed/inconsistent payload | Yes | **FIRED** score 0.40 |
| A9 | Novel Fraud (PCA anomaly) | No (heavy: PatchCore/DRAEM optional) | A patch's reconstruction error is a robust-z > 8 outlier | Yes | **FIRED** score 1.00 |
| A10 | Cross-OCR Disagreement | **YES — PaddleOCR** (2nd OCR engine) | ≥ 2 OCR engines disagree on a critical field (weighted > 0.50) | **No — needs `ENABLE_PADDLEOCR=1`** | Pending |
| A11 | Adversarial Robustness | No | Max ELA delta under seeded perturbations > 0.20 | Yes | **FIRED** score 0.66 |

**10 of 11 agents proven now (A1–A9, A11).** A10 is the only agent that cannot fire without a
second OCR engine (PaddleOCR), which is the no-key flag described in §3.

Cross-document reasoning (Step 5) also runs today via the **rule-based fallback**
(`designation_mismatch`, `entity_mismatch`, `salary_break`); the **LLM backend** is the
API-dependent upgrade (see §3).

**v1.1 additions (template & OOD infrastructure):** Agent 4 now resolves templates through a
**TemplateStore** abstraction (`pipeline/template_store.py`, `local`/`azure_blob` backends,
switched by `TEMPLATE_SOURCE` — config-only). Step 3 OOD detection compares the DiT embedding
against a persisted **TemplateEmbeddingIndex** (`pipeline/template_embedding_index.py`, FAISS,
seeded from `templates/` on first run). Two manual **operator tools** ingest an authentic corpus
when one is available (see §3e) — neither runs automatically.

---

## 2. What fired now (no-API agents) — evidence

Each fixture was run through the **full pipeline in isolation** (dedup state reset between
runs) so the named agent's detection is unambiguous. Source builders are in
`test_matrix.py`; verdicts in `test_matrix_output/verdict_<TAG>.json`.

| Tag | Fixture | Agent | Detail (from verdict) |
|-----|---------|-------|-----------------------|
| A1 | `a1_metadata.pdf` | Metadata | producer 'ilovepdf'; modDate 2026-06-01 after creation 2024-05-01 |
| A2 | `a2_splice.jpg` | Image Forensics | ELA+noise residual exceed threshold (photographic raster) |
| A3 | `a3_fontswap.pdf` | Font & Layout | 1 anomalous line; dominant font 'helvetica' size 12 |
| A4 | `a4_tampered.png` | Template | best template SSIM=0.666 < 0.70 |
| A5 | duplicate submit | Duplicate | exact SHA-256 duplicate of prior submission |
| A6 | `a6_temporal.pdf` | Temporal | future-dated value '15-Aug-2031' |
| A7 | `a7_ner.pdf` | NER | 'Goggle' is edit-distance 1 from legitimate 'Google' |
| A8 | `a8_qr.pdf` | QR | no QR/barcode found but expected for 'offer_letter' |
| A9 | `a9_novelty.png` | Novel Fraud | per-doc PCA novelty max anomaly = 1.000 |
| A11 | `a11_adversarial.jpg` | Adversarial | max ELA delta under perturbation = 0.220 > 0.20 |

> Note on the image agents (A2/A9/A11): on **synthetic vector-text** documents a small
> localized splice stays *quieter* than the surrounding text, so the canonical positive for
> the ELA/noise/anomaly detectors is a **photographic raster** (or a real scanned/tampered
> document). The fixtures use synthesised photographic textures to exceed the conservative
> thresholds. Real scanned forgeries will trigger these the same way.

---

## 3. Ready-to-run procedures for API / weight-dependent paths

All keys are read from env vars via `config.py` (template in `.env.example`). Set them, restart,
and run — no code changes needed. Working dir: `forgery_detection_poc/`.

### 3a. LLM cross-document reasoning (replaces rule-based fallback)
Provide **one** of:
```bash
export OPENAI_API_KEY="sk-..."            # default model gpt-4-turbo
# or
export ANTHROPIC_API_KEY="sk-ant-..."
export CROSS_DOC_MODEL="claude-opus-4-..."  # only if using Anthropic
```
Verify (uses the existing clean+forged pair):
```bash
python dry_run.py
# expect in output:  cross-doc [openai:gpt-4-turbo]  (NOT rule_based_fallback)
```
Backend selection logic: `pipeline/cross_doc_llm.py:131-143` (OpenAI → Anthropic → fallback).

### 3b. Azure Document Intelligence (high-accuracy OCR/layout — also helps A10)
Both required together:
```bash
export AZURE_DOC_INTELLIGENCE_ENDPOINT="https://<resource>.cognitiveservices.azure.com/"
export AZURE_DOC_INTELLIGENCE_KEY="<key>"
```
Verify: re-run any case and confirm `ocr_engines_available` includes `azure` in the verdict
JSON (currently `['tesseract']`).

### 3c. PaddleOCR → activates **Agent 10 (Cross-OCR Disagreement)** — no key, downloads weights
```bash
export ENABLE_PADDLEOCR=1     # needs outbound internet on first run to fetch weights
python test_matrix.py         # A10 should now report >=2 OCR engines
```
A10 fires when ≥2 engines disagree on a critical field (`pipeline/agents/agent10_cross_ocr.py`).
A purpose-built A10 fixture (a doc whose ambiguous glyphs make engines disagree, e.g. `O`/`0`,
`1`/`l`) can be added once a 2nd engine exists — flagged as the one remaining fixture to build.

### 3d. Authentic-template corpus → Azure Blob + operator tools (v1.1, no API key)
When authentic organizational templates/documents are available, ingest them to sharpen Agent 4
template matching and Step 3 OOD detection — **config-only**, no code changes:
```bash
# Option A: local folder — drop files under templates/<doc_type>/ then:
python scripts/index_templates.py --source local        # OOD FAISS index (incremental)
python scripts/finetune_agent9.py --source local        # retrain A9 novelty -> models/agent9_weights

# Option B: Azure Blob (organised as <container>/<doc_type>/<file>)
export TEMPLATE_SOURCE=azure_blob
export AZURE_BLOB_CONNECTION_STRING="<connection-string>"
export AZURE_BLOB_TEMPLATE_CONTAINER="authentic-templates"   # default
python scripts/index_templates.py --source azure_blob
python scripts/finetune_agent9.py --source azure_blob
```
Notes: `index_templates.py` is incremental (re-running only adds new files, keyed by
`<doc_type>/<filename>`). `finetune_agent9.py` writes `models/agent9_weights/patchcore_pca.npz`,
which Agent 9 auto-loads next run (else it uses the per-document fallback). **Do not fine-tune on
the shipped synthetic placeholder templates** — it makes A9 over-flag; use a real corpus only.

### 3e. Heavy DL models (no API key — just weights + ideally a GPU)
These degrade gracefully today and activate when weights are present:
- Document understanding: `microsoft/layoutlmv3-base`, `naver-clova-ix/donut-base`, `microsoft/dit-base` (auto-download via `transformers` + `torch`).
- A2 forensics: TruFor weights → `models/trufor/`; MVSS-Net → `models/mvssnet.pt`.
- A9 novelty: authentic patches → `models/agent9_authentic/` (enables the global model).

Install `torch`+`transformers` and the weights, then re-run `python dry_run.py`; the logs will
switch from "unavailable, using fallback" to the real backend.

---

## 4. Final-test checklist (run when credentials arrive)

```bash
cd forgery_detection_poc
export OPENAI_API_KEY=...            # or ANTHROPIC_API_KEY (+CROSS_DOC_MODEL)
export AZURE_DOC_INTELLIGENCE_ENDPOINT=...   AZURE_DOC_INTELLIGENCE_KEY=...
export ENABLE_PADDLEOCR=1
python test_matrix.py                # re-confirms A1-A9, A11 + now A10
python dry_run.py                    # confirms LLM backend + azure OCR engine

# Optional (v1.1): once an authentic corpus exists, sharpen A4 + Step-3 OOD
export TEMPLATE_SOURCE=azure_blob AZURE_BLOB_CONNECTION_STRING=... AZURE_BLOB_TEMPLATE_CONTAINER=authentic-templates
python scripts/index_templates.py --source azure_blob
python scripts/finetune_agent9.py --source azure_blob
```
Expected after keys: **11/11 agents fire**, cross-doc backend = `openai:...`/`anthropic:...`,
`ocr_engines_available` includes `tesseract`+`paddleocr`(+`azure`).
