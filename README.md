

## AI Document Forgery Detection POC

An end-to-end proof-of-concept that ingests candidate background-verification
documents (PDF / DOCX / images), runs a multi-agent forensic pipeline, and
produces a structured fraud verdict (P0/P1/P2) with suspicious regions
highlighted in an HTML UI.

Project lives under [`forgery_detection_poc/`](forgery_detection_poc/).

### Pipeline (8 steps)
1. **Intake** — upload, validate, SHA-256 hash, metadata.
2. **Pre-processing & triple OCR** — PyMuPDF/Pillow normalisation; Azure DI +
   PaddleOCR + Tesseract in parallel with a per-field disagreement vector.
3. **Document understanding** — LayoutLMv3 / Donut / DiT; DiT embedding compared
   against a FAISS **TemplateEmbeddingIndex** for out-of-distribution (OOD) detection.
4. **Specialist agent ensemble** — 11 mandatory agents run in parallel
   (`ThreadPoolExecutor`); each returns scored, bbox-tagged findings.
5. **Cross-document LLM reasoning** — inter-document contradictions; backend
   preference Azure OpenAI (Azure AI Foundry) → OpenAI → Anthropic → rule-based fallback.
6. **Meta-learner** — weighted ensemble + SHAP attribution + band assignment.
7. **Verdict & routing** — escalation rules → final verdict JSON.
8. **HTML UI** — FastAPI `/analyze`; canvas overlays with colour-coded fraud boxes.

### v1.1 additions (template & OOD infrastructure)
- **TemplateStore** (`pipeline/template_store.py`) — switchable template backend
  (`local` / `azure_blob`) behind `get_templates(doc_type)`; Agent 4 never touches
  paths directly. Switching to Azure Blob is **config-only** (`TEMPLATE_SOURCE`).
- **TemplateEmbeddingIndex** (`pipeline/template_embedding_index.py`) — persisted
  FAISS OOD index (load / seed / upsert / query), seeded from `templates/` on first run.
- **Operator tools** (manual, not auto-run):
  - `scripts/index_templates.py --source local|azure_blob` — ingest authentic
    templates into the OOD FAISS index (incremental).
  - `scripts/finetune_agent9.py --source local|azure_blob` — retrain the Agent 9
    novelty detector on an authentic corpus.

### Documentation
- Setup & run guide: [`forgery_detection_poc/SETUP.md`](forgery_detection_poc/SETUP.md)
- Agent coverage matrix & ready-to-run procedures: [`forgery_detection_poc/docs/COVERAGE_MATRIX.md`](forgery_detection_poc/docs/COVERAGE_MATRIX.md)
- UI test plan: [`forgery_detection_poc/docs/TEST_PLAN.md`](forgery_detection_poc/docs/TEST_PLAN.md)
- UI test report: [`forgery_detection_poc/docs/TEST_REPORT.md`](forgery_detection_poc/docs/TEST_REPORT.md)

### Quick start
```bash
cd forgery_detection_poc
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn main:app --host 0.0.0.0 --port 8000   # then open http://localhost:8000
```
All external keys/thresholds live in `config.py` (env via `.env.example`). Agents
degrade gracefully when a model/key is missing, so the pipeline runs out of the box.
