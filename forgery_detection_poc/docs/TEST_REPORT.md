# Test Report — AI Document Forgery Detection POC (PR #1)

**How tested:** Launched the FastAPI app locally (`uvicorn main:app`, `http://127.0.0.1:8000`) and drove the real HTML UI end-to-end — uploaded a clean payslip and a forged payslip as one candidate, clicked **Analyze documents**, and inspected the rendered verdict (banner, per-doc cards, canvas bbox overlay, cross-doc table) plus the underlying verdict JSON via the browser.

Documents: `dry_run_output/testA_clean_payslip.pdf` (clean) and `dry_run_output/testB_forged_payslip.pdf` (same payslip with the Designation field altered "Software Engineer"→"Senior Architect" in a different font, plus edited PDF metadata producer=iLovePDF).

## Escalations / things to know

- **Stateful dedup persists across runs (by design) and initially produced a false P0 on the clean doc.** On my first attempt, the clean doc A came back **P0 / 100%** with *zero* flagged regions and *no* agent findings. Root cause (confirmed in the verdict JSON): Agent 5 fired `score=1.0` with detail *"exact SHA-256 duplicate of 'testA_clean_payslip.pdf'"* — the file's hash was already in the persistent registries (`uploads/processed_hashes.json`, `faiss_index/dedup_registry.json`) from the earlier dry run / browser session. The pipeline treats re-submission of a byte-identical file as an exact-duplicate fraud signal. After clearing those gitignored runtime registries (the same reset `dry_run.py._reset_state()` performs) and re-running, the clean doc correctly scored **P2 / 10%**. This is environment/test-state contamination, **not** a detection-logic defect — but it is worth flagging: anyone re-testing with the same files on a dirty state will see the clean doc flagged as a duplicate. Consider documenting this, or auto-resetting state in a demo/test mode.
- **Cross-doc reasoning ran via the rule-based fallback** (`cross-doc engine: rule_based_fallback`), because no OpenAI/Anthropic key is provisioned — this is the documented graceful-degradation path, and it still produced the expected `designation_mismatch`. The LLM path was not exercised.
- Heavy DL models / Azure DI / PaddleOCR were not provisioned (documented in `SETUP.md`); those agent paths used fallbacks and were not exercised.
- **v1.1 infra exercised on the `local` backend.** Agent 4 ran through the `TemplateStore` abstraction and Step 3 OOD through the persisted `TemplateEmbeddingIndex` (FAISS, seeded from `templates/` on first run); both worked with the shipped placeholder templates and did not alter the verdicts below. The `azure_blob` backend and the `scripts/index_templates.py` / `scripts/finetune_agent9.py` operator tools are present but not run here (they require an authentic corpus / Blob credentials) — see `docs/COVERAGE_MATRIX.md` §3d.

## Results (clean-state run)

Overall verdict: **P0 / 100%**; Summary: `2 documents analyzed · 1 cross-document contradiction · bands: P0=1, P1=0, P2=1`.

- **Clean doc → P2, no overlays** — passed. `testA_clean_payslip.pdf`: band **P2**, score **10%**, **0 flagged regions**, "Flagged findings: None", all agent contributions 0.000.
- **Forged doc → P0 with bbox** — passed. `testB_forged_payslip.pdf`: band **P0**, score **100%**, **1 flagged region**; amber bounding box drawn over the altered **"Senior Architect"** designation; Font & Layout finding: *"font 'times' differs from dominant 'helvetica'"* (conf 0.65).
- **Clean vs forged discrimination** — passed. Same template, one byte-level edit → P2 vs P0 (the discriminating signal a broken pipeline would miss).
- **Cross-document contradiction** — passed. Table shows type **`designation_mismatch`**, docs `testA…` / `testB…`, Quote A **"software engineer"** vs Quote B **"senior architect"**, conf **0.7**.
- **Verdict JSON integrity** — passed. Live `lastVerdict` matched the rendered UI (doc bands/scores, agent_5 detail, contradiction array).

## Evidence

| Overview — P0 100%, A=P2 / B=P0 | Forged doc — amber bbox on "Senior Architect" | Cross-doc — designation_mismatch |
|---|---|---|
| ![overview](01_overview_P0.png) | ![forged](02_forged_bbox.png) | ![crossdoc](03_cross_doc.png) |

Devin session: https://cognizant.devinenterprise.com/sessions/a4013f43b584436aba2666a4411c58ff
