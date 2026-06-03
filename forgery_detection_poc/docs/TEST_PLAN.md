# Test Plan — AI Document Forgery Detection POC (PR #1)

App under test: FastAPI POC at `http://127.0.0.1:8000` (served from `forgery_detection_poc/`).
Evidence from code: UI flow in `static/index.html` (fileInput L106, candidateId L102, per-file doc-type `<select>` L162-164, Analyze L110/170, `/analyze` fetch L193); backend `/analyze` in `main.py` L44-67; verdict/agents via `pipeline/orchestrator.py`. Test docs: `dry_run_output/testA_clean_payslip.pdf` (clean) and `dry_run_output/testB_forged_payslip.pdf` (designation altered "Software Engineer"→"Senior Architect", producer=iLovePDF, modDate after creation).

Note on environment: external LLM/Azure/heavy-model paths are not provisioned, so cross-doc reasoning uses the documented rule-based fallback. This does NOT affect the assertions below — all are produced by the always-on agents + rule-based cross-doc.

## Primary flow (single recording): clean + forged pair

Setup state already done: server running, browser open, window maximized.

1. Go to `http://127.0.0.1:8000`. Candidate ID field pre-filled `CAND-DEMO` (leave as-is).
   - PASS: Upload panel "1 · Upload candidate documents" visible with a file picker and "Analyze documents" button.
2. Upload BOTH files via the file input: `testA_clean_payslip.pdf` and `testB_forged_payslip.pdf`.
   - PASS: Two file rows appear, each with a doc-type dropdown. Set/confirm both dropdowns to `payslip`.
3. Click "Analyze documents". Wait for spinner to finish.
   - PASS: Results section appears (no `alert()` error dialog).

### Assertions (each would differ if the change were broken)

A. **Overall verdict banner** — element `#banner` / `#overallBand` / `#overallScore`.
   - EXPECT: band text **`P0`**, banner styled red (class `P0`), score **`100%`**.
   - BROKEN-LOOKS-LIKE: P2/green or low score → fraud not detected.

B. **Per-document discrimination** — cards in `#docs`.
   - EXPECT clean `testA_clean_payslip.pdf`: pill **`P2`** (green), score ~**0%**.
   - EXPECT forged `testB_forged_payslip.pdf`: pill **`P0`** (red), score **100%**.
   - BROKEN-LOOKS-LIKE: both same band → no discrimination between clean and forged.

C. **Forged-region overlay** — canvas viewer inside the forged doc card.
   - EXPECT: at least one **red/amber bounding box** drawn over the "Senior Architect" designation line. Hover tooltip names a firing agent (e.g. font/layout or metadata) with a reason.
   - EXPECT clean doc viewer: **no red/amber fraud boxes** (clean).
   - BROKEN-LOOKS-LIKE: no box on forged field, or boxes on the clean doc.

D. **Flagged agents on forged doc** — forged card detail/findings list.
   - EXPECT: includes Metadata (Agent 1), Font & Layout (Agent 3), Duplicate (Agent 5) among flagged. (SHAP bar panel shows these as top drivers.)
   - BROKEN-LOOKS-LIKE: zero agents flagged on the forged doc.

E. **Cross-document contradiction** — table in `#contradictions`.
   - EXPECT: at least one row, type **`designation_mismatch`**, with quotes **`Software Engineer`** vs **`Senior Architect`** across the two docs.
   - BROKEN-LOOKS-LIKE: empty contradictions table.

F. (Light, optional) **Export verdict JSON** button — click `#exportBtn`.
   - EXPECT: a `verdict_CAND-DEMO.json` download is triggered. (Informational; not core.)

## Pass criteria
A–E all pass. C (visible overlay on the forged field) and E (designation_mismatch) are the headline proofs a reviewer needs. Any failure in A–E is reported as a failure, not glossed over.
