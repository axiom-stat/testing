# E2E Test Suite Ready

The opaque-box 4-tier end-to-end test suite (`tests/e2e/`) for CeramScan («КерамикСкан») has passed Cycle 2 review, empirical adversarial challenge, and forensic audit, and is cleared for milestone and final verification gates.

## Test Runner

Run the full E2E suite serially (recommended to avoid host CPU jitter on `< 300 ms/frame` latency assertions):

```bash
.venv/bin/pytest tests/e2e/ --strict-markers -v
```

Run sequential per-tier verification for Final Milestone Phase 1:

```bash
.venv/bin/pytest tests/e2e/ -m tier1 --strict-markers -v
.venv/bin/pytest tests/e2e/ -m tier2 --strict-markers -v
.venv/bin/pytest tests/e2e/ -m tier3 --strict-markers -v
.venv/bin/pytest tests/e2e/ -m tier4 --strict-markers -v
```

Expected outcome once all M1 through M4 targets are built: all `220` collected tests pass with exit code `0`, `0` skips, and `0` mocks. Unbuilt targets fail loudly with `ModuleNotFoundError`, `FileNotFoundError`, or `RuntimeError`.

## Coverage Summary

Verified live via `.venv/bin/pytest tests/e2e/ --collect-only --strict-markers`:

| Metric / Module | Collected Items | Distinct Functions | Target / Scope |
|---|---|---|---|
| `tier1` (Feature Coverage) | `86` | `85` | `>= 75` (`>= 5` per feature `F1`..`F15`) |
| `tier2` (Boundary & Corner Cases) | `101` | `90` | `>= 75` (`>= 5` per feature `F1`..`F15`) |
| `tier3` (Cross-Feature Pairwise) | `22` | `22` | `>= 16` multi-module contracts |
| `tier4` (Real-World Scenarios) | `11` | `11` | `>= 8` end-to-end operator journeys |
| `tests/e2e/test_e2e_pipeline.py` | `126` | `114` | Pipeline features `F1`..`F9` |
| `tests/e2e/test_e2e_api_and_ui.py` | `38` | `38` | FastAPI & Web UI features `F10`..`F11` |
| `tests/e2e/test_e2e_grant_and_repo.py` | `56` | `56` | Grant package & repo features `F12`..`F15` |
| Total (`tests/e2e/`) | `220` | `208` | `>= 173` total collected items |

## Feature Checklist

Per-feature counts collected via `.venv/bin/pytest tests/e2e/ --collect-only --strict-markers -q -m "<feature> and <tier>"` (note that `tier3` and `tier4` tests carry multiple feature markers when verifying cross-module flows):

| Feature | Scope | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Total Tests |
|---|---|---|---|---|---|---|
| `F1` | Python 3.12 `.venv`, `requirements.txt`, `.gitignore` | `5` | `5` | `✓` (`1`) | `-` (`0`) | `11` |
| `F2` | `PROJECT.md`, `ARCHITECTURE.md`, `DESIGN_SPEC.md` | `5` | `5` | `-` (`0`) | `-` (`0`) | `10` |
| `F3` | `src/cv_core/preprocessing.py` ROI, warp, LAB CLAHE, masks | `6` | `9` | `✓` (`3`) | `✓` (`6`) | `24` |
| `F4` | `src/cv_core/defect_detector.py` GOST triage, `< 300 ms`, ONNX | `7` | `10` | `✓` (`7`) | `✓` (`5`) | `29` |
| `F5` | `src/cv_core/fingerprint.py` 289-FZ `ORB + pHash + cobalt` | `5` | `6` | `✓` (`4`) | `✓` (`4`) | `19` |
| `F6` | `src/cv_core/storage.py` SQLite WAL, schema, QR PNGs | `5` | `11` | `✓` (`4`) | `✓` (`6`) | `26` |
| `F7` | `scripts/generate_synthetic_ceramic.py` stand v0.1 (`>= 24`) | `5` | `5` | `✓` (`3`) | `✓` (`2`) | `15` |
| `F8` | `scripts/run_benchmark.py` latency, Precision/Recall/F1, overlays | `5` | `5` | `✓` (`7`) | `✓` (`4`) | `21` |
| `F9` | `scripts/ingest_real_photos.py` mapping, YOLO export, `--rebuild-all` | `5` | `5` | `✓` (`4`) | `✓` (`1`) | `15` |
| `F10` | `src/api.py` FastAPI endpoints, upload guards, error codes | `7` | `9` | `✓` (`6`) | `✓` (`2`) | `24` |
| `F11` | `src/web/` Zero-Build UI, `.woff2` fonts, `375px` & `1440px` | `7` | `7` | `✓` (`3`) | `✓` (`3`) | `20` |
| `F12` | `grant_case/project_description.md` & `.docx` (`<= 15` pp.) | `6` | `6` | `✓` (`1`) | `✓` (`3`) | `16` |
| `F13` | `grant_case/build_presentation.py` & `.pptx` (`10` slides `16:9`) | `6` | `6` | `✓` (`2`) | `✓` (`3`) | `17` |
| `F14` | `pitch_and_video_script.md`, `pilot_letter_template.md`, roadmap | `6` | `6` | `✓` (`1`) | `✓` (`2`) | `15` |
| `F15` | Public portfolio `README.md`, `.gitignore`, `slop_scan.py` gate | `6` | `6` | `✓` (`1`) | `✓` (`2`) | `15` |

## Downstream Implementation Advisories

- M3 headless Chrome `/favicon.ico` requirement - Playwright `ui_page` captures HTTP `>= 400` responses and Chrome console `error` events; `src/web/index.html` should declare `<link rel="icon" href="data:,">` or an inline icon, or `src/api.py` should serve `/favicon.ico`, to avoid Chrome 404 console errors on page load.

## Gate Sign-Off

- Cycle 2 Review 1 (`teamwork_reviewer_r2_contracts`) - `APPROVE` (contracts, marker taxonomy, `conftest.py` fixture boundaries, lazy-import and opaque-box `F6` policy verified).
- Cycle 2 Review 2 (`teamwork_reviewer_r2_ui_grant`) - `APPROVE` (Playwright `320px`/`375x812`/`1440x900` UI checks, `F10`..`F15` grant/repo assertions, and `TEST_INFRA.md` accuracy verified).
- Cycle 2 Challenge 1 (`teamwork_challenger_r2_runtime`) - `APPROVE` (`CeramicImageFactory`, `onnx_weights_factory`, process reaping, isolated `--rebuild-all` workspace, and F1/F2/F3 execution verified).
- Cycle 2 Challenge 2 (`teamwork_challenger_r2_slop`) - `APPROVE` (strict & `--ignore-inline-code` `slop_scan.py` zero-finding check, LibreOffice/Poppler/Chrome stress probes, and write-ownership verified).
- Forensic Audit (`teamwork_auditor_e2e`) - `CLEAN` (`0` mocks, `0` skips, `0` vacuous tests, `220` collected items, `0` `slop_scan.py` indicators across `TEST_INFRA.md` and `tests/e2e/`).
