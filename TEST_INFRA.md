# CeramScan E2E Test Infrastructure

Verification architecture, environment inventory, fixture contracts, and quality gates for the opaque-box 4-tier end-to-end test suite (`tests/e2e/`) covering CeramScan («КерамикСкан») features `F1` through `F15`.

## 1. Verified Environment & Binary Matrix

All Python test runners and application dependencies execute inside the `uv`-managed `.venv` (`Python 3.12.15`, 43 pinned distributions in `requirements.txt`). Note that `.venv` omits the `pip` module by design (`uv` management), so environment completeness is checked directly via `importlib.metadata` against `requirements.txt`.

| Component | Path / Command | Verified Version / Probe | Purpose |
|---|---|---|---|
| Python runtime | `.venv/bin/python` | `Python 3.12.15` (`uv`) | Primary app & test interpreter |
| Pytest runner | `.venv/bin/pytest` | `pytest 9.2.0` | Test discovery and execution |
| HTTP & browser libs | `httpx`, `playwright` | `0.28.1`, `1.65.2` | Live ASGI HTTP & Chrome automation |
| CV & ML runtime | `opencv-python`, `numpy`, `scikit-image`, `onnxruntime`, `protobuf` | `5.0.0.93`, `2.5.4`, `0.26.0`, `1.32.0`, `7.36.2` | CPU CV core, ONNX weights, protobuf |
| Office & PDF libs | `python-docx`, `python-pptx`, `pypdf`, `qrcode`, `pillow` | `1.2.0`, `1.0.2`, `7.1.1`, `8.2`, `13.0.0` | Grant artifact & QR verification |
| LibreOffice CLI | `/opt/homebrew/bin/soffice` | `LibreOffice 26.8.3.1` | Headless `.docx`/`.pptx` PDF export |
| Poppler PDF renderer | `/opt/homebrew/bin/pdftoppm` | `pdftoppm 26.11.0` | 150 DPI page/slide PNG rendering |
| Google Chrome | `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` | `157.0.8103.79` | Headless UI checks via `channel="chrome"` |
| Anti-AI slop scanner | `/Users/alex/analytics_env/bin/python .../slop_scan.py` | `Python 3.15.1` runner | Prose, web UI, and Python lint gate |

## 2. Four-Tier Test Methodology & Marker Taxonomy

Custom markers are registered programmatically inside `tests/e2e/conftest.py` (`pytest_configure`) so `.venv/bin/pytest --strict-markers` operates cleanly without modifying root config files. Every test carries one methodology tier marker (`tier1`..`tier4`) and one or more feature markers (`f1`..`f15`) matching `PROJECT.md`.

| Marker | Design Methodology Tier | Target Minimum | Planned Scope (`F1`..`F15`) |
|---|---|---|---|
| `tier1` | Feature Coverage (happy-path equivalence classes per feature) | `>= 75` (`>= 5` / feature) | Core contracts for `F1`..`F15` |
| `tier2` | Boundary & Corner Cases (negative inputs, error codes, limits) | `>= 75` (`>= 5` / feature) | Invalid payloads, security, constraints |
| `tier3` | Cross-Feature Pairwise Combinations (multi-module contracts) | `>= 16` | Pipeline, API+DB, UI+API, grant parity |
| `tier4` | Real-World Application Scenarios (end-to-end operator flows) | `>= 8` | Shift QC, `ceramic_defect_1..4.jpg`, 289-FZ, mobile/desktop journeys |
| Total | `tests/e2e/` collected items (`--collect-only --strict-markers`) | `>= 173` | Pipeline (`test_e2e_pipeline.py`), API/UI (`test_e2e_api_and_ui.py`), Grant/Repo (`test_e2e_grant_and_repo.py`) |

Feature markers `f1` through `f15` align 1-to-1 with `PROJECT.md`:
1. `f1` - Python 3.12 `.venv`, pinned `requirements.txt`, `.gitignore` hygiene
2. `f2` - Foundation documentation integrity (`PROJECT.md`, `ARCHITECTURE.md`, `DESIGN_SPEC.md`)
3. `f3` - `src/cv_core/preprocessing.py` ROI, perspective warp, CLAHE, glare & cobalt masks
4. `f4` - `src/cv_core/defect_detector.py` GOST 28390-89 triage, `< 300 ms` CPU budget, `--weights` ONNX
5. `f5` - `src/cv_core/fingerprint.py` 289-FZ `ORB + pHash + cobalt` authenticity verification
6. `f6` - `src/cv_core/storage.py` SQLite WAL DDL, foreign keys, `CERAMSCAN:<item_id>` QR PNGs
7. `f7` - `scripts/generate_synthetic_ceramic.py` synthetic validation stand v0.1 (`>= 24` samples)
8. `f8` - `scripts/run_benchmark.py` real CPU latency, Precision/Recall/F1, PNG overlays
9. `f9` - `scripts/ingest_real_photos.py` folder mapping, YOLO export, `--rebuild-all`
10. `f10` - `src/api.py` FastAPI HTTP endpoints, upload limits, error codes, static mounts
11. `f11` - `src/web/` Zero-Build UI, self-hosted `.woff2` fonts, `375x812` & `1440x900` responsiveness
12. `f12` - `grant_case/project_description.md` & `CeramScan_Sistema_2026.docx` (`<= 15` pages)
13. `f13` - `grant_case/build_presentation.py` & `CeramScan_Sistema_2026.pptx` (`10` slides `16:9`)
14. `f14` - `pitch_and_video_script.md`, `pilot_letter_template.md`, `REFINEMENTS_ROADMAP.md`
15. `f15` - Public portfolio `README.md`, `.gitignore`, and full `slop_scan.py` clearance

## 3. Lazy-Import & Fail-Loud Policy

Top-level imports in `tests/e2e/conftest.py` and `tests/e2e/test_*.py` stay strictly restricted to Python stdlib and third-party test packages (`pytest`, `httpx`, `cv2`, `numpy`, `PIL`, `pypdf`, `playwright.sync_api`). Application imports (`src.*`, `scripts.*`, `grant_case.*`) happen lazily inside test function bodies or runtime fixtures.

This guarantees two properties:
1. `pytest tests/e2e/ --collect-only --strict-markers` succeeds cleanly even before `M2`, `M3`, or `M4` targets are built on disk.
2. Tests never call `pytest.skip(...)` or `pytest.importorskip(...)` on unbuilt deliverables; running a test against missing or broken code fails loudly with `ModuleNotFoundError`, `FileNotFoundError`, or `AssertionError`.

## 4. Locked Fixture & Isolation Catalog (`tests/e2e/conftest.py`)

`tests/e2e/conftest.py` prepends `PROJECT_ROOT` to `sys.path` and exposes isolated, mock-free fixtures for all three E2E test modules.

| Fixture | Scope | Purpose & Isolation Contract |
|---|---|---|
| `repo_root` | `session` | Resolves `PROJECT_ROOT` (`ceramscan-sistema`) |
| `venv_python` | `session` | Path to `.venv/bin/python`, verifying file existence |
| `slop_scanner_cmd` | `session` | Command list `[analytics_python, slop_scan.py]` |
| `reference_defect_images` | `session` | Maps `defect_1..defect_4` to `data/reference_materials/ceramic_defect_samples/ceramic_defect_1..4.jpg` |
| `ceramic_image_factory` | `session` | Deterministic BGR/PNG plate generator (`bisque`, `glazed`, `cobalt`, `crack`, `chip`, `warp`, `glare`, `glaze_bare`, `pinhole_count`, `drip`, `quad_frame`, counterfeit plates, `16x16` tiny PNG) |
| `invalid_image_payloads` | `function` | Returns `empty`, `corrupt`, `too_small_png`, and `oversized_over_10mb` byte payloads |
| `onnx_weights_factory` | `function` | Pure-Python protobuf `ModelProto` builder verified against `onnxruntime.InferenceSession` (`yolov8`, `yolov8_clean`, `yolov5`, `bad_rank`, `corrupt`, `non_onnx`, `missing`) |
| `isolated_workspace` | `function` | Temporary directories for DB, reports, overlays, QR codes, passports, synthetic assets, real photos, and YOLO export |
| `isolated_repo_workspace` | `function` | Copies `src/`, `scripts/`, `data/reference_materials/`, and `grant_case/` (if present) into `tmp_path` for isolated `--rebuild-all` CLI runs |
| `real_photos_dir_factory` | `function` | Builds `pass/chip/crack/glaze/fake` directory trees (`populate=False` or `populate=True` with `bisque_plate_01.jpg` and `util_cup_02.jpg`) |
| `api_server_factory` | `session` | Spawns isolated `uvicorn src.api:app` on a free localhost TCP port with `CERAMSCAN_DB_PATH`, `CERAMSCAN_OVERLAY_DIR`, `CERAMSCAN_QR_DIR`, and `CERAMSCAN_PASSPORT_DIR` inside `tmp_path`, polls `/health` up to 15s, exposes `.stop(base_url)`, and reaps remaining servers at session end |
| `live_api_url` | `module` | Module-scoped isolated FastAPI base URL for read-only and UI tests, terminated and reaped on module teardown |
| `isolated_api_server` | `function` | Function-scoped `(base_url, paths_dict)` with a fresh SQLite WAL database; terminates and reaps its `uvicorn` subprocess right after each test yields |
| `api_client` | `function` | `httpx.Client` bound to `isolated_api_server` (`timeout=20.0`) |
| `playwright_browser` | `module` | Headless Chrome instance via `playwright.sync_api` (`channel="chrome", headless=True`) |
| `ui_page` | `function` | Fresh Playwright context/page capturing `console_errors`, `page_errors`, `failed_requests`, and `http_errors` |

## 5. Opaque-Box Storage & QA Gate Rules

1. Storage verification (`F6`) checks SQLite WAL pragmas, exact DDL schemas, primary keys, `CHECK` constraints, `PRAGMA foreign_keys=ON`, and `ON DELETE CASCADE` directly via `CeramStorage(...)` plus `sqlite3`, and verifies end-to-end DB rows, `cv2.QRCodeDetector` QR PNG decoding, passport snapshots, and `item_id` traversal guards via `isolated_api_server` HTTP endpoints.
2. Performance and boundary checks enforce `< 300 ms/frame` CPU inference latency, strict HTTP error statuses (`400`, `404`, `409`, `413`, `422`), and `320px`/`375x812`/`1440x900` Playwright Chrome layout gates (`0` console errors, `0` failed requests, `scrollWidth <= innerWidth`, `>= 44px` mobile tap targets).
3. Document QA checks run LibreOffice `soffice --headless --convert-to pdf`, `pypdf.PdfReader`, and `pdftoppm -png -r 150` (`<= 15` `.docx` pages, `== 10` `.pptx` slides).
4. Hygiene checks run `slop_scan.py` on all deliverables and test files with zero U+2014 characters, zero broad `except:` clauses, zero mutable default arguments, and zero literal forbidden project names.

## 6. Execution Commands

Collect full E2E suite without skipping unbuilt modules:

```bash
.venv/bin/pytest tests/e2e/ --collect-only --strict-markers
```

Run pipeline tests (`F1`..`F9`) or milestone subsets:

```bash
.venv/bin/pytest tests/e2e/test_e2e_pipeline.py -k "f1 or f2" -m "tier1 or tier2"
.venv/bin/pytest tests/e2e/test_e2e_pipeline.py --strict-markers
```

Run by design methodology tier:

```bash
.venv/bin/pytest tests/e2e/ -m tier1 --strict-markers
.venv/bin/pytest tests/e2e/ -m tier2 --strict-markers
.venv/bin/pytest tests/e2e/ -m tier3 --strict-markers
.venv/bin/pytest tests/e2e/ -m tier4 --strict-markers
```

Run full E2E suite and anti-slop audit:

```bash
.venv/bin/pytest tests/e2e/ --strict-markers
/Users/alex/analytics_env/bin/python /Users/alex/.gemini/config/skills/anti-ai-slop-architect/scripts/slop_scan.py TEST_INFRA.md tests/e2e/
```
