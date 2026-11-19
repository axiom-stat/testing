# Project: CeramScan («КерамикСкан»)

Autonomous two-stage optical quality control (OTK) system aligned with GOST 28390-89 and a 289-FZ anti-counterfeit digital passport engine for hand-painted Gzhel porcelain (`Объединение «Гжель»`), prepared as an individual submission by Pavlovets Alexander Ivanovich (Gatchina State University, 1st year, Economics of Enterprises and Organizations) for the 4th season of the Sistema Charitable Foundation scholarship competition (`lift-bf.ru`, deadline November 30, 2026).

## Architecture

CeramScan splits into four strict layers with unidirectional dependencies: a pure CPU computer vision and persistence core (`src/cv_core`), CLI dataset and benchmark tooling (`scripts/`), a FastAPI HTTP layer (`src/api.py`) serving a Zero-Build laboratory web workspace (`src/web/`), and reproducible grant artifact builders (`grant_case/`).

```text
[Smartphone Camera / File Upload / Synthetic Stand / Real Factory Photos]
                                   |
                                   v
                   src/cv_core/preprocessing.py
  (Calibration quad or contour ROI -> perspective warp -> 640x640 RGB/LAB
   -> masks M_item, M_rim, M_interior, M_spec, M_cobalt + LAB CLAHE)
            |                                           |
            v                                           v
  src/cv_core/defect_detector.py              src/cv_core/fingerprint.py
  (Deterministic OpenCV/skimage CV            (ORB-500 + 64-bit DCT pHash +
   or --weights ONNX/YOLO + GOST 28390-89      cobalt HSV/LAB histogram -> %
   triage: PASS / REWORK / REJECT + PNG)       match + 289-FZ verdict)
            |                                           |
            +---------------------+---------------------+
                                  |
                                  v
                       src/cv_core/storage.py
    (SQLite WAL: inspections, passports, passport_verifications + QR PNGs)
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
     src/api.py (FastAPI)                         scripts/ (CLI)
     -> src/web/ (Zero-Build UI)                  -> generate_synthetic_ceramic.py
        1. Пост ОТК (Смартфон)                    -> run_benchmark.py -> reports/
        2. Проверка подлинности (Ozon/WB)         -> ingest_real_photos.py
        3. Дашборд ОТК                               --rebuild-all -> grant_case/
```

Module boundaries remain strictly isolated:
- `src/cv_core/` accepts `numpy.ndarray` / `Path` inputs and never imports FastAPI, Starlette, or `python-pptx`
- `src/api.py` validates HTTP uploads (10 MB cap, magic/decode checks, regex-safe `item_id`), invokes `src/cv_core/`, and mounts static assets without embedding CV thresholds
- `src/web/` runs pure ES6 Vanilla JS, HTML5, and CSS3 with self-hosted `.woff2` fonts (`src/web/fonts/`), zero bundler steps, and zero external CDN runtime dependencies
- `scripts/` orchestrates synthetic stand generation, real-photo ingestion, CPU benchmarking, and `--rebuild-all` propagation into `reports/benchmark_results.json` and `grant_case/`

## Feature Inventory

| ID | Feature | Milestone | Requirement / Acceptance | Target Paths |
|---|---|---|---|---|
| F1 | Python 3.12 `.venv`, pinned `requirements.txt`, `.gitignore` | M1 | R2, Git hygiene AC | `.venv/`, `requirements.txt`, `.gitignore` |
| F2 | Foundation docs (`PROJECT.md`, `ARCHITECTURE.md`, `DESIGN_SPEC.md`) | M1 | R2, R4 | `PROJECT.md`, `ARCHITECTURE.md`, `DESIGN_SPEC.md` |
| F3 | ROI segmentation, perspective warp, LAB CLAHE, `bisque`/`glazed` glare & cobalt masks | M2 | R3 | `src/cv_core/preprocessing.py` |
| F4 | Hybrid CPU defect detector (`< 300 ms`), GOST 28390-89 triage, `--weights` ONNX/YOLO hook | M2 | R1, R3, CV AC | `src/cv_core/defect_detector.py` |
| F5 | 289-FZ hand-painted cobalt fingerprint (`ORB + pHash + cobalt histogram`) | M2 | R2, R3, CV AC | `src/cv_core/fingerprint.py` |
| F6 | SQLite WAL inspection & passport registry + QR generator (`qrcode`) | M2 | R3 | `src/cv_core/storage.py` |
| F7 | Synthetic validation stand v0.1 (`>= 24` labeled `bisque` & `glazed` samples) | M2 | R1, R3, CV AC | `scripts/generate_synthetic_ceramic.py`, `data/synthetic/` |
| F8 | Real CPU latency + `Precision/Recall/F1` + authenticity benchmark + overlays | M2 | R1, R3, CV AC | `scripts/run_benchmark.py`, `reports/` |
| F9 | Real photo ingestion (`pass/`, `chip/`, `crack/`, `glaze/`, `fake/`), YOLO export, `--rebuild-all` | M2 | R2, R3, CLI AC | `scripts/ingest_real_photos.py`, `data/real_photos/` |
| F10 | FastAPI service (`/health`, `/api/demo-samples`, `/api/inspect`, `/api/passport/*`, `/api/analytics`) | M3 | R4, API AC | `src/api.py` |
| F11 | Zero-Build laboratory web UI (`375px - 1440px`, 3 modes, `#1D4ED8`/`#1E3A8A`, self-hosted fonts) | M3 | R4, Browser AC, Slop AC | `src/web/index.html`, `src/web/styles.css`, `src/web/app.js`, `src/web/fonts/` |
| F12 | Grant project description (`<= 15` pp., Appendix 2, 1st person Pavlovets A.I.) | M4 | R1, R5, Grant AC | `grant_case/project_description.md`, `grant_case/build_docx.py`, `grant_case/CeramScan_Sistema_2026.docx` |
| F13 | Grant presentation (`10` slides `16:9`, Appendix 3, live JSON + PNG overlays) | M4 | R1, R5, Grant AC | `grant_case/build_presentation.py`, `grant_case/CeramScan_Sistema_2026.pptx` |
| F14 | Video/defense scripts, 12 jury Q&As, factory pilot letter, Nov 30 roadmap | M4 | R1, R5 | `grant_case/pitch_and_video_script.md`, `grant_case/pilot_letter_template.md`, `grant_case/REFINEMENTS_ROADMAP.md` |
| F15 | Public portfolio `README.md` & full `slop_scan.py` zero-indicator clearance | M4 | R5, Slop AC | `README.md` |
| F16 | Independent E2E suite, Tier 5 adversarial check, Chrome QA, local Git commit | Final | All Acceptance Criteria | `tests/`, `tests/e2e/` |

## Milestones

| Milestone | Scope | Dependencies | Status |
|---|---|---|---|
| M1 | Environment `.venv`, `requirements.txt`, `.gitignore`, `PROJECT.md`, `ARCHITECTURE.md`, `DESIGN_SPEC.md` | None | DONE |
| M2 | `src/cv_core/*`, `scripts/*`, synthetic stand v0.1, real CPU benchmark, `data/real_photos/` scaffolding | M1 | DONE |
| M3 | `src/api.py`, `src/web/` (including self-hosted `.woff2` fonts), API & headless Chrome QA | M1, M2 | DONE |
| M4 | `grant_case/` (`.docx`, `.pptx`, scripts, roadmap), `README.md`, LibreOffice + Poppler visual PDF/slide audit | M2, M3 | DONE |
| E2E | Full automated test suite (`tests/test_cv_core.py`, `tests/test_api.py`, `tests/test_scripts.py`, `tests/e2e/`) | M2, M3, M4 | DONE |
| Final | `slop_scan.py` zero-finding clearance, `ssh -T git@github.com` check, local git commit (no `git push`) | E2E | DONE |

## Interface Contracts

1. `src/cv_core/preprocessing.py` exposes `preprocess_image(image_bgr: np.ndarray, stage: StageType = "glazed", config: PreprocessConfig | None = None) -> PreprocessedFrame`, transforming raw BGR arrays into canonical `640x640` frames with `m_item`, `m_rim`, `m_interior`, `m_spec`, `m_cobalt`, and `EllipseGeometry`.
2. `src/cv_core/defect_detector.py` exposes `DefectDetector(weights_path: Path | None = None, config: DetectorConfig | None = None, preprocess_config: PreprocessConfig | None = None).inspect(image_bgr: np.ndarray, stage: StageType = "glazed", reference_diameter_mm: float = 200.0) -> InspectionResult`, executing either deterministic CV (`model_mode="deterministic"`) or ONNX/YOLO inference (`model_mode="onnx"`) with GOST 28390-89 triage (`PASS` / `REWORK` / `REJECT`) and PNG overlay bytes.
3. `src/cv_core/fingerprint.py` exposes `extract_fingerprint(frame: PreprocessedFrame, config: FingerprintConfig | None = None) -> FingerprintSignature` and `verify_fingerprint(reference: FingerprintSignature, probe_frame: PreprocessedFrame, config: FingerprintConfig | None = None) -> VerificationResult` using weighted `ORB + 64-bit pHash + cobalt HSV histogram` similarity (`0.50 / 0.25 / 0.25`) and returning `AuthVerdictType` (`"Подлинник"` vs counterfeit verdict defined in `ARCHITECTURE.md`).
4. `src/cv_core/storage.py` exposes `CeramStorage(db_path: Path, qr_dir: Path, passport_ref_dir: Path | None = None)` managing SQLite WAL tables (`inspections`, `passports` with `batch_note` and `reference_image_path`, `passport_verifications`), safe parameterized queries, and `CERAMSCAN:<item_id>` QR PNG generation.
5. `src/api.py` serves `GET /health`, `GET /api/demo-samples` (both inspection `samples` and demo `passports`), `POST /api/inspect`, `POST /api/passport/register` (returning `reference_image_url`), `POST /api/passport/verify/{item_id}` (returning `reference_image_url` alongside `match_percent` and sub-scores), and `GET /api/analytics` (`?limit=25&stage=&verdict=`), enforcing a 10 MB upload cap and image decode guard.
6. `scripts/generate_synthetic_ceramic.py` writes `data/synthetic/manifest.json` (`>= 24` labeled samples); `scripts/run_benchmark.py` executes real CPU measurements into `reports/benchmark_results.json`, `reports/benchmark_summary.md`, and `reports/overlays/*.png`; `scripts/ingest_real_photos.py --input data/real_photos --rebuild-all` updates the benchmark and triggers `grant_case/build_presentation.py`.

## Code Layout

```text
ceramscan-sistema/
├── .gitignore
├── requirements.txt
├── PROJECT.md
├── ARCHITECTURE.md
├── DESIGN_SPEC.md
├── README.md
├── data/
│   ├── reference_materials/                 # Pre-loaded GOST, Sistema PDFs, ceramic_defect_1..4.jpg (READ-ONLY)
│   ├── synthetic/                           # Generated by scripts/generate_synthetic_ceramic.py (v0.1)
│   │   ├── manifest.json
│   │   └── images/
│   └── real_photos/                         # Operator drop-in folders for 20-30 factory photos
│       ├── .gitkeep
│       ├── pass/.gitkeep
│       ├── chip/.gitkeep
│       ├── crack/.gitkeep
│       ├── glaze/.gitkeep
│       └── fake/.gitkeep
├── src/
│   ├── __init__.py
│   ├── api.py                               # FastAPI service & static mounts
│   ├── cv_core/
│   │   ├── __init__.py
│   │   ├── preprocessing.py                 # ROI, perspective, LAB CLAHE, glare & cobalt masks
│   │   ├── defect_detector.py               # Deterministic CV + --weights ONNX + GOST triage
│   │   ├── fingerprint.py                   # 289-FZ ORB + pHash + cobalt histogram matcher
│   │   └── storage.py                       # SQLite WAL + QR generator (qrcode)
│   └── web/
│       ├── index.html                       # Zero-Build 3-mode UI shell
│       ├── styles.css                       # Normative tokens, layout, GPU motion
│       ├── app.js                           # Client logic & API binding
│       └── fonts/                           # Self-hosted IBM Plex Sans & JetBrains Mono .woff2
├── scripts/
│   ├── __init__.py
│   ├── generate_synthetic_ceramic.py
│   ├── run_benchmark.py
│   └── ingest_real_photos.py
├── reports/
│   ├── benchmark_results.json
│   ├── benchmark_summary.md
│   ├── overlays/                            # Benchmark PNGs + runtime/ (ignored)
│   ├── passports/                           # Runtime passport reference snapshots (ignored)
│   └── qr/                                  # Runtime QR PNGs (ignored)
├── grant_case/
│   ├── project_description.md
│   ├── build_docx.py
│   ├── build_presentation.py
│   ├── CeramScan_Sistema_2026.docx
│   ├── CeramScan_Sistema_2026.pptx
│   ├── pitch_and_video_script.md
│   ├── pilot_letter_template.md
│   └── REFINEMENTS_ROADMAP.md
└── tests/
    ├── __init__.py
    ├── test_cv_core.py
    ├── test_api.py
    ├── test_scripts.py
    ├── test_adversarial_tier5.py
    └── e2e/
        ├── __init__.py
        ├── conftest.py
        ├── test_e2e_pipeline.py
        ├── test_e2e_api_and_ui.py
        └── test_e2e_grant_and_repo.py
```
