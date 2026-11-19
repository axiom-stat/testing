# CeramScan Architecture & Locked Contracts

## 1. Type Aliases, Enums & `src/cv_core/preprocessing.py`

All CV modules target Python 3.12 with strict type hints, flat `dataclass(slots=True)` structures, and zero mutable default arguments (`None` or `default_factory` only).

```python
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal
import numpy as np

StageType = Literal["bisque", "glazed"]
VerdictType = Literal["PASS", "REWORK", "REJECT"]
AuthVerdictType = Literal["Подлинник", "Подозрение на контрафакт или штамповку"]


class DefectClass(str, Enum):
    CRACK = "crack"                  # трещина (bisque/glazed) or цек (glazed)
    CHIP = "chip"                    # скол кромки / откол
    GLAZE_BARE = "glaze_bare"        # плешина / сборка глазури / свободный край / выгорка
    PINHOLE_SPECK = "pinhole_speck"  # накол / мушка / засорка
    GLAZE_DRIP = "glaze_drip"        # натек / прыщ
    SHAPE_WARP = "shape_warp"        # деформация от характерного размера (> 1.5%)


@dataclass(slots=True)
class PreprocessConfig:
    target_size: int = 640
    clahe_clip_bisque: float = 2.0
    clahe_clip_glazed: float = 1.8
    clahe_grid: tuple[int, int] = (8, 8)
    glare_l_thresh: int = 235
    glare_s_thresh: int = 30
    rim_band_ratio: float = 0.055          # 5.5% of minor axis (calibrated 4-6%)
    cobalt_s_min: int = 45                 # OpenCV HSV S in [0..255]
    cobalt_h_range: tuple[int, int] = (90, 135)  # OpenCV HSV H in [0..179]
    cobalt_b_max: int = 118                # OpenCV LAB b* (<128 is blue shift)
    reference_diameter_mm: float = 200.0
    min_item_area_ratio: float = 0.12      # Fallback to patch mode if <12% or >92%


@dataclass(slots=True)
class EllipseGeometry:
    center: tuple[float, float]
    axes: tuple[float, float]              # (major_px, minor_px)
    angle_deg: float
    contour_deviation_pct: float           # max radial deviation / mean_radius * 100
    axis_ratio_error_pct: float            # |1 - major/minor| check for circular reference
    px_per_mm: float
    scale_calibrated: bool                 # True if item/frame contour found; False on macro crop


@dataclass(slots=True)
class PreprocessedFrame:
    stage: StageType
    original_bgr: np.ndarray               # Original decoded BGR uint8 (max dim <= 1600)
    roi_bgr: np.ndarray                    # Canonical (target_size, target_size, 3) uint8
    roi_lab_l_clahe: np.ndarray            # (target_size, target_size) uint8 enhanced L channel
    roi_hsv: np.ndarray                    # (target_size, target_size, 3) uint8
    m_item: np.ndarray                     # uint8 binary mask (0/255) of ceramic object
    m_rim: np.ndarray                      # uint8 binary mask (0/255) contour rim band
    m_interior: np.ndarray                 # uint8 binary mask (0/255) eroded interior
    m_spec: np.ndarray                     # uint8 binary mask (0/255) specular glare (L>235 & S<30)
    m_cobalt: np.ndarray                   # uint8 binary mask (0/255) Gzhel cobalt pigment
    geometry: EllipseGeometry
    perspective_applied: bool
    glare_ratio: float
    cobalt_ratio: float
    latency_ms: float
```

Entry function in `src/cv_core/preprocessing.py`:

```python
def preprocess_image(
    image_bgr: np.ndarray,
    stage: StageType = "glazed",
    config: PreprocessConfig | None = None,
) -> PreprocessedFrame: ...
```

Validation and ROI rules:
1. Raise `ValueError` if `image_bgr` is `None`, empty, `ndim != 3`, or `min(h, w) < 32`.
2. Calibration quad detection: search high-contrast 4-vertex `cv2.approxPolyDP` polygons covering `20%..95%` of frame area. If found, apply `cv2.getPerspectiveTransform` + `cv2.warpPerspective` to `(target_size, target_size)`, setting `perspective_applied = True` and `scale_calibrated = True`.
3. Otherwise segment ceramic foreground `m_item` via Otsu/adaptive luminance + saturation separation from neutral/dark backdrop and morphological close/fill-holes.
4. If `m_item` covers `12%..92%` of frame area and touches fewer than `3` image borders, fit `cv2.fitEllipse(largest_contour)` (when `len(contour) >= 5`), center-pad/crop to `(target_size, target_size)`, compute `contour_deviation_pct = max(|r_actual - r_ellipse|) / mean_radius * 100`, and set `px_per_mm = major_px / config.reference_diameter_mm`.
5. If input is a tight macro patch without a complete outer rim (matching `ceramic_defect_1..4.jpg` 200x200 patches or tilted tile edges), resize cleanly to `(target_size, target_size)`, assign inner `90%` (`5%` border inset) to `m_interior`, assign the outer `5%` ring to `m_rim` only when an actual rim edge is present, and set `scale_calibrated = False` (`px_per_mm = target_size / 160.0`). This prevents tilted patch boundaries (seen at top-left and bottom-left of `ceramic_defect_4.jpg`) from triggering false crack detections.

---

## 2. Calibration Rules from `ceramic_defect_1..4.jpg` & CPU Budget

Every filter parameter maps directly to measured properties of `data/reference_materials/ceramic_defect_samples/ceramic_defect_1..4.jpg`:

1. Rim vs interior isolation (`m_rim` vs `m_interior`):
   - `ceramic_defect_4.jpg` exhibits dark tilted frame boundaries at top-left and bottom-left.
   - Erode `m_item` by `k_rim = max(7, int(minor_axis * config.rim_band_ratio))` to construct `m_interior`. Surface crack, pinhole, and glaze-bare detectors execute exclusively inside `m_interior`. Rim chip (`скол кромки`) and contour warp (`перекос контура`) execute exclusively on `m_rim = cv2.dilate(contour_mask, k_rim) & ~cv2.erode(m_item, k_rim)`.
2. Stage-aware glare suppression without erasing crazing (`цек`):
   - `ceramic_defect_4.jpg` carries `58.5%` specular glare (`L > 235 & S < 25`), `gray_min = 41`, `gray_max = 255`, a bold horizontal crack, and faint lower crazing (`цек`) partially obscured by glare. `ceramic_defect_1.jpg` and `ceramic_defect_2.jpg` show `0.0%` glare.
   - For `bisque` (`~900°C`, matte), apply `CLAHE(clipLimit=2.0, tileGridSize=(8,8))` on LAB `L` with no glare masking (`m_spec` remains zeros).
   - For `glazed` (`~1350°C`, glossy), compute `m_spec = ((L > 235) & (S < 30)).astype(np.uint8) * 255`. Never apply Telea inpainting across wide `m_spec` zones (`> 15%` area), because inpainting blurs faint `цек` lines. Instead:
     a) Suppress edge/Canny false alarms on specular borders using `~cv2.dilate(m_spec, 5x5_kernel)`.
     b) Run `CLAHE(clipLimit=1.8, tileGridSize=(8,8))` and extract **Black-Hat morphology** (`L_closed - L`) plus dark-ridge response (`skimage.filters.sato(black_ridges=True)` / directional Black-Hat). Bright glare plateaus produce `0` Black-Hat response, while dark crack troughs (`gray_min = 41` in `defect_4`, `gray_min = 27` in `defect_1`) yield sharp positive ridges (`blackhat p99 = 97` on `defect_4`, `47` on `defect_1`).
3. Texture-resistant hairline crack separation (`defect_2` and `defect_3`):
   - `ceramic_defect_2.jpg` has orange-peel matte texture (`laplacian_var = 1335.0`, `blackhat p99 = 58`) with a thin horizontal crack in the lower third and a shallow bare patch (`плешина`/`выгорка`) at bottom-left.
   - `ceramic_defect_3.jpg` has dense vertical striations (`laplacian_var = 23091.9`, `blackhat p99 = 154`) with a deep cavity (`gray_min = 2`, bright rim `250`) mid-right and a smaller pit mid-left.
   - Pre-smooth `L_clahe` with edge-preserving `cv2.bilateralFilter(L_clahe, d=5, sigmaColor=25, sigmaSpace=5)` to attenuate high-frequency glaze grain while keeping 2-4 px crack troughs sharp.
   - Reject periodic striation noise (`defect_3`) by comparing anisotropic directional Black-Hat responses and requiring crack candidates after skeletonization / connected-component analysis to pass minimum skeleton length (`>= 24 px` at `640x640`), high elongation (`eccentricity >= 0.88` or `major/minor >= 4.0`), narrow width (`<= 10 px`), and peak contrast above local texture noise (`> mean + 2.5 * local_std`).
   - Deep compact cavities (`solidity > 0.60`, `eccentricity < 0.85`, high Black-Hat `> 80`, `L_min < 35`) classify as `GLAZE_BARE` (`выгорка`) or `PINHOLE_SPECK` (`накол`/`мушка`) rather than cracks.
4. Gzhel cobalt pigment masking:
   - Compute `m_cobalt = (((S > 45) & (H >= 90) & (H <= 135)) | (b_lab < 118)).astype(np.uint8) * 255`.
   - Exclude `m_cobalt` from dark-speck (`мушка`) and glaze-drip (`натек`) detectors so hand-painted blue strokes never trigger defect alerts. Allow crack detection across `m_cobalt` only when a sharp `L` trough (`blackhat > 45` and strong ridge sharpness) cuts orthogonally across pigment boundaries. Pass `m_cobalt` directly to `fingerprint.py`.
5. CPU latency budget (`< 300 ms/frame` hard SLA, `< 120 ms/frame` target on `640x640`):
   - Image decode + `preprocess_image`: `<= 20 ms` (steady-state OpenCV probe `1.74 ms`)
   - Bilateral + Black-Hat + ridge/skeleton crack and compact defect extraction: `<= 80 ms` (`4.11 ms` core OpenCV ops, reserving headroom for `skimage` ridge/skeleton on active patches)
   - GOST triage + PNG overlay render: `<= 30 ms`
   - Total `inspect()`: `<= 130 ms` typical, strictly `< 300 ms` on CPU.

---

## 3. GOST 28390-89 Verbatim 1st-Grade Limits vs CeramScan Operational Triage

CeramScan keeps **verbatim 1st-grade (`1-й сорт`) tolerances** from `data/reference_materials/gost_28390_89_quality_table.md` (STENOVA HOME PORCELAIN 2023 based on ГОСТ 28390-89) separate from **CeramScan operational triage (`PASS` / `REWORK` / `REJECT`)** across `bisque` (`~900°C`) and `glazed` (`~1350°C`), because the reference table covers 1st grade only without separate 2nd/3rd grade columns.

| Defect / Rule | Verbatim GOST 28390-89 (1st Grade) | `DefectClass` | `bisque` (`~900°C`) Triage | `glazed` (`~1350°C`) Triage |
|---|---|---|---|---|
| Global defect cap | `<= 3` defects per item total (`Общее количество недочетов <= 3`) | All | `> 3` -> `REWORK` (if sandable) or `REJECT` | `> 3` -> `REJECT` (or `REWORK` if minor glaze only) |
| `Трещина` / `цек` | Not allowed (`Не допускается`) | `CRACK` | Unconditional `REJECT` (prevents 1350°C kiln waste) | Unconditional `REJECT` |
| `Деформация от характерного размера` | Flat & hollowware: `<= 1.5%` | `SHAPE_WARP` | `<= 1.5%` -> `PASS`; `> 1.5%` -> `REJECT` | `<= 1.5%` -> `PASS`; `> 1.5%` -> `REJECT` |
| `Откол` / `скол кромки` | Glazed/ground chip: small `0.5-1 mm <= 1`; med/large `1.5-2 mm <= 2`; raw chip not allowed | `CHIP` | Shallow (`<= 1.5 mm`, `<= 1`) -> `REWORK` (grind); deeper or `> 1` -> `REJECT` | Unglazed or deep chip -> `REJECT` |
| `Плешина`, `сборка глазури` | Small items: not allowed; Med/Large: face `<= 20 mm2`, back `<= 50 mm2` | `GLAZE_BARE` | Not applicable (`PASS` unless `выгорка`) | `<= 20 mm2` (single) -> `PASS` (med/large) or `REWORK`; `> 20 mm2` isolated -> `REWORK` (re-glaze & refire), else `REJECT` |
| `Свободный от глазури край` | `<= 2 mm` | `GLAZE_BARE` (rim) | N/A | `<= 2 mm` -> `PASS`; `> 2 mm` -> `REWORK` |
| `Выгорка` | Not allowed (`Не допускается`) | `GLAZE_BARE` (deep pit) | Tiny isolated -> `REWORK` (fill/sand); deep/multi -> `REJECT` | `REJECT` |
| `Накол` | Fine scattered `<= 8 pcs` | `PINHOLE_SPECK` | N/A | `<= 8` scattered (within total defect cap) -> `PASS` / `REWORK`; `> 8` -> `REJECT` |
| `Мушка` / `Засорка` | `Мушка`: `<= 0.5 mm` (`<= 2` face); `Засорка`: `<= 2.0 mm` (`<= 2-3` pcs) | `PINHOLE_SPECK` | `<= 2` sandable -> `REWORK` (sand pre-glaze); `> 3` -> `REJECT` | Within `<= 0.5 mm` & `<= 2` -> `PASS`; above limit -> `REJECT` |
| `Натек` / `Прыщ` | `Натек`: barely noticeable; `Прыщ` face: not allowed, back `<= 1.5 mm <= 1` | `GLAZE_DRIP` | N/A | Localized `натек` -> `REWORK` (grind); face `прыщ` or severe -> `REJECT` |

---

## 4. `src/cv_core/defect_detector.py` & `--weights` ONNX/YOLO Contract

```python
@dataclass(slots=True)
class DefectRecord:
    defect_class: DefectClass
    gost_name_ru: str
    severity: Literal["minor", "rework", "critical"]
    bbox_xywh: tuple[int, int, int, int]   # Canonical target_size coordinates
    area_px: int
    area_mm2: float
    length_mm: float
    confidence: float                      # [0.0, 1.0]
    zone: Literal["rim", "interior"]
    source: Literal["deterministic", "onnx"]


@dataclass(slots=True)
class DetectorConfig:
    conf_threshold: float = 0.25
    nms_iou_threshold: float = 0.45
    min_crack_skeleton_px: int = 24
    min_crack_eccentricity: float = 0.88
    max_gost_defects_grade1: int = 3
    max_contour_deviation_pct: float = 1.5
    max_glaze_bare_mm2: float = 20.0
    max_pinhole_count: int = 8


@dataclass(slots=True)
class InspectionResult:
    stage: StageType
    verdict: VerdictType
    gost_grade: Literal["1 сорт", "Доработка (REWORK)", "Брак (REJECT)"]
    model_mode: Literal["deterministic", "onnx"]
    contour_deviation_pct: float
    defect_count: int
    defects: list[DefectRecord]
    gost_rules_triggered: list[str]
    glare_ratio: float
    cobalt_ratio: float
    scale_calibrated: bool
    latency_ms: float
    overlay_bgr: np.ndarray
    overlay_png_bytes: bytes


class DefectDetector:
    def __init__(
        self,
        weights_path: Path | None = None,
        config: DetectorConfig | None = None,
        preprocess_config: PreprocessConfig | None = None,
    ) -> None: ...

    def inspect(
        self,
        image_bgr: np.ndarray,
        stage: StageType = "glazed",
        reference_diameter_mm: float = 200.0,
    ) -> InspectionResult: ...
```

Exact `--weights` ONNX/YOLO contract:
1. When `weights_path is None`: execute deterministic CV (`model_mode = "deterministic"`).
2. When `weights_path is not None`:
   - Check `weights_path.is_file()` and `weights_path.suffix.lower() == ".onnx"`. Raise `FileNotFoundError` or `ValueError` immediately if missing or invalid (never fall back silently when an invalid `--weights` path was supplied).
   - Create `onnxruntime.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])` and inspect input tensor shape `(1, 3, H, W)` (default `640x640`).
   - Letterbox `frame.roi_bgr`, convert BGR to RGB `float32 / 255.0`, transpose to `(1, 3, H, W)`.
   - Run `session.run(None, {input_name: tensor})[0]`.
   - Decode both Ultralytics YOLOv8/v11 `(1, 4 + C, N)` (squeeze, transpose to `(N, 4 + C)`, `class_scores = row[4:]`) and YOLOv5 `(1, N, 5 + C)` (`class_scores = row[4] * row[5:]`). Raise `ValueError` on unsupported tensor rank or channel dimension.
   - Run `cv2.dnn.NMSBoxes` with `config.conf_threshold` and `config.nms_iou_threshold`, un-letterbox `(x_c, y_c, w, h)` back to canonical `target_size`, and map `class_id` via `{0: CRACK, 1: CHIP, 2: GLAZE_BARE, 3: PINHOLE_SPECK, 4: GLAZE_DRIP, 5: SHAPE_WARP}` (overridden by ONNX `names` metadata if present).
   - Retain the deterministic GOST geometry check (`contour_deviation_pct > 1.5%` -> `SHAPE_WARP`) and run the identical GOST triage engine, setting `model_mode = "onnx"`.

---

## 5. `src/cv_core/fingerprint.py` Contract (289-FZ Anti-Counterfeit)

```python
@dataclass(slots=True)
class FingerprintConfig:
    orb_nfeatures: int = 500
    lowe_ratio: float = 0.75
    ransac_reproj_thresh: float = 5.0
    weight_orb: float = 0.50
    weight_phash: float = 0.25
    weight_color: float = 0.25
    authentic_threshold_pct: float = 62.0
    min_matched_kp: int = 12
    min_cobalt_ratio: float = 0.015        # Guard against non-cobalt blank/monochrome fakes


@dataclass(slots=True)
class FingerprintSignature:
    phash_hex: str                         # 16-char hex representing 64-bit DCT hash
    cobalt_hist: list[float]               # L1-normalized 2D HSV (18x16 = 288 bins)
    cobalt_ratio: float                    # Fraction of m_item covered by m_cobalt
    keypoints_xy: list[list[float]]        # [[x, y], ...] length K
    descriptors_b64: str                   # Base64-encoded uint8 (K, 32) ORB descriptors ("" if K==0)
    orb_count: int
    version: str = "1.0"


@dataclass(slots=True)
class VerificationResult:
    match_percent: float                   # [0.0, 100.0]
    orb_score: float                       # [0.0, 1.0]
    phash_score: float                     # [0.0, 1.0]
    color_score: float                     # [0.0, 1.0]
    matched_kp: int
    verdict: AuthVerdictType               # Authentic or counterfeit label
    latency_ms: float


def extract_fingerprint(
    frame: PreprocessedFrame,
    config: FingerprintConfig | None = None,
) -> FingerprintSignature: ...


def verify_fingerprint(
    reference: FingerprintSignature,
    probe_frame: PreprocessedFrame,
    config: FingerprintConfig | None = None,
) -> VerificationResult: ...
```

Similarity calculation:
1. **64-bit DCT pHash**: resize grayscale `roi` inside `m_item` to `32x32` `float32`, run `cv2.dct`, take top-left `8x8` block, compare against median of the 63 AC coefficients (`[0, 0]` excluded), pack into 64 bits (`phash_hex`). `phash_score = 1.0 - (hamming_bits / 64.0)`.
2. **Cobalt HSV histogram**: compute `cv2.calcHist([roi_hsv], [0, 1], m_cobalt, [18, 16], [85, 140, 35, 256])`, L1-normalize. Compute Bhattacharyya distance `bhatt = cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)`. Penalize coverage mismatch via `coverage_factor = min(r1, r2) / max(r1, r2, 1e-6)` when both ratios `>= min_cobalt_ratio`, `0.0` if one has cobalt and the other does not (if both are non-cobalt monochrome test patches `< min_cobalt_ratio`, fall back to `m_interior` HSV histogram with a `0.5` factor). `color_score = float(np.clip((1.0 - bhatt) * (0.4 + 0.6 * coverage_factor), 0.0, 1.0))`.
3. **ORB + Lowe + RANSAC**: detect `cv2.ORB_create(nfeatures=500)` inside `cv2.dilate(m_item, 5x5)` excluding `m_spec`. Match descriptors with `cv2.BFMatcher(cv2.NORM_HAMMING)` `knnMatch(k=2)`, keeping `m.distance < 0.75 * n.distance`. When `len(good) >= 8`, run `cv2.findHomography(..., cv2.RANSAC, 5.0)` and count RANSAC inliers (`orb_score = min(1.0, inliers / 35.0)`); otherwise `orb_score = min(1.0, len(good) / 45.0)`.
4. **Combined verdict**: `match_percent = round(100.0 * (0.50 * orb_score + 0.25 * phash_score + 0.25 * color_score), 2)`. Return `"Подлинник"` iff `match_percent >= 62.0` AND `matched_kp >= 12`; otherwise `"Подозрение на контрафакт или штамповку"`.

---

## 6. `src/cv_core/storage.py` Contract (SQLite WAL + QR)

Every SQLite connection runs:

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
```

Exact DDL:

```sql
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT,
    stage TEXT NOT NULL CHECK (stage IN ('bisque', 'glazed')),
    shift TEXT NOT NULL DEFAULT 'S1',
    verdict TEXT NOT NULL CHECK (verdict IN ('PASS', 'REWORK', 'REJECT')),
    gost_grade TEXT NOT NULL,
    model_mode TEXT NOT NULL CHECK (model_mode IN ('deterministic', 'onnx')),
    defect_count INTEGER NOT NULL,
    primary_defect TEXT,
    contour_deviation_pct REAL NOT NULL,
    glare_ratio REAL NOT NULL,
    latency_ms REAL NOT NULL,
    defects_json TEXT NOT NULL,
    gost_rules_json TEXT NOT NULL,
    overlay_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS passports (
    item_id TEXT PRIMARY KEY,
    sku TEXT NOT NULL,
    master_name TEXT NOT NULL,
    batch_note TEXT NOT NULL DEFAULT '',
    stage TEXT NOT NULL CHECK (stage IN ('bisque', 'glazed')),
    qr_payload TEXT NOT NULL,
    qr_image_path TEXT NOT NULL,
    reference_image_path TEXT NOT NULL DEFAULT '',
    signature_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS passport_verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES passports(item_id) ON DELETE CASCADE,
    match_percent REAL NOT NULL,
    orb_score REAL NOT NULL,
    phash_score REAL NOT NULL,
    color_score REAL NOT NULL,
    matched_kp INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL
);
```

QR & path-safety rules:
- QR payload string: `f"CERAMSCAN:{item_id}"`.
- Saved via `qrcode.QRCode` (`box_size=8, border=2`) as PNG to `qr_dir / f"{item_id}.png"`, exposed via `/qr/{item_id}.png`.
- Passport reference snapshot saved to `passport_ref_dir / f"{item_id}.png"` (default `reports/passports/`), exposed via `/passports/{item_id}.png` so Mode 2 verification displays the stored reference beside the probe image.
- Validate `item_id` against `^[A-Za-z0-9_-]{3,64}$` before DB lookup or file path construction, and enforce `Path.resolve()` containment inside `qr_dir` / `passport_ref_dir`. All SQL uses `?` parameters.

```python
class CeramStorage:
    def __init__(
        self,
        db_path: Path,
        qr_dir: Path,
        passport_ref_dir: Path | None = None,
    ) -> None: ...
```

---

## 7. Reconciled FastAPI Service Contract (`src/api.py`)

Environment variables & limits:
- `CERAMSCAN_DB_PATH` (default `data/ceramscan.db`)
- `CERAMSCAN_WEIGHTS_PATH` (optional path to `.onnx`)
- `CERAMSCAN_OVERLAY_DIR` (default `reports/overlays/runtime`)
- `CERAMSCAN_QR_DIR` (default `reports/qr`)
- `CERAMSCAN_PASSPORT_DIR` (default `reports/passports`)
- `MAX_UPLOAD_BYTES = 10 * 1024 * 1024` (10 MB)

Static & asset mounts:
- `GET /` -> `src/web/index.html`
- `/static/*` -> `src/web/` (`styles.css`, `app.js`, `fonts/*.woff2`)
- `/overlays/*` -> `reports/overlays/` (includes `runtime/`)
- `/qr/*` -> `reports/qr/`
- `/passports/*` -> `reports/passports/`
- `/demo-assets/*` -> `data/synthetic/images/` (falls back to `data/reference_materials/ceramic_defect_samples/` if synthetic assets are absent)

Endpoints & reconciled JSON schemas:

| Method & Path | Request | Response (`200`) | Error Codes |
|---|---|---|---|
| `GET /health` | None | `{"status": "ok", "version": "0.1.0", "model_mode": "deterministic"\|"onnx", "db_ok": true}` | `500` on DB error |
| `GET /api/demo-samples` | None | `{"samples": [{"id": str, "label": str, "stage": "bisque"\|"glazed", "image_url": str}], "passports": [{"item_id": str, "label": str, "sku": str, "master_name": str, "probe_image_url": str, "is_counterfeit": bool}]}` | `200` empty arrays if unpopulated |
| `POST /api/inspect` | `multipart/form-data`: `file` (required), `stage` (`"bisque"`\|`"glazed"`, default `"glazed"`), `item_id` (optional), `shift` (default `"S1"`), `reference_diameter_mm` (float, default `200.0`), `save_to_db` (bool, default `true`) | `{"inspection_id": int\|null, "item_id": str\|null, "stage": str, "verdict": "PASS"\|"REWORK"\|"REJECT", "gost_grade": str, "model_mode": str, "contour_deviation_pct": float, "defect_count": int, "defects": [...], "gost_rules_triggered": [...], "metrics": {"latency_ms": float, "glare_ratio": float, "cobalt_ratio": float, "scale_calibrated": bool}, "overlay_url": str, "overlay_base64": str}` | `400` empty/corrupt image or bad `item_id`; `413` `> 10 MB`; `422` invalid fields |
| `POST /api/passport/register` | `multipart/form-data`: `file` (required), `item_id` (optional, auto `GZHEL-YYYYMMDD-<6hex>`), `sku` (default `"GZHEL-SKU-01"`), `master_name` (default `"Мастер Гжель"`), `batch_note` (default `""`), `stage` (default `"glazed"`) | `{"item_id": str, "sku": str, "master_name": str, "batch_note": str, "stage": str, "qr_payload": str, "qr_image_url": str, "reference_image_url": str, "orb_count": int, "cobalt_ratio": float, "created_at": str}` | `400` corrupt image or invalid `item_id`; `409` duplicate `item_id`; `413` `> 10 MB`; `422` invalid fields |
| `POST /api/passport/verify/{item_id}` | Path `item_id` + `multipart/form-data`: `file` (required) | `{"item_id": str, "verdict": AuthVerdictType, "match_percent": float, "orb_score": float, "phash_score": float, "color_score": float, "matched_kp": int, "latency_ms": float, "registered_sku": str, "registered_master": str, "registered_batch_note": str, "reference_image_url": str}` | `400` corrupt image / bad `item_id`; `404` unknown `item_id`; `413` `> 10 MB` |
| `GET /api/analytics` | Query: `limit: int = 25` (`1..200`), `stage: str \| None = None` (`bisque`\|`glazed`), `verdict: str \| None = None` (`PASS`\|`REWORK`\|`REJECT`) | `{"summary": {"total_inspections": int, "pass_count": int, "rework_count": int, "reject_count": int, "pass_rate_pct": float, "avg_latency_ms": float, "total_passports": int, "total_verifications": int, "counterfeit_flags": int}, "by_defect_class": [...], "by_stage": [...], "by_shift": [...], "recent_inspections": [...]}` | `422` invalid query bounds or enum |

Upload & image validation guard:
- Read upload stream up to `MAX_UPLOAD_BYTES + 1` bytes; return HTTP `413` if exceeded.
- Return HTTP `400` if `len(raw) == 0`.
- Verify magic header (`PNG`, `JPEG`, `WEBP`, `BMP`, `TIFF`) or PIL decode, decode via `np.frombuffer` + `cv2.imdecode(..., cv2.IMREAD_COLOR)` (with PIL RGBA/RGB fallback), and reject with HTTP `400` if decode fails or `min(h, w) < 32`.

---

## 8. CLI & Benchmark Contracts (`scripts/`)

1. `scripts/generate_synthetic_ceramic.py`
   - Flags: `--output-dir` (default `data/synthetic`), `--count` (default `26`, minimum `24`), `--seed` (default `42`).
   - Writes PNG images to `<output-dir>/images/` and `<output-dir>/manifest.json` with:
     - `stand_label`: `"синтетический валидационный стенд v0.1"`
     - `seed`, `count`, `samples`: list of `{"id": str, "image_path": str, "stage": "bisque"|"glazed", "defect_class": "pass"|DefectClass, "expected_verdict": "PASS"|"REWORK"|"REJECT", "auth_pair_id": str|null, "is_counterfeit": bool}`
   - Default 26-sample distribution (`>= 24`): `bisque` pass (4), `bisque` crack (3), `bisque` chip (2), `bisque` warp (2), `glazed` pass with cobalt pattern (4), `glazed` crack/`цек` + specular glare (3), `glazed` `glaze_bare` (`плешина`) (2), `glazed` `pinhole_speck` + `glaze_drip` (2), counterfeit pattern variants paired to genuine `glazed` pass IDs (4).
2. `scripts/run_benchmark.py`
   - Flags: `--dataset` (default `data/synthetic/manifest.json`), `--weights` (optional `.onnx`), `--output-json` (default `reports/benchmark_results.json`), `--output-md` (default `reports/benchmark_summary.md`), `--overlays-dir` (default `reports/overlays`).
   - Exact `reports/benchmark_results.json` schema:

```json
{
  "generated_at": "ISO-8601 UTC",
  "dataset_label": "синтетический валидационный стенд v0.1",
  "model_mode": "deterministic",
  "sample_count": 26,
  "avg_latency_ms": 0.0,
  "p95_latency_ms": 0.0,
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "verdict_accuracy": 0.0,
  "per_class": {
    "crack": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}
  },
  "auth_benchmark": {
    "pair_count": 0,
    "auth_accuracy": 0.0,
    "genuine_mean_match": 0.0,
    "fake_mean_match": 0.0
  },
  "overlay_files": ["reports/overlays/...png"]
}
```

   - Every benchmark PNG under `reports/overlays/` renders side-by-side `"Исходник | Разметка CeramScan"` for both UI inspection and Slide 6/7 embedding.
3. `scripts/ingest_real_photos.py`
   - Flags: `--input` (default `data/real_photos`), `--export-yolo` (optional output dir for YOLO `.txt` + `data.yaml`), `--rebuild-all` (flag), `--weights` (optional `.onnx`).
   - Folder mapping under `--input`:
     - `pass/` -> `defect_class="pass"`, `expected_verdict="PASS"`, stage `"glazed"` (or `"bisque"` if filename contains `"bisque"` or `"util"`)
     - `chip/` -> `defect_class="chip"`, `expected_verdict="REJECT"`
     - `crack/` -> `defect_class="crack"`, `expected_verdict="REJECT"`
     - `glaze/` -> `defect_class="glaze_bare"`, `expected_verdict="REWORK"` (scored in glaze defect group)
     - `fake/` -> `is_counterfeit=true`, paired against the first genuine `pass/` item for 289-FZ authenticity testing
   - With `--rebuild-all`: emits real-photo `manifest.json` (or falls back cleanly to synthetic manifest if `0` real images exist, logging the fallback), runs `scripts/run_benchmark.py`, refreshes PNG overlays, and invokes `grant_case/build_presentation.py` to rebuild `CeramScan_Sistema_2026.pptx`.

---

## 9. Grant Case Contracts (`grant_case/`)

1. Author & 3-LPR alignment (`R1`):
   - Strictly 1st person singular: Павловец Александр Иванович, 1 курс экономического факультета Гатчинского Государственного университета (profile: «экономика предприятий и организаций»), г. Гатчина, `+79052010295`, `pavlovec000@gmail.com`.
   - Addresses all three decision-maker groups:
     a) БФ «Система» (Пастухова Л., Ксенофонтова А.): working GitHub prototype ready for factory pilot, not a paper-only concept.
     b) Объединение «Гжель»: addresses 130-140 тыс. annual output QC bottleneck, catches `~900°C` bisque defects before `~1350°C` firing, and protects hand-painted cobalt strokes under 289-FZ on Ozon/WB.
     c) МТС / АФК «Система» technical experts: deterministic `< 300 ms` CPU CV core, clean `--weights` ONNX hook, real pytest suite, and reproducible CPU benchmark.
2. Regulation date note (`sistema_2026_regulation_vuz.pdf`):
   - The cached regulation PDF records 2025/2026 cycle dates (20 Nov 2025 intake, Mar 2026 payout). Submission targets the 4th season (deadline November 30, 2026); Appendices 1-6 govern formatting and structure while 2026/2027 intake dates and university enrollment certificate format require final confirmation on `lift-bf.ru`.
3. `CeramScan_Sistema_2026.docx` (Appendix 2 / Приложение № 2):
   - Strictly `<= 15` pages, Times New Roman `12 pt`, line spacing `1.0`, numbered pages, numbered inline figures and tables, mandatory source list, all 7 structural sections and 10 required content topics (including import substitution and pilot readiness), `0` `slop_scan.py` findings, RU `-ание/-ение` ratio `< 0.05`.
4. `CeramScan_Sistema_2026.pptx` (Appendix 3 / Приложение № 3):
   - Exactly `10` slides `16:9` in Swiss minimal layout, `40 - 60` words/slide, live numbers pulled directly from `reports/benchmark_results.json`, real PNG overlays from `reports/overlays/`, mapped to `1.1` Title (Slide 1), `1.2` Problem/Relevance & Goal (Slides 2-3), `1.3` Theory, Architecture, Benchmark, 289-FZ UI & Unit Economics (Slides 4-8), `1.4` Conclusions (Slide 9), `1.5` Sources & Contacts (Slide 10).
5. Pitch, letter & Nov 30 roadmap:
   - `grant_case/pitch_and_video_script.md` (90-second video script, 7-minute oral defense, 12 hard jury Q&As), `grant_case/pilot_letter_template.md`, `grant_case/REFINEMENTS_ROADMAP.md` (step-by-step instructions for capturing 20-30 real factory photos and running `ingest_real_photos.py --rebuild-all`).

---

## 10. Verified macOS Office/PDF & Browser QA Toolchain

Empirical probes run during M1 established the exact headless rendering pipeline on this Apple Silicon host:

1. Rejected Office automation paths:
   - `osascript` automation of Microsoft PowerPoint and Microsoft Word hangs and exits with `AppleEvent timed out (-1712)` due to macOS App Sandbox / GUI modal dialogs.
   - `sdef` fails without full `Xcode.app` (`CommandLineTools` active).
   - `qlmanage -t` renders only a single cover thumbnail (`slide 1` / `page 1`) and `qlmanage -p` produces unpaginated HTML without `.docx` page boundaries.
2. Locked M4 Office/PDF QA pipeline:
   - Primary headless export: install `brew install --cask libreoffice && brew install poppler`.
   - Convert `.docx` / `.pptx` to PDF: `soffice --headless --convert-to pdf <file> --outdir <out_dir>`.
   - Verify `.docx <= 15` pages and `.pptx == 10` slides using `.venv/bin/python` with `pypdf.PdfReader(pdf_path).pages`.
   - Render every PDF page/slide to PNG at 150 DPI for visual inspection: `pdftoppm -png -r 150 <pdf_path> <prefix>`.
   - Native macOS fallback for PDF -> PNG (if `pdftoppm` is temporarily unavailable): `/usr/bin/swift` importing `PDFKit` and `AppKit` (note `/usr/bin/python3` lacks `Quartz`).
3. Locked M3/E2E Browser QA pipeline:
   - Google Chrome `157.0.8103.79` at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome` and Playwright `1.65.2` run headless cleanly via `playwright.sync_api` (`browser.chromium.launch(channel="chrome", headless=True)`) or CLI `--headless=new --disable-gpu --window-size=375,812` without downloading a separate Playwright browser binary.
   - Verify viewports `375x812` and `1440x900` with `0` console errors, `0` failed requests / `404` responses (guaranteed by self-hosted `src/web/fonts/*.woff2`), and `document.documentElement.scrollWidth <= window.innerWidth`.
