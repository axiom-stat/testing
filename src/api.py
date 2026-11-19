"""FastAPI HTTP service and static asset server for CeramScan."""

from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import struct
import threading
import uuid
import zlib
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from src.cv_core import (
    CeramStorage,
    DefectDetector,
    DuplicatePassportError,
    InvalidItemIdError,
    PassportNotFoundError,
    StageType,
    VerdictType,
    extract_fingerprint,
    preprocess_image,
    verify_fingerprint,
)
from src.cv_core.storage import VALID_STAGES, VALID_VERDICTS, _validate_item_id

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 89_478_485
MAX_IMAGE_DIMENSION = 16384
APP_VERSION = "0.1.0"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

_DEMO_SAMPLE_CURATED: tuple[tuple[str, str], ...] = (
    ("SYN-BISQUE-CRACK-01", "Утиль: трещина 900°C"),
    ("SYN-BISQUE-PASS-01", "Утиль: без дефектов"),
    ("SYN-GLAZED-PASS-01", "Глазурь: эталон Гжель"),
    ("SYN-GLAZED-CRACK-01", "Глазурь: цек и блики"),
    ("SYN-GLAZED-BARE-01", "Глазурь: плешина"),
)

_DEMO_PASSPORT_SPECS: dict[str, dict[str, Any]] = {
    "GZHEL-DEMO-01": {
        "item_id": "GZHEL-DEMO-01",
        "label": "Тарелка Гжель (оригинал)",
        "sku": "GZHEL-SKU-01",
        "master_name": "Мастер Гжель",
        "batch_note": "Демо-эталон 289-ФЗ",
        "stage": "glazed",
        "ref_filename": "SYN-GLAZED-PASS-01.png",
        "probe_filename": "SYN-GLAZED-PASS-01.png",
        "is_counterfeit": False,
    },
    "GZHEL-DEMO-02": {
        "item_id": "GZHEL-DEMO-02",
        "label": "Штамповка (контрафакт)",
        "sku": "GZHEL-SKU-02",
        "master_name": "Мастер Гжель",
        "batch_note": "Демо-проверка 289-ФЗ",
        "stage": "glazed",
        "ref_filename": "SYN-GLAZED-PASS-02.png",
        "probe_filename": "SYN-FAKE-HALFTONE-02.png",
        "is_counterfeit": True,
    },
}

_DECODE_ERRORS = (
    UnidentifiedImageError,
    Image.DecompressionBombError,
    EOFError,
    OSError,
    ValueError,
    SyntaxError,
    struct.error,
    zlib.error,
    cv2.error,
)
_STORAGE_ERRORS = (sqlite3.Error, OSError)


def _resolve_env_path(env_name: str, default_path: Path) -> Path:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return default_path
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def _decode_uploaded_image(raw: bytes) -> np.ndarray:
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(raw)) as pil_header:
            hdr_w, hdr_h = pil_header.size
    except _DECODE_ERRORS as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid image payload: unsupported or corrupted image format.",
        ) from exc

    if min(hdr_w, hdr_h) < 32:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions ({hdr_w}x{hdr_h}) are below minimum 32x32.",
        )
    if (hdr_w * hdr_h) > MAX_IMAGE_PIXELS or max(hdr_w, hdr_h) > MAX_IMAGE_DIMENSION:
        raise HTTPException(
            status_code=400,
            detail="Image dimensions exceed maximum allowed pixel limit.",
        )

    arr = np.frombuffer(raw, dtype=np.uint8)
    try:
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error:
        img_bgr = None

    if img_bgr is None:
        try:
            with Image.open(io.BytesIO(raw)) as pil_img:
                rgb = np.asarray(pil_img.convert("RGB"), dtype=np.uint8)
        except _DECODE_ERRORS as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid image payload: unsupported or corrupted image format.",
            ) from exc
        if rgb.size == 0 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise HTTPException(status_code=400, detail="Decoded image has unsupported shape.")
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if img_bgr.size == 0 or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise HTTPException(status_code=400, detail="Decoded image must be 3-channel BGR.")

    height, width = img_bgr.shape[:2]
    if min(height, width) < 32:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions ({width}x{height}) are below minimum 32x32.",
        )
    return img_bgr


async def _read_validated_upload(file: UploadFile) -> np.ndarray:
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded file exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    return _decode_uploaded_image(raw)


def _load_demo_catalog(
    synthetic_dir: Path,
    demo_assets_dir: Path,
    using_synthetic: bool,
) -> dict[str, list[dict[str, Any]]]:
    samples: list[dict[str, Any]] = []
    manifest_path = synthetic_dir / "manifest.json"

    if using_synthetic and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        by_id = {
            str(item.get("id", "")): item
            for item in manifest.get("samples", [])
            if isinstance(item, dict)
        }
        for sample_id, label in _DEMO_SAMPLE_CURATED:
            entry = by_id.get(sample_id)
            if entry is None:
                continue
            filename = Path(str(entry.get("image_path", ""))).name
            if not filename or not (demo_assets_dir / filename).is_file():
                continue
            stage_val = str(entry.get("stage", "glazed"))
            if stage_val not in VALID_STAGES:
                stage_val = "glazed"
            samples.append(
                {
                    "id": sample_id,
                    "label": label,
                    "stage": stage_val,
                    "image_url": f"/demo-assets/{filename}",
                }
            )

    if not samples and demo_assets_dir.is_dir():
        fallback_files = sorted(
            p for p in demo_assets_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        for idx, file_path in enumerate(fallback_files[:5]):
            stage_val = "bisque" if idx == 0 else "glazed"
            samples.append(
                {
                    "id": file_path.stem,
                    "label": f"Образец {file_path.stem}",
                    "stage": stage_val,
                    "image_url": f"/demo-assets/{file_path.name}",
                }
            )

    passports: list[dict[str, Any]] = []
    for spec in _DEMO_PASSPORT_SPECS.values():
        ref_exists = (demo_assets_dir / str(spec["ref_filename"])).is_file()
        probe_exists = (demo_assets_dir / str(spec["probe_filename"])).is_file()
        if ref_exists and probe_exists:
            passports.append(
                {
                    "item_id": spec["item_id"],
                    "label": spec["label"],
                    "sku": spec["sku"],
                    "master_name": spec["master_name"],
                    "probe_image_url": f"/demo-assets/{spec['probe_filename']}",
                    "is_counterfeit": bool(spec["is_counterfeit"]),
                }
            )

    return {"samples": samples, "passports": passports}


def create_app() -> FastAPI:
    """Create and configure the CeramScan FastAPI application."""
    db_path = _resolve_env_path("CERAMSCAN_DB_PATH", PROJECT_ROOT / "data" / "ceramscan.db")
    overlay_dir = _resolve_env_path(
        "CERAMSCAN_OVERLAY_DIR", PROJECT_ROOT / "reports" / "overlays" / "runtime"
    )
    overlays_root_dir = PROJECT_ROOT / "reports" / "overlays"
    qr_dir = _resolve_env_path("CERAMSCAN_QR_DIR", PROJECT_ROOT / "reports" / "qr")
    passport_dir = _resolve_env_path("CERAMSCAN_PASSPORT_DIR", PROJECT_ROOT / "reports" / "passports")
    web_dir = PROJECT_ROOT / "src" / "web"

    synthetic_dir = PROJECT_ROOT / "data" / "synthetic"
    synthetic_images_dir = synthetic_dir / "images"
    reference_samples_dir = PROJECT_ROOT / "data" / "reference_materials" / "ceramic_defect_samples"
    has_synthetic = synthetic_images_dir.is_dir() and any(synthetic_images_dir.glob("*.png"))
    demo_assets_dir = synthetic_images_dir if has_synthetic else reference_samples_dir

    weights_env = os.environ.get("CERAMSCAN_WEIGHTS_PATH", "").strip()
    weights_path: Path | None = None
    if weights_env:
        candidate = Path(weights_env)
        weights_path = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)

    for directory in (
        overlay_dir,
        overlays_root_dir,
        qr_dir,
        passport_dir,
        db_path.parent,
        demo_assets_dir,
        web_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    detector = DefectDetector(weights_path=weights_path)
    model_mode: Literal["deterministic", "onnx"] = "onnx" if weights_path is not None else "deterministic"

    storage_lock = threading.Lock()
    storage_holder: dict[str, CeramStorage | None] = {"instance": None}

    def _get_storage() -> CeramStorage:
        existing = storage_holder["instance"]
        if existing is not None:
            return existing
        with storage_lock:
            if storage_holder["instance"] is None:
                storage_holder["instance"] = CeramStorage(
                    db_path=db_path,
                    qr_dir=qr_dir,
                    passport_ref_dir=passport_dir,
                )
            return cast(CeramStorage, storage_holder["instance"])

    def _ensure_demo_passport(storage: CeramStorage, item_id: str) -> dict[str, Any]:
        spec = _DEMO_PASSPORT_SPECS.get(item_id)
        if spec is None:
            raise PassportNotFoundError(f"Passport '{item_id}' not found.")
        ref_path = demo_assets_dir / str(spec["ref_filename"])
        if not ref_path.is_file():
            raise PassportNotFoundError(f"Demo reference asset missing for '{item_id}'.")

        ref_bgr = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if ref_bgr is None or ref_bgr.size == 0:
            raise PassportNotFoundError(f"Failed to read demo reference image for '{item_id}'.")

        stage_val = cast(StageType, spec["stage"])
        frame = preprocess_image(ref_bgr, stage=stage_val)
        signature = extract_fingerprint(frame)
        try:
            storage.register_passport(
                signature=signature,
                item_id=item_id,
                sku=str(spec["sku"]),
                master_name=str(spec["master_name"]),
                batch_note=str(spec["batch_note"]),
                stage=stage_val,
                reference_image_bgr=frame.roi_bgr,
            )
        except DuplicatePassportError:
            pass
        return storage.get_passport(item_id)

    application = FastAPI(title="CeramScan API", version=APP_VERSION)
    application.state.db_path = db_path
    application.state.overlay_dir = overlay_dir
    application.state.qr_dir = qr_dir
    application.state.passport_dir = passport_dir
    application.state.demo_assets_dir = demo_assets_dir

    @application.get("/")
    async def serve_index() -> Response:
        index_path = web_dir / "index.html"
        if not index_path.is_file():
            return JSONResponse(status_code=404, content={"detail": "index.html not found"})
        return FileResponse(index_path)

    @application.get("/favicon.ico")
    async def favicon() -> Response:
        return Response(status_code=204)

    @application.get("/health")
    async def health() -> dict[str, Any]:
        try:
            ok = await run_in_threadpool(lambda: _get_storage().health_check())
        except _STORAGE_ERRORS as exc:
            raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc
        if not ok:
            raise HTTPException(status_code=500, detail="Database health check failed.")
        return {
            "status": "ok",
            "version": APP_VERSION,
            "model_mode": model_mode,
            "db_ok": True,
        }

    @application.get("/api/demo-samples")
    async def demo_samples() -> dict[str, list[dict[str, Any]]]:
        return _load_demo_catalog(
            synthetic_dir=synthetic_dir,
            demo_assets_dir=demo_assets_dir,
            using_synthetic=has_synthetic,
        )

    @application.post("/api/inspect")
    async def inspect_ceramic(
        file: UploadFile = File(...),
        stage: StageType = Form(default="glazed"),
        item_id: str | None = Form(default=None),
        shift: str = Form(default="S1"),
        reference_diameter_mm: float = Form(default=200.0, gt=0.0, allow_inf_nan=False),
        save_to_db: bool = Form(default=True),
    ) -> dict[str, Any]:
        norm_item_id: str | None = None
        if item_id is not None and item_id.strip() != "":
            try:
                norm_item_id = _validate_item_id(item_id)
            except InvalidItemIdError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        image_bgr = await _read_validated_upload(file)

        try:
            result = await run_in_threadpool(
                detector.inspect,
                image_bgr,
                stage,
                reference_diameter_mm,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        filename = f"insp_{uuid.uuid4().hex[:12]}.png"
        overlay_file = (overlay_dir.resolve() / filename).resolve()
        if not overlay_file.is_relative_to(overlay_dir.resolve()):
            raise HTTPException(status_code=500, detail="Invalid overlay output path.")

        try:
            await run_in_threadpool(overlay_file.write_bytes, result.overlay_png_bytes)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Failed to save overlay PNG: {exc}") from exc

        overlay_url = f"/overlays/runtime/{filename}"
        overlay_base64 = base64.b64encode(result.overlay_png_bytes).decode("ascii")

        inspection_id: int | None = None
        if save_to_db:
            try:
                inspection_id = await run_in_threadpool(
                    lambda: _get_storage().save_inspection(
                        result=result,
                        overlay_path=overlay_url,
                        item_id=norm_item_id,
                        shift=shift,
                    )
                )
            except InvalidItemIdError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except _STORAGE_ERRORS as exc:
                raise HTTPException(status_code=500, detail=f"Database write error: {exc}") from exc

        return {
            "inspection_id": inspection_id,
            "item_id": norm_item_id,
            "stage": result.stage,
            "verdict": result.verdict,
            "gost_grade": result.gost_grade,
            "model_mode": result.model_mode,
            "contour_deviation_pct": float(result.contour_deviation_pct),
            "defect_count": result.defect_count,
            "defects": [
                {
                    "defect_class": d.defect_class.value,
                    "gost_name_ru": d.gost_name_ru,
                    "severity": d.severity,
                    "bbox_xywh": list(d.bbox_xywh),
                    "area_px": d.area_px,
                    "area_mm2": d.area_mm2,
                    "length_mm": d.length_mm,
                    "confidence": d.confidence,
                    "zone": d.zone,
                    "source": d.source,
                }
                for d in result.defects
            ],
            "gost_rules_triggered": list(result.gost_rules_triggered),
            "metrics": {
                "latency_ms": float(result.latency_ms),
                "glare_ratio": float(result.glare_ratio),
                "cobalt_ratio": float(result.cobalt_ratio),
                "scale_calibrated": bool(result.scale_calibrated),
            },
            "overlay_url": overlay_url,
            "overlay_base64": overlay_base64,
        }

    @application.post("/api/passport/register")
    async def register_passport(
        file: UploadFile = File(...),
        item_id: str | None = Form(default=None),
        sku: str = Form(default="GZHEL-SKU-01"),
        master_name: str = Form(default="Мастер Гжель"),
        batch_note: str = Form(default=""),
        stage: StageType = Form(default="glazed"),
    ) -> dict[str, Any]:
        norm_item_id: str | None = None
        if item_id is not None and item_id.strip() != "":
            try:
                norm_item_id = _validate_item_id(item_id)
            except InvalidItemIdError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        image_bgr = await _read_validated_upload(file)

        def _register_sync() -> dict[str, Any]:
            frame = preprocess_image(image_bgr, stage=stage)
            signature = extract_fingerprint(frame)
            return _get_storage().register_passport(
                signature=signature,
                item_id=norm_item_id,
                sku=sku,
                master_name=master_name,
                batch_note=batch_note,
                stage=stage,
                reference_image_bgr=frame.roi_bgr,
            )

        try:
            reg = await run_in_threadpool(_register_sync)
        except DuplicatePassportError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (InvalidItemIdError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (*_STORAGE_ERRORS, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=f"Passport registration failed: {exc}") from exc

        resolved_id = str(reg["item_id"])
        return {
            "item_id": resolved_id,
            "sku": reg["sku"],
            "master_name": reg["master_name"],
            "batch_note": reg["batch_note"],
            "stage": reg["stage"],
            "qr_payload": reg["qr_payload"],
            "qr_image_url": f"/qr/{resolved_id}.png",
            "reference_image_url": f"/passports/{resolved_id}.png",
            "orb_count": reg["orb_count"],
            "cobalt_ratio": reg["cobalt_ratio"],
            "created_at": reg["created_at"],
        }

    @application.post("/api/passport/verify/{item_id}")
    async def verify_passport(
        item_id: str,
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        try:
            valid_id = _validate_item_id(item_id)
        except InvalidItemIdError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        probe_bgr = await _read_validated_upload(file)

        def _verify_sync() -> tuple[dict[str, Any], Any]:
            storage = _get_storage()
            try:
                passport = storage.get_passport(valid_id)
            except PassportNotFoundError:
                passport = _ensure_demo_passport(storage, valid_id)

            passport_stage = cast(StageType, passport["stage"])
            probe_frame = preprocess_image(probe_bgr, stage=passport_stage)
            ver_result = verify_fingerprint(passport["signature"], probe_frame)
            storage.record_verification(valid_id, ver_result)
            return passport, ver_result

        try:
            passport, ver_result = await run_in_threadpool(_verify_sync)
        except PassportNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (InvalidItemIdError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (*_STORAGE_ERRORS, RuntimeError) as exc:
            raise HTTPException(status_code=500, detail=f"Passport verification failed: {exc}") from exc

        return {
            "item_id": valid_id,
            "verdict": ver_result.verdict,
            "match_percent": float(ver_result.match_percent),
            "orb_score": float(ver_result.orb_score),
            "phash_score": float(ver_result.phash_score),
            "color_score": float(ver_result.color_score),
            "matched_kp": int(ver_result.matched_kp),
            "latency_ms": float(ver_result.latency_ms),
            "registered_sku": passport["sku"],
            "registered_master": passport["master_name"],
            "registered_batch_note": passport["batch_note"],
            "reference_image_url": f"/passports/{valid_id}.png",
        }

    @application.get("/api/analytics")
    async def analytics(
        limit: int = Query(default=25, ge=1, le=200),
        stage: str | None = Query(default=None),
        verdict: str | None = Query(default=None),
    ) -> dict[str, Any]:
        norm_stage: StageType | None = None
        if stage is not None and stage.strip() != "":
            if stage not in VALID_STAGES:
                raise HTTPException(status_code=422, detail=f"Invalid stage filter: {stage}")
            norm_stage = cast(StageType, stage)

        norm_verdict: VerdictType | None = None
        if verdict is not None and verdict.strip() != "":
            if verdict not in VALID_VERDICTS:
                raise HTTPException(status_code=422, detail=f"Invalid verdict filter: {verdict}")
            norm_verdict = cast(VerdictType, verdict)

        try:
            return await run_in_threadpool(
                lambda: _get_storage().get_analytics(
                    limit=limit,
                    stage=norm_stage,
                    verdict=norm_verdict,
                )
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except _STORAGE_ERRORS as exc:
            raise HTTPException(status_code=500, detail=f"Analytics query failed: {exc}") from exc

    # Mount order is critical: /overlays/runtime must precede /overlays.
    application.mount("/static", StaticFiles(directory=str(web_dir)), name="static")
    application.mount("/overlays/runtime", StaticFiles(directory=str(overlay_dir)), name="overlays-runtime")
    application.mount("/overlays", StaticFiles(directory=str(overlays_root_dir)), name="overlays")
    application.mount("/qr", StaticFiles(directory=str(qr_dir)), name="qr")
    application.mount("/passports", StaticFiles(directory=str(passport_dir)), name="passports")
    application.mount("/demo-assets", StaticFiles(directory=str(demo_assets_dir)), name="demo-assets")

    return application


app = create_app()
