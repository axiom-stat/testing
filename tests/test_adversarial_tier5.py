"""Tier 5 white-box adversarial and regression tests for CeramScan."""

from __future__ import annotations

import concurrent.futures
import io
import json
import math
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from PIL import Image

from scripts.run_benchmark import run_benchmark
from src.api import MAX_UPLOAD_BYTES, PROJECT_ROOT
from src.cv_core.defect_detector import DefectDetector
from src.cv_core.fingerprint import (
    VERDICT_AUTHENTIC,
    VERDICT_COUNTERFEIT,
    extract_fingerprint,
    verify_fingerprint,
)
from src.cv_core.preprocessing import DefectClass, PreprocessConfig, preprocess_image
from src.cv_core.storage import CeramStorage, InvalidItemIdError


def _pb_varint(val: int) -> bytes:
    out = bytearray()
    while True:
        byte = val & 0x7F
        val >>= 7
        if val:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            break
    return bytes(out)


def _pb_field_varint(field_num: int, val: int) -> bytes:
    return _pb_varint((field_num << 3) | 0) + _pb_varint(val)


def _pb_field_bytes(field_num: int, data: bytes) -> bytes:
    return _pb_varint((field_num << 3) | 2) + _pb_varint(len(data)) + data


def _pb_value_info(name: str, elem_type: int, dims: list[int]) -> bytes:
    shape_msg = b"".join(_pb_field_bytes(1, _pb_field_varint(1, d)) for d in dims)
    tensor_type = _pb_field_varint(1, elem_type) + _pb_field_bytes(2, shape_msg)
    type_proto = _pb_field_bytes(1, tensor_type)
    return _pb_field_bytes(1, name.encode("utf-8")) + _pb_field_bytes(2, type_proto)


def _pb_tensor_proto(name: str, arr: np.ndarray) -> bytes:
    raw = arr.astype(np.float32).tobytes()
    parts = [_pb_field_varint(1, dim) for dim in arr.shape]
    parts.append(_pb_field_varint(2, 1))
    parts.append(_pb_field_bytes(8, name.encode("utf-8")))
    parts.append(_pb_field_bytes(9, raw))
    return b"".join(parts)


def _build_onnx_bytes(
    output_array: np.ndarray,
    metadata: dict[str, str] | None = None,
) -> bytes:
    tensor_proto = _pb_tensor_proto("const_val", output_array)
    attr_proto = (
        _pb_field_bytes(1, b"value")
        + _pb_field_bytes(5, tensor_proto)
        + _pb_field_varint(20, 4)
    )
    node_proto = (
        _pb_field_bytes(2, b"output0")
        + _pb_field_bytes(3, b"ConstNode")
        + _pb_field_bytes(4, b"Constant")
        + _pb_field_bytes(5, attr_proto)
    )
    graph_proto = (
        _pb_field_bytes(1, node_proto)
        + _pb_field_bytes(2, b"CeramScanTier5Graph")
        + _pb_field_bytes(11, _pb_value_info("images", 1, [1, 3, 640, 640]))
        + _pb_field_bytes(12, _pb_value_info("output0", 1, list(output_array.shape)))
    )
    parts = [
        _pb_field_varint(1, 7),
        _pb_field_bytes(2, b"ceramscan-tier5"),
        _pb_field_bytes(7, graph_proto),
        _pb_field_bytes(8, _pb_field_varint(2, 13)),
    ]
    if metadata:
        for k, v in metadata.items():
            entry = _pb_field_bytes(1, k.encode("utf-8")) + _pb_field_bytes(2, v.encode("utf-8"))
            parts.append(_pb_field_bytes(14, entry))
    return b"".join(parts)


def _make_cobalt_plate(
    size: int = 640,
    cobalt: bool = True,
    chip: bool = False,
    speck: bool = False,
) -> np.ndarray:
    img = np.full((size, size, 3), (30, 32, 36), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = int(size * 0.375)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    if chip:
        cv2.circle(mask, (cx + radius - 4, cy), 18, 0, -1)

    plate = np.full((size, size, 3), (218, 222, 226), dtype=np.uint8)
    if cobalt:
        dark = (176, 54, 20)
        mid = (206, 92, 46)
        cv2.circle(plate, (cx, cy), int(radius * 0.73), mid, 3, cv2.LINE_AA)
        for idx in range(8):
            ang = idx * (2.0 * math.pi / 8.0)
            px = int(cx + radius * 0.41 * math.cos(ang))
            py = int(cy + radius * 0.41 * math.sin(ang))
            deg = int(round(math.degrees(ang)))
            cv2.ellipse(plate, (px, py), (34, 15), deg, 0, 360, dark, -1, cv2.LINE_AA)
            vx = int(cx + radius * 0.49 * math.cos(ang))
            vy = int(cy + radius * 0.49 * math.sin(ang))
            cv2.line(plate, (cx, cy), (vx, vy), (218, 222, 226), 1, cv2.LINE_AA)
            dx = int(cx + radius * 0.63 * math.cos(ang + 0.16))
            dy = int(cy + radius * 0.63 * math.sin(ang + 0.16))
            cv2.circle(plate, (dx, dy), 6, mid, -1, cv2.LINE_AA)
            lx = int(cx + radius * 0.24 * math.cos(ang + 0.25))
            ly = int(cy + radius * 0.24 * math.sin(ang + 0.25))
            cv2.ellipse(plate, (lx, ly), (16, 6), deg + 35, 0, 360, dark, -1, cv2.LINE_AA)
        cv2.circle(plate, (cx, cy), int(radius * 0.13), dark, 4, cv2.LINE_AA)
    if speck:
        cv2.circle(plate, (cx + 15, cy - 20), 2, (14, 14, 16), -1)
    img[mask > 0] = plate[mask > 0]
    return img


def _make_plate_png(
    *,
    stage: str = "glazed",
    cobalt: bool = True,
    crack: bool = False,
    size: int = 320,
) -> bytes:
    img = np.full((size, size, 3), (28, 28, 30), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = int(size * 0.36)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)
    base = (176, 178, 182) if stage == "bisque" else (218, 220, 224)
    plate = np.full((size, size, 3), base, dtype=np.uint8)
    if cobalt:
        for idx in range(10):
            theta = idx * (2.0 * np.pi / 10.0)
            px = int(cx + radius * 0.44 * np.cos(theta))
            py = int(cy + radius * 0.44 * np.sin(theta))
            cv2.ellipse(plate, (px, py), (22, 10), int(np.degrees(theta)), 0, 360, (186, 54, 22), -1)
    if crack:
        cv2.line(plate, (cx - radius // 2, cy), (cx + radius // 2, cy + 8), (16, 16, 18), 4)
    img[mask > 0] = plate[mask > 0]
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture()
def live_isolated_url(tmp_path: Path) -> Iterator[tuple[str, int]]:
    port = _find_free_port()
    env = os.environ.copy()
    env["CERAMSCAN_DB_PATH"] = str(tmp_path / "adv.db")
    env["CERAMSCAN_OVERLAY_DIR"] = str(tmp_path / "overlays" / "runtime")
    env["CERAMSCAN_QR_DIR"] = str(tmp_path / "qr")
    env["CERAMSCAN_PASSPORT_DIR"] = str(tmp_path / "passports")
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env.pop("CERAMSCAN_WEIGHTS_PATH", None)
    for sub in ("overlays/runtime", "qr", "passports"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)

    log_path = tmp_path / "uvicorn.log"
    log_file = log_path.open("w+", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 15.0
    try:
        with httpx.Client(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"uvicorn exited early:\n{log_path.read_text(encoding='utf-8')}")
                try:
                    if client.get(f"{base_url}/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.15)
            else:
                raise TimeoutError("uvicorn did not become healthy")
        yield base_url, port
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        log_file.close()


# ---------------------------------------------------------------------------
# Part A: CV Core, Storage & CLI Adversarial & Regression Tests (G1..G8)
# ---------------------------------------------------------------------------


def test_preprocess_and_detector_extreme_aspect_colors_and_slices() -> None:
    """Verify extreme aspect ratios, uniform colors, and non-contiguous slices never crash."""
    detector = DefectDetector()
    cases = (
        (np.full((32, 1000, 3), 180, dtype=np.uint8), "glazed"),
        (np.full((1000, 32, 3), 180, dtype=np.uint8), "bisque"),
        (np.zeros((640, 640, 3), dtype=np.uint8), "glazed"),
        (np.full((640, 640, 3), 255, dtype=np.uint8), "glazed"),
        (np.full((640, 640, 3), 255, dtype=np.uint8), "bisque"),
        (np.full((640, 640, 3), (220, 40, 10), dtype=np.uint8), "glazed"),
        (_make_cobalt_plate()[:, ::-1, :], "glazed"),
        (_make_cobalt_plate()[::2, ::2, :], "bisque"),
    )
    for img, stage in cases:
        res = detector.inspect(img, stage=stage)  # type: ignore[arg-type]
        assert res.overlay_bgr.shape == (640, 640, 3)
        assert res.overlay_png_bytes.startswith(b"\x89PNG\r\n\x1a\n")

    ro_img = _make_cobalt_plate().copy()
    ro_img.flags.writeable = False
    assert detector.inspect(ro_img, stage="glazed").verdict == "PASS"


def test_high_res_2400x1800_circular_plate_downsample_and_cpu_sla() -> None:
    """Verify 2400x1800 plate downsamples original_bgr <= 1600 and completes under 300 ms."""
    detector = DefectDetector()
    _ = detector.inspect(_make_cobalt_plate(), stage="glazed")

    big = np.full((1800, 2400, 3), (30, 32, 36), dtype=np.uint8)
    cv2.circle(big, (1200, 900), 680, (218, 222, 226), -1)
    cv2.circle(big, (1200, 900), 320, (176, 54, 20), 18)

    frame = preprocess_image(big, stage="glazed")
    assert max(frame.original_bgr.shape[:2]) <= 1600

    res = detector.inspect(big, stage="glazed")
    assert res.verdict == "PASS"
    assert res.latency_ms < 300.0


def test_invalid_stage_rejected() -> None:
    """Verify invalid stage raises ValueError across preprocess_image and DefectDetector."""
    plate = _make_cobalt_plate()
    with pytest.raises(ValueError):
        preprocess_image(plate, stage="raw")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        DefectDetector().inspect(plate, stage="raw")  # type: ignore[arg-type]


def test_onnx_zero_detections_and_malformed_names_fallback(tmp_path: Path) -> None:
    """Verify ONNX empty predictions and malformed names metadata handle safely."""
    plate = _make_cobalt_plate()

    for tag, arr in (
        ("v8_empty", np.zeros((1, 10, 0), dtype=np.float32)),
        ("v5_empty", np.zeros((1, 0, 11), dtype=np.float32)),
    ):
        p = tmp_path / f"{tag}.onnx"
        p.write_bytes(_build_onnx_bytes(arr))
        res = DefectDetector(weights_path=p).inspect(plate, stage="glazed")
        assert res.model_mode == "onnx"
        assert res.defect_count == 0
        assert res.verdict == "PASS"

    v8_one = np.zeros((1, 10, 4), dtype=np.float32)
    v8_one[0, 0:4, 0] = [320.0, 320.0, 60.0, 20.0]
    v8_one[0, 4, 0] = 0.90
    p_mal = tmp_path / "malformed_meta.onnx"
    p_mal.write_bytes(_build_onnx_bytes(v8_one, metadata={"names": "{bad_json"}))
    res_mal = DefectDetector(weights_path=p_mal).inspect(plate, stage="glazed")
    assert res_mal.verdict == "REJECT"
    assert res_mal.defects[0].defect_class == DefectClass.CRACK


def test_onnx_partial_nan_scores_and_empty_names_regression(tmp_path: Path) -> None:
    """Verify G1/G2: partial NaN class scores do not drop detections and empty/bad names fallback safely."""
    plate = _make_cobalt_plate()

    arr_nan = np.zeros((1, 10, 4), dtype=np.float32)
    arr_nan[0, 0:4, 0] = [320.0, 320.0, 80.0, 20.0]
    arr_nan[0, 4, 0] = 0.95
    arr_nan[0, 5, 0] = np.nan
    p_nan = tmp_path / "partial_nan.onnx"
    p_nan.write_bytes(_build_onnx_bytes(arr_nan))
    res_nan = DefectDetector(weights_path=p_nan).inspect(plate, stage="glazed")
    assert res_nan.verdict == "REJECT"
    assert any(d.defect_class == DefectClass.CRACK for d in res_nan.defects)

    arr_empty_meta = np.zeros((1, 10, 4), dtype=np.float32)
    arr_empty_meta[0, 0:4, 0] = [320.0, 320.0, 60.0, 20.0]
    arr_empty_meta[0, 4, 0] = 0.90
    for bad_names in ("{}", "[]", "{'crack': 0}"):
        p_meta = tmp_path / f"meta_{abs(hash(bad_names))}.onnx"
        p_meta.write_bytes(_build_onnx_bytes(arr_empty_meta, metadata={"names": bad_names}))
        res_meta = DefectDetector(weights_path=p_meta).inspect(plate, stage="glazed")
        assert res_meta.verdict == "REJECT"


def test_onnx_nonfinite_boxes_inf_scores_and_noncontiguous_names(tmp_path: Path) -> None:
    """Verify G1/G2: non-finite boxes and Inf scores are rejected and non-contiguous names map safely."""
    plate = _make_cobalt_plate()

    arr = np.zeros((1, 10, 4), dtype=np.float32)
    arr[0, 0:4, 0] = [np.nan, 320.0, 50.0, 50.0]
    arr[0, 4, 0] = 0.95
    arr[0, 0:4, 1] = [np.inf, 320.0, 50.0, 50.0]
    arr[0, 4, 1] = 0.95
    arr[0, 0:4, 2] = [320.0, 320.0, 60.0, 20.0]
    arr[0, 4, 2] = np.inf
    p_nonfin = tmp_path / "nonfin.onnx"
    p_nonfin.write_bytes(_build_onnx_bytes(arr))
    res_nonfin = DefectDetector(weights_path=p_nonfin).inspect(plate, stage="glazed")
    assert res_nonfin.verdict == "PASS"
    assert res_nonfin.defect_count == 0

    arr_clip = np.zeros((1, 10, 2), dtype=np.float32)
    arr_clip[0, 0:4, 0] = [320.0, 320.0, 60.0, 20.0]
    arr_clip[0, 4, 0] = 1.4
    p_clip = tmp_path / "over_one.onnx"
    p_clip.write_bytes(_build_onnx_bytes(arr_clip))
    res_clip = DefectDetector(weights_path=p_clip).inspect(plate, stage="glazed")
    assert res_clip.defect_count == 1
    assert 0.0 <= res_clip.defects[0].confidence <= 1.0

    arr_noncontig = np.zeros((1, 6, 2), dtype=np.float32)
    arr_noncontig[0, 0:4, 0] = [320.0, 320.0, 40.0, 40.0]
    arr_noncontig[0, 5, 0] = 0.92
    p_nc = tmp_path / "noncontig.onnx"
    p_nc.write_bytes(_build_onnx_bytes(arr_noncontig, metadata={"names": "{10: 'crack', 11: 'chip'}"}))
    res_nc = DefectDetector(weights_path=p_nc).inspect(plate, stage="glazed")
    assert res_nc.verdict == "REJECT"
    assert res_nc.defects[0].defect_class == DefectClass.CHIP


def test_onnx_one_based_names_mapping_regression(tmp_path: Path) -> None:
    """Verify 0-based, 1-based overlapping, and shifted ONNX names map positional columns cleanly."""
    plate = _make_cobalt_plate()
    arr = np.zeros((1, 6, 2), dtype=np.float32)
    arr[0, 0:4, 0] = [300.0, 320.0, 40.0, 40.0]
    arr[0, 5, 0] = 0.92
    arr[0, 0:4, 1] = [180.0, 220.0, 50.0, 18.0]
    arr[0, 4, 1] = 0.88

    for tag, names_meta in (
        ("zero_based", "{0: 'crack', 1: 'chip'}"),
        ("one_based", "{1: 'crack', 2: 'chip'}"),
        ("shifted", "{10: 'crack', 11: 'chip'}"),
    ):
        p = tmp_path / f"{tag}.onnx"
        p.write_bytes(_build_onnx_bytes(arr, metadata={"names": names_meta}))
        res = DefectDetector(weights_path=p).inspect(plate, stage="glazed")
        assert res.verdict == "REJECT"
        classes = {d.defect_class for d in res.defects}
        assert classes == {DefectClass.CRACK, DefectClass.CHIP}, f"{tag} mapped to {classes}"


def test_fingerprint_rotation_scale_crop_collinear_and_monochrome_swap() -> None:
    """Verify fingerprint withstands rotation/scale/crop while rejecting collinear and monochrome swaps."""
    cobalt = _make_cobalt_plate()
    c_frame = preprocess_image(cobalt, stage="glazed")
    c_sig = extract_fingerprint(c_frame)

    for angle in (15.0, 45.0, 90.0):
        rot = cv2.getRotationMatrix2D((320.0, 320.0), angle, 1.0)
        warped = cv2.warpAffine(cobalt, rot, (640, 640), borderValue=(30, 32, 36))
        res_rot = verify_fingerprint(c_sig, preprocess_image(warped, stage="glazed"))
        assert res_rot.verdict == VERDICT_AUTHENTIC

    for scale in (0.7, 0.85, 1.15):
        rot = cv2.getRotationMatrix2D((320.0, 320.0), 0.0, scale)
        scaled = cv2.warpAffine(cobalt, rot, (640, 640), borderValue=(30, 32, 36))
        res_sc = verify_fingerprint(c_sig, preprocess_image(scaled, stage="glazed"))
        assert res_sc.verdict == VERDICT_AUTHENTIC

    crop = cobalt[120:520, 120:520]
    res_crop = verify_fingerprint(c_sig, preprocess_image(crop, stage="glazed"))
    assert res_crop.verdict == VERDICT_AUTHENTIC

    collinear_kps = [[float(100 + i), 320.0] for i in range(c_sig.orb_count)]
    collinear_sig = replace(c_sig, keypoints_xy=collinear_kps)
    res_col = verify_fingerprint(collinear_sig, c_frame)
    assert res_col.matched_kp == 0
    assert res_col.verdict == VERDICT_COUNTERFEIT

    gray_ring = np.full((640, 640, 3), (30, 32, 36), dtype=np.uint8)
    cv2.circle(gray_ring, (320, 320), 240, (218, 222, 226), -1)
    cv2.circle(gray_ring, (320, 320), 140, (60, 60, 60), 10)
    res_mono = verify_fingerprint(c_sig, preprocess_image(gray_ring, stage="glazed"))
    assert res_mono.color_score == 0.0
    assert res_mono.verdict == VERDICT_COUNTERFEIT


def test_fingerprint_truncated_keypoints_and_nan_hist_rejected() -> None:
    """Verify G4: truncated keypoints_xy and NaN or wrong-length cobalt_hist raise ValueError."""
    cobalt = _make_cobalt_plate()
    c_frame = preprocess_image(cobalt, stage="glazed")
    c_sig = extract_fingerprint(c_frame)
    assert c_sig.orb_count >= 12

    trunc_sig = replace(c_sig, keypoints_xy=c_sig.keypoints_xy[:4])
    with pytest.raises(ValueError):
        verify_fingerprint(trunc_sig, c_frame)

    nan_hist = list(c_sig.cobalt_hist)
    nan_hist[0] = float("nan")
    nan_sig = replace(c_sig, cobalt_hist=nan_hist)
    with pytest.raises(ValueError):
        verify_fingerprint(nan_sig, c_frame)

    short_sig = replace(c_sig, cobalt_hist=c_sig.cobalt_hist[:10])
    with pytest.raises(ValueError):
        verify_fingerprint(short_sig, c_frame)


def test_storage_multithread_wal_writes_and_unicode_item_id_guards(tmp_path: Path) -> None:
    """Verify concurrent WAL writes succeed with zero lock errors and hostile IDs fail safely."""
    db_path = tmp_path / "wal.db"
    qr_dir = tmp_path / "qr"
    ref_dir = tmp_path / "passports"
    storage = CeramStorage(db_path=db_path, qr_dir=qr_dir, passport_ref_dir=ref_dir)

    plate = _make_cobalt_plate()
    insp_res = DefectDetector().inspect(plate, stage="glazed")
    frame = preprocess_image(plate, stage="glazed")
    sig = extract_fingerprint(frame)
    ver_res = verify_fingerprint(sig, frame)

    def _worker(idx: int) -> None:
        item_id = f"CONC-{idx:03d}"
        storage.save_inspection(insp_res, overlay_path=f"ov_{idx}.png", item_id=item_id, shift="S1")
        storage.register_passport(sig, item_id=item_id, reference_image_bgr=plate)
        storage.record_verification(item_id, ver_res)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futs = [pool.submit(_worker, i) for i in range(32)]
        for fut in futs:
            fut.result()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM inspections;").fetchone()[0] == 32
        assert conn.execute("SELECT COUNT(*) FROM passports;").fetchone()[0] == 32
        assert conn.execute("SELECT COUNT(*) FROM passport_verifications;").fetchone()[0] == 32

    for bad_id in ("../foo", "A\x00B", "GZHEL-01/", "GZHEL-01\n", "АВС-123", "GZHEL\u200b01"):
        with pytest.raises(InvalidItemIdError):
            storage.register_passport(sig, item_id=bad_id)


def test_storage_register_passport_db_failure_leaves_no_orphan_pngs(tmp_path: Path) -> None:
    """Verify G3: SQLite INSERT failure during register_passport cleans up QR and reference PNGs."""
    db_path = tmp_path / "rollback.db"
    qr_dir = tmp_path / "qr"
    ref_dir = tmp_path / "passports"
    storage = CeramStorage(db_path=db_path, qr_dir=qr_dir, passport_ref_dir=ref_dir)

    plate = _make_cobalt_plate()
    sig = extract_fingerprint(preprocess_image(plate, stage="glazed"))

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TRIGGER abort_ins BEFORE INSERT ON passports "
            "WHEN NEW.item_id = 'ORPHAN-FAIL-01' "
            "BEGIN SELECT RAISE(ABORT, 'simulated failure'); END;"
        )
        conn.commit()

    with pytest.raises(sqlite3.IntegrityError):
        storage.register_passport(sig, item_id="ORPHAN-FAIL-01", reference_image_bgr=plate)

    assert not (qr_dir / "ORPHAN-FAIL-01.png").exists()
    assert not (ref_dir / "ORPHAN-FAIL-01.png").exists()


def test_storage_non_string_item_id_raises_invalid_item_id(tmp_path: Path) -> None:
    """Verify G8: non-string item_id raises InvalidItemIdError instead of AttributeError."""
    storage = CeramStorage(
        db_path=tmp_path / "types.db",
        qr_dir=tmp_path / "qr",
        passport_ref_dir=tmp_path / "passports",
    )
    plate = _make_cobalt_plate()
    insp_res = DefectDetector().inspect(plate, stage="glazed")
    sig = extract_fingerprint(preprocess_image(plate, stage="glazed"))

    with pytest.raises(InvalidItemIdError):
        storage.save_inspection(insp_res, overlay_path="ov.png", item_id=12345)  # type: ignore[arg-type]
    with pytest.raises(InvalidItemIdError):
        storage.register_passport(sig, item_id=12345)  # type: ignore[arg-type]


def test_preprocess_invalid_dtype_diameter_and_target_size_rejected() -> None:
    """Verify G7: non-uint8 dtype, non-positive/NaN diameter, and tiny target_size raise ValueError."""
    plate = _make_cobalt_plate()
    with pytest.raises(ValueError):
        preprocess_image(plate.astype(np.float32), stage="glazed")
    for bad_diam in (0.0, -10.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            preprocess_image(plate, stage="glazed", config=PreprocessConfig(reference_diameter_mm=bad_diam))
        with pytest.raises(ValueError):
            DefectDetector().inspect(plate, stage="glazed", reference_diameter_mm=bad_diam)
    for bad_ts in (0, -10, 16):
        with pytest.raises(ValueError):
            preprocess_image(plate, stage="glazed", config=PreprocessConfig(target_size=bad_ts))


def test_multidefect_severity_order_primary_defect_and_benchmark_recall(tmp_path: Path) -> None:
    """Verify G6: chip + speck plate orders critical defect first and scores chip recall 1.0."""
    img = _make_cobalt_plate(cobalt=False, chip=True, speck=True)
    res = DefectDetector().inspect(img, stage="glazed")
    assert res.verdict == "REJECT"
    assert res.defects[0].defect_class == DefectClass.CHIP

    storage = CeramStorage(
        db_path=tmp_path / "sev.db",
        qr_dir=tmp_path / "qr",
        passport_ref_dir=tmp_path / "passports",
    )
    row_id = storage.save_inspection(res, overlay_path="ov.png", item_id="MULTI-01")
    rows = storage.list_inspections(limit=5)
    assert rows[0]["id"] == row_id
    assert rows[0]["primary_defect"] == "chip"

    img_path = tmp_path / "chip_speck.png"
    cv2.imwrite(str(img_path), img)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stand_label": "multi-defect-test",
                "count": 1,
                "samples": [
                    {
                        "id": "MULTI-CHIP-01",
                        "image_path": str(img_path),
                        "stage": "glazed",
                        "defect_class": "chip",
                        "expected_verdict": "REJECT",
                        "auth_pair_id": None,
                        "is_counterfeit": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = run_benchmark(
        dataset_path=manifest_path,
        output_json=tmp_path / "bench.json",
        output_md=tmp_path / "bench.md",
        overlays_dir=tmp_path / "overlays",
    )
    assert report["per_class"]["chip"]["recall"] == 1.0


def test_cli_stem_collision_and_grant_builders_from_outside_cwd(tmp_path: Path) -> None:
    """Verify cross-category stem collisions and outside-CWD grant builds work cleanly."""
    repo_root = Path(__file__).resolve().parents[1]
    py_bin = Path(sys.executable)

    ws = tmp_path / "ws"
    shutil.copytree(repo_root / "src", ws / "src")
    shutil.copytree(repo_root / "scripts", ws / "scripts")
    photos = ws / "data" / "real_photos"
    for cat in ("pass", "chip", "crack", "glaze", "fake"):
        (photos / cat).mkdir(parents=True, exist_ok=True)

    plate = _make_cobalt_plate()
    cracked = plate.copy()
    cv2.line(cracked, (220, 320), (420, 330), (15, 15, 18), 4)
    cv2.imwrite(str(photos / "pass" / "img.jpg"), plate)
    cv2.imwrite(str(photos / "crack" / "img.jpg"), cracked)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ws)
    proc = subprocess.run(
        [str(py_bin), "scripts/ingest_real_photos.py", "--input", str(photos), "--rebuild-all"],
        cwd=str(ws),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    man = json.loads((photos / "manifest.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in man["samples"]] == ["REAL-PASS-001", "REAL-CRACK-002"]

    iso = tmp_path / "iso_grant"
    shutil.copytree(repo_root / "grant_case", iso / "grant_case")
    shutil.copytree(repo_root / "reports", iso / "reports")
    (iso / "grant_case" / "CeramScan_Sistema_2026.docx").unlink(missing_ok=True)
    (iso / "grant_case" / "CeramScan_Sistema_2026.pptx").unlink(missing_ok=True)

    outside = tmp_path / "outside"
    outside.mkdir()
    assert subprocess.run([str(py_bin), str(iso / "grant_case" / "build_docx.py")], cwd=str(outside), check=False).returncode == 0
    assert subprocess.run([str(py_bin), str(iso / "grant_case" / "build_presentation.py")], cwd=str(outside), check=False).returncode == 0
    assert (iso / "grant_case" / "CeramScan_Sistema_2026.docx").is_file()
    assert (iso / "grant_case" / "CeramScan_Sistema_2026.pptx").is_file()


def test_ingest_corrupt_image_fails_cleanly_and_outside_cwd_anchors_repo_root(tmp_path: Path) -> None:
    """Verify G5: corrupt real photo fails cleanly without manifest and outside-CWD rebuilds in REPO_ROOT."""
    repo_root = Path(__file__).resolve().parents[1]
    py_bin = Path(sys.executable)

    ws_bad = tmp_path / "ws_bad"
    shutil.copytree(repo_root / "src", ws_bad / "src")
    shutil.copytree(repo_root / "scripts", ws_bad / "scripts")
    photos_bad = ws_bad / "data" / "real_photos"
    for cat in ("pass", "chip", "crack", "glaze", "fake"):
        (photos_bad / cat).mkdir(parents=True, exist_ok=True)
    (photos_bad / "pass" / "corrupt.jpg").write_bytes(b"not-an-image")

    env_bad = os.environ.copy()
    env_bad["PYTHONPATH"] = str(ws_bad)
    proc_bad = subprocess.run(
        [str(py_bin), str(ws_bad / "scripts" / "ingest_real_photos.py"), "--input", str(photos_bad)],
        cwd=str(tmp_path),
        env=env_bad,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc_bad.returncode == 1
    assert "Traceback" not in proc_bad.stderr
    assert not (photos_bad / "manifest.json").exists()

    ws_out = tmp_path / "ws_out"
    shutil.copytree(repo_root / "src", ws_out / "src")
    shutil.copytree(repo_root / "scripts", ws_out / "scripts")
    photos_out = ws_out / "data" / "real_photos"
    for cat in ("pass", "chip", "crack", "glaze", "fake"):
        (photos_out / cat).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(photos_out / "pass" / "plate.jpg"), _make_cobalt_plate())

    outside = tmp_path / "outside_cwd"
    outside.mkdir()
    env_out = os.environ.copy()
    env_out["PYTHONPATH"] = str(ws_out)
    proc_out = subprocess.run(
        [str(py_bin), str(ws_out / "scripts" / "ingest_real_photos.py"), "--rebuild-all"],
        cwd=str(outside),
        env=env_out,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert proc_out.returncode == 0, proc_out.stderr
    assert (ws_out / "reports" / "benchmark_results.json").is_file()
    assert not (outside / "reports").exists()


# ---------------------------------------------------------------------------
# Part B: FastAPI & Zero-Build Web UI Adversarial & Regression Tests (B1..B6)
# ---------------------------------------------------------------------------


def test_adv_valid_decompression_bomb_rejected_400(live_isolated_url: tuple[str, int]) -> None:
    """Verify B1: valid 20000x20000 PNG decompression bomb is rejected with HTTP 400."""
    base_url, _port = live_isolated_url
    old_max = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        buf = io.BytesIO()
        Image.new("L", (20000, 20000), 220).save(buf, format="PNG")
        bomb_png = buf.getvalue()
    finally:
        Image.MAX_IMAGE_PIXELS = old_max

    assert len(bomb_png) < MAX_UPLOAD_BYTES
    resp = httpx.post(
        f"{base_url}/api/inspect",
        files={"file": ("bomb_20k.png", bomb_png, "image/png")},
        data={"stage": "glazed", "save_to_db": "false"},
        timeout=15.0,
    )
    assert resp.status_code == 400


def test_adv_upload_exact_10mb_and_polyglot_boundaries(live_isolated_url: tuple[str, int]) -> None:
    """Verify exact 10 MB boundary succeeds while 10 MB + 1 fails 413 and polyglots fail 400."""
    base_url, _port = live_isolated_url
    valid_png = _make_plate_png(stage="glazed", cobalt=True)
    exact_10mb = valid_png + b"\x00" * (MAX_UPLOAD_BYTES - len(valid_png))
    over_10mb = exact_10mb + b"\x00"

    buf_31 = io.BytesIO()
    Image.new("RGB", (31, 31), (200, 200, 200)).save(buf_31, format="PNG")

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        assert client.post(
            "/api/inspect",
            files={"file": ("exact.png", exact_10mb, "image/png")},
            data={"stage": "glazed", "save_to_db": "false"},
        ).status_code == 200

        assert client.post(
            "/api/inspect",
            files={"file": ("over.png", over_10mb, "image/png")},
            data={"stage": "glazed", "save_to_db": "false"},
        ).status_code == 413

        for fname, payload in (
            ("polyglot.png", b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'),
            ("html.png", b"<html><body><script>alert(1)</script></body></html>"),
            ("onebyte.png", b"\x89"),
            ("31x31.png", buf_31.getvalue()),
        ):
            assert client.post(
                "/api/inspect",
                files={"file": (fname, payload, "image/png")},
                data={"stage": "glazed", "save_to_db": "false"},
            ).status_code == 400


def test_adv_raw_path_traversal_static_and_endpoints(live_isolated_url: tuple[str, int]) -> None:
    """Verify raw socket path traversal and encoded dot-segment probes fail 400/404."""
    base_url, port = live_isolated_url
    valid_png = _make_plate_png(stage="glazed", cobalt=True)

    for raw_path in ("/static/../api.py", "/overlays/runtime/../../adv.db", "/static/..%2fapi.py"):
        with socket.create_connection(("127.0.0.1", port), timeout=3.0) as sock:
            req = f"GET {raw_path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
            sock.sendall(req.encode("ascii"))
            status_line = sock.recv(256).decode("utf-8", errors="replace").splitlines()[0]
            assert "404" in status_line or "400" in status_line

    with httpx.Client(base_url=base_url, timeout=10.0) as client:
        for url in ("/qr/..%2Fadv.db", "/passports/..%2Fadv.db", "/demo-assets/..%2F..%2Fsrc%2Fapi.py"):
            assert client.get(url).status_code in (400, 404)
        for bad_id in ("..%2F..%2Fetc%2Fpasswd", "%2e%2e%2fetc", "GZHEL-01%00.png"):
            resp = client.post(
                f"/api/passport/verify/{bad_id}",
                files={"file": ("p.png", valid_png, "image/png")},
            )
            assert resp.status_code in (400, 404)


def test_adv_concurrent_register_demo_verify_and_wal_rw(live_isolated_url: tuple[str, int]) -> None:
    """Verify concurrent duplicate register returns single 200 + 409s and mixed WAL R/W succeeds."""
    base_url, _port = live_isolated_url
    valid_png = _make_plate_png(stage="glazed", cobalt=True)
    crack_png = _make_plate_png(stage="bisque", cobalt=False, crack=True)

    def _dup_register(idx: int) -> int:
        with httpx.Client(base_url=base_url, timeout=20.0) as client:
            return client.post(
                "/api/passport/register",
                files={"file": (f"r{idx}.png", valid_png, "image/png")},
                data={"item_id": "ADV-RACE-DUP-01", "stage": "glazed"},
            ).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        dup_codes = list(pool.map(_dup_register, range(10)))
    assert dup_codes.count(200) == 1
    assert dup_codes.count(409) == 9

    def _demo_verify(idx: int) -> int:
        target_id = "GZHEL-DEMO-01" if idx % 2 == 0 else "GZHEL-DEMO-02"
        with httpx.Client(base_url=base_url, timeout=20.0) as client:
            return client.post(
                f"/api/passport/verify/{target_id}",
                files={"file": (f"d{idx}.png", valid_png, "image/png")},
            ).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        demo_codes = list(pool.map(_demo_verify, range(8)))
    assert all(code == 200 for code in demo_codes)

    def _mixed_rw(idx: int) -> int:
        with httpx.Client(base_url=base_url, timeout=20.0) as client:
            if idx % 3 == 0:
                return client.get("/api/analytics", params={"limit": 25}).status_code
            return client.post(
                "/api/inspect",
                files={"file": (f"m{idx}.png", crack_png if idx % 2 else valid_png, "image/png")},
                data={"stage": "bisque" if idx % 2 else "glazed", "save_to_db": "true"},
            ).status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        rw_codes = list(pool.map(_mixed_rw, range(18)))
    assert all(code == 200 for code in rw_codes)


def test_adv_ui_original_view_unclipped_after_wipe(live_isolated_url: tuple[str, int]) -> None:
    """Verify B2: switching to Original view after wipe=0 shows unclipped #img-original."""
    from playwright.sync_api import sync_playwright

    base_url, _port = live_isolated_url
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        try:
            page = context.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.locator("#demo-sample-list button").first.click()
            page.locator("#btn-run-inspect").click()
            page.locator("#inspect-verdict-badge[data-verdict]").wait_for(state="visible", timeout=10000)

            page.evaluate(
                """() => {
                    const rng = document.querySelector('#compare-range');
                    rng.value = '0';
                    rng.dispatchEvent(new Event('input', {bubbles: true}));
                }"""
            )
            page.locator('#compare-mode-group [data-compare-mode="original"]').click()
            clip_val = page.evaluate("getComputedStyle(document.querySelector('#img-original')).clipPath")
            assert clip_val in ("none", "inset(0px)"), f"Original image remains clipped: {clip_val}"
        finally:
            context.close()
            browser.close()


def test_adv_ui_xss_metadata_escaped(live_isolated_url: tuple[str, int], tmp_path: Path) -> None:
    """Verify HTML/JS payloads in passport metadata render safely via textContent."""
    from playwright.sync_api import sync_playwright

    base_url, _port = live_isolated_url
    plate_bytes = _make_plate_png(stage="glazed", cobalt=True)
    plate_file = tmp_path / "plate.png"
    plate_file.write_bytes(plate_bytes)
    with httpx.Client(base_url=base_url, timeout=15.0) as client:
        assert client.post(
            "/api/inspect",
            files={"file": ('<img src=x onerror="window.__xss=4">.png', plate_bytes, "image/png")},
            data={"stage": "glazed", "shift": '<script>window.__xss=5</script>'},
        ).status_code == 200

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        try:
            page = context.new_page()
            page_errors: list[str] = []
            page.on("pageerror", lambda err: page_errors.append(str(err)))

            page.goto(base_url, wait_until="networkidle")
            page.locator("#demo-sample-list button").first.click()
            page.locator("#btn-run-inspect").click()
            page.locator("#inspect-verdict-badge[data-verdict]").wait_for(state="visible", timeout=10000)

            page.locator("#tab-passport").click()
            page.locator('#passport-action-tabs [data-action="register"]').click()
            page.locator("#register-item-id").fill("GZHEL-ADV-XSS")
            page.locator("#register-sku").fill('<img src=x onerror="window.__xss=1">')
            page.locator("#register-master-name").fill('<script>window.__xss=2</script>')
            page.locator("#register-batch-note").fill('"><svg onload="window.__xss=3">')
            page.locator("#register-file-input").set_input_files(str(plate_file))
            page.locator("#btn-register-passport").click()
            page.locator("#register-qr-img").wait_for(state="visible", timeout=10000)

            page.locator('#passport-action-tabs [data-action="verify"]').click()
            page.locator("#passport-id-input").fill("GZHEL-ADV-XSS")
            page.locator("#passport-file-input").set_input_files(str(plate_file))
            page.locator("#btn-verify-passport").click()
            page.locator('#passport-verdict-banner[data-authentic="true"]').wait_for(state="visible", timeout=10000)

            page.locator("#tab-dashboard").click()
            page.locator("#ledger-tbody tr").first.wait_for(state="visible", timeout=10000)
            assert page.evaluate("window.__xss === undefined") is True
            assert page.evaluate("document.querySelectorAll('img[src=\"x\"]').length") == 0
            assert page_errors == []
        finally:
            context.close()
            browser.close()


def test_adv_ui_double_submit_and_sample_switch_guards(live_isolated_url: tuple[str, int]) -> None:
    """Verify B3/B4: double-submit sends 1 POST and sample switching ignores stale response."""
    from playwright.sync_api import sync_playwright

    base_url, _port = live_isolated_url
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        try:
            page = context.new_page()

            inspect_posts: list[str] = []
            page.on("request", lambda req: inspect_posts.append(req.url) if "/api/inspect" in req.url else None)
            page.add_init_script(
                """(() => {
                    const origFetch = window.fetch.bind(window);
                    window.fetch = async (input, init) => {
                        const url = typeof input === 'string' ? input : input.url;
                        if (url.includes('/demo-assets/')) {
                            await new Promise((r) => setTimeout(r, 350));
                        }
                        if (url.includes('/api/inspect')) {
                            const resp = await origFetch(input, init);
                            await new Promise((r) => setTimeout(r, 400));
                            return resp;
                        }
                        return origFetch(input, init);
                    };
                })();"""
            )

            page.goto(base_url, wait_until="networkidle")
            chips = page.locator("#demo-sample-list button")
            chips.nth(0).click()
            page.locator("#btn-run-inspect").click()
            assert page.locator("#btn-run-inspect").is_disabled()
            page.locator("#btn-run-inspect").click(force=True)
            page.wait_for_timeout(1200)
            assert len(inspect_posts) == 1, f"Expected 1 POST /api/inspect, got {len(inspect_posts)}"

            chips.nth(0).click()
            page.wait_for_timeout(500)
            page.locator("#btn-run-inspect").click()
            page.wait_for_timeout(80)
            chips.nth(2).click()
            page.wait_for_timeout(900)
            summary = page.locator("#inspect-gost-summary").inner_text()
            assert "Утиль" not in summary or page.locator("#inspect-verdict-badge").is_hidden()
        finally:
            context.close()
            browser.close()


def test_adv_ui_analytics_stale_response_guard(live_isolated_url: tuple[str, int]) -> None:
    """Verify B5: stale unfiltered /api/analytics response cannot overwrite filtered ledger."""
    from playwright.sync_api import sync_playwright

    base_url, _port = live_isolated_url
    bisque_crack = _make_plate_png(stage="bisque", cobalt=False, crack=True)
    glazed_clean = _make_plate_png(stage="glazed", cobalt=True, crack=False)

    with httpx.Client(base_url=base_url, timeout=15.0) as client:
        assert client.post(
            "/api/inspect",
            files={"file": ("b.png", bisque_crack, "image/png")},
            data={"stage": "bisque", "item_id": "ADV-BISQUE-1"},
        ).status_code == 200
        assert client.post(
            "/api/inspect",
            files={"file": ("g.png", glazed_clean, "image/png")},
            data={"stage": "glazed", "item_id": "ADV-GLAZED-1"},
        ).status_code == 200

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        try:
            page = context.new_page()
            page.add_init_script(
                """(() => {
                    const origFetch = window.fetch.bind(window);
                    window.fetch = async (input, init) => {
                        const url = typeof input === 'string' ? input : input.url;
                        if (url.includes('/api/analytics') && !url.includes('stage=bisque')) {
                            const resp = await origFetch(input, init);
                            await new Promise((r) => setTimeout(r, 600));
                            return resp;
                        }
                        return origFetch(input, init);
                    };
                })();"""
            )

            page.goto(base_url, wait_until="networkidle")
            page.locator("#tab-dashboard").click()
            page.wait_for_timeout(50)
            page.locator("#filter-stage").select_option("bisque")
            page.wait_for_timeout(900)

            rows = page.locator("#ledger-tbody tr").all_inner_texts()
            assert len(rows) == 1
            assert "Глазурь" not in rows[0]
        finally:
            context.close()
            browser.close()


def test_adv_ui_chip_btn_ellipsis_at_768px(live_isolated_url: tuple[str, int]) -> None:
    """Verify B6: .chip-btn uses block layout with working ellipsis and zero overflow at 768x900."""
    from playwright.sync_api import sync_playwright

    base_url, _port = live_isolated_url
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 768, "height": 900})
        try:
            page = context.new_page()
            page.goto(base_url, wait_until="networkidle")
            page.locator("#demo-sample-list button").first.wait_for(state="visible", timeout=10000)

            info = page.evaluate(
                """() => {
                    const overflow = document.documentElement.scrollWidth > window.innerWidth;
                    const chips = [...document.querySelectorAll('#demo-sample-list .chip-btn')].map((el) => {
                        const cs = getComputedStyle(el);
                        const rect = el.getBoundingClientRect();
                        return {
                            text: el.textContent,
                            display: cs.display,
                            textOverflow: cs.textOverflow,
                            overflowX: cs.overflowX,
                            whiteSpace: cs.whiteSpace,
                            height: rect.height,
                        };
                    });
                    return {overflow, chips};
                }"""
            )
            assert info["overflow"] is False
            assert len(info["chips"]) >= 4
            for chip in info["chips"]:
                assert chip["height"] >= 43.5
                assert chip["display"] == "block"
                assert chip["textOverflow"] == "ellipsis"
                assert chip["overflowX"] == "hidden"
                assert chip["whiteSpace"] == "nowrap"
        finally:
            context.close()
            browser.close()


def test_adv_ui_stage_toggle_preserves_pending_demo_fetch(live_isolated_url: tuple[str, int]) -> None:
    """Verify toggling stage during slow demo asset fetch preserves pendingFilePromise and runs inspect."""
    from playwright.sync_api import sync_playwright

    base_url, _port = live_isolated_url
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        try:
            page = context.new_page()
            page.add_init_script(
                """(() => {
                    const origFetch = window.fetch.bind(window);
                    window.fetch = async (input, init) => {
                        const url = typeof input === 'string' ? input : input.url;
                        if (url.includes('/demo-assets/')) {
                            await new Promise((r) => setTimeout(r, 450));
                        }
                        return origFetch(input, init);
                    };
                })();"""
            )

            page.goto(base_url, wait_until="networkidle")
            first_chip = page.locator("#demo-sample-list button").first
            first_chip.wait_for(state="visible", timeout=10000)
            initial_stage = first_chip.get_attribute("data-stage") or "glazed"
            target_btn = "#stage-glazed" if initial_stage == "bisque" else "#stage-bisque"
            expected_stage_text = "Глазурь" if initial_stage == "bisque" else "Утиль"

            first_chip.click()
            page.wait_for_timeout(60)
            page.locator(target_btn).click()
            page.locator("#btn-run-inspect").click()

            page.locator("#inspect-verdict-badge[data-verdict]").wait_for(state="visible", timeout=10000)
            assert page.locator("#inspect-error").is_hidden()
            summary = page.locator("#inspect-gost-summary").inner_text()
            assert expected_stage_text in summary
        finally:
            context.close()
            browser.close()
