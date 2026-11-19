"""Unit and integration tests for CeramScan CV core, 289-FZ fingerprint, and SQLite WAL storage."""

from __future__ import annotations

import math
import re
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.cv_core.defect_detector import (
    DefectDetector,
    DefectRecord,
    DetectorConfig,
    _apply_gost_triage,
)
from src.cv_core.fingerprint import (
    VERDICT_AUTHENTIC,
    VERDICT_COUNTERFEIT,
    FingerprintConfig,
    extract_fingerprint,
    verify_fingerprint,
)
from src.cv_core.preprocessing import DefectClass, PreprocessConfig, preprocess_image
from src.cv_core.storage import (
    CeramStorage,
    DuplicatePassportError,
    InvalidItemIdError,
    PassportNotFoundError,
)


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


def _build_constant_onnx_bytes(
    output_array: np.ndarray,
    in_shape: tuple[int, int, int, int] = (1, 3, 640, 640),
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
        + _pb_field_bytes(2, b"CeramScanTestGraph")
        + _pb_field_bytes(11, _pb_value_info("images", 1, list(in_shape)))
        + _pb_field_bytes(12, _pb_value_info("output0", 1, list(output_array.shape)))
    )
    opset_proto = _pb_field_varint(2, 13)
    return (
        _pb_field_varint(1, 7)
        + _pb_field_bytes(2, b"ceramscan-test")
        + _pb_field_bytes(7, graph_proto)
        + _pb_field_bytes(8, opset_proto)
    )


def _make_plate(
    stage: str = "glazed",
    cobalt: bool = False,
    glare: bool = False,
    crack: bool = False,
    chip: bool = False,
    warp: bool = False,
    green_pigment: bool = False,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = 640
    img = np.full((size, size, 3), (30, 32, 36), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    rx = 240
    ry = int(rx * 0.85) if warp else rx

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    if chip:
        cv2.circle(mask, (cx + rx - 4, cy), 15, 0, -1)

    base_color = (178, 180, 184) if stage == "bisque" else (218, 222, 226)
    plate = np.full((size, size, 3), base_color, dtype=np.uint8)
    noise = rng.integers(-2, 3, size=(size, size, 3))
    plate = np.clip(plate.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if cobalt or green_pigment:
        dark = (58, 138, 66) if green_pigment else (176, 54, 20)
        mid = (82, 168, 92) if green_pigment else (206, 92, 46)
        cv2.circle(plate, (cx, cy), int(rx * 0.72), mid, 3, cv2.LINE_AA)
        for idx in range(8):
            ang = idx * (2.0 * math.pi / 8.0)
            px = int(cx + rx * 0.42 * math.cos(ang))
            py = int(cy + rx * 0.42 * math.sin(ang))
            cv2.ellipse(
                plate,
                (px, py),
                (34, 15),
                int(round(math.degrees(ang))),
                0,
                360,
                dark,
                -1,
                cv2.LINE_AA,
            )
            cv2.line(
                plate,
                (cx, cy),
                (int(cx + rx * 0.48 * math.cos(ang)), int(cy + rx * 0.48 * math.sin(ang))),
                base_color,
                1,
                cv2.LINE_AA,
            )
        cv2.circle(plate, (cx, cy), int(rx * 0.14), dark, 4, cv2.LINE_AA)

    if crack:
        pts = np.array(
            [[cx - 90, cy - 22], [cx - 35, cy - 6], [cx + 15, cy + 12], [cx + 85, cy + 2]],
            dtype=np.int32,
        )
        cv2.polylines(plate, [pts], False, (18, 18, 20), 3, cv2.LINE_AA)

    if glare and stage == "glazed":
        g_mask = np.zeros((size, size), dtype=np.float32)
        cv2.ellipse(g_mask, (cx - 45, cy - 55), (60, 32), -20, 0, 360, 1.0, -1)
        g_mask = cv2.GaussianBlur(g_mask, (15, 15), 4.0)[:, :, None]
        plate = np.clip(
            plate.astype(np.float32) * (1.0 - g_mask) + 254.0 * g_mask,
            0,
            255,
        ).astype(np.uint8)

    img[mask > 0] = plate[mask > 0]
    return img


def test_preprocess_macro_and_item_modes() -> None:
    """Verify full-item ellipse axes ordering and 150x200 macro fallback geometry."""
    repo_root = Path(__file__).resolve().parents[1]
    full_img = _make_plate(stage="glazed", cobalt=True, glare=True)
    frame = preprocess_image(full_img, stage="glazed")
    assert frame.geometry.scale_calibrated is True
    major_px, minor_px = frame.geometry.axes
    assert major_px >= minor_px > 400.0
    assert frame.glare_ratio > 0.01
    assert frame.cobalt_ratio > 0.015
    assert int(np.count_nonzero(frame.m_rim & frame.m_interior)) == 0

    ref_3 = repo_root / "data" / "reference_materials" / "ceramic_defect_samples" / "ceramic_defect_3.jpg"
    img_3 = cv2.imread(str(ref_3))
    assert img_3 is not None and img_3.shape[:2] == (150, 200)
    macro_frame = preprocess_image(img_3, stage="glazed")
    assert macro_frame.geometry.scale_calibrated is False
    assert macro_frame.geometry.px_per_mm == pytest.approx(640.0 / 160.0, rel=1e-3)
    assert int(np.count_nonzero(macro_frame.m_rim)) == 0
    assert int(np.count_nonzero(macro_frame.m_interior)) > 0


def test_reference_defects_1_to_4_detected_within_sla() -> None:
    """Verify ceramic_defect_1..4.jpg all trigger expected non-PASS verdicts under 300 ms."""
    repo_root = Path(__file__).resolve().parents[1]
    detector = DefectDetector()
    _ = detector.inspect(_make_plate(stage="bisque"), stage="bisque")

    samples_dir = repo_root / "data" / "reference_materials" / "ceramic_defect_samples"
    expected_classes = {
        1: DefectClass.CRACK,
        2: DefectClass.CRACK,
        3: DefectClass.GLAZE_BARE,
        4: DefectClass.CRACK,
    }
    for idx in range(1, 5):
        img = cv2.imread(str(samples_dir / f"ceramic_defect_{idx}.jpg"))
        assert img is not None
        stage = "bisque" if idx <= 2 else "glazed"
        res = detector.inspect(img, stage=stage)  # type: ignore[arg-type]
        assert res.verdict in ("REWORK", "REJECT")
        assert res.latency_ms < 300.0
        assert any(d.defect_class == expected_classes[idx] for d in res.defects)
        if idx == 3:
            assert not any(d.defect_class == DefectClass.CRACK for d in res.defects)


def test_cobalt_with_glare_passes_and_crack_across_cobalt_rejects() -> None:
    """Verify Gzhel cobalt + glare passes cleanly while a crack across cobalt rejects."""
    detector = DefectDetector()
    clean = _make_plate(stage="glazed", cobalt=True, glare=True)
    res_clean = detector.inspect(clean, stage="glazed")
    assert res_clean.verdict == "PASS"
    assert res_clean.defect_count == 0

    cracked = _make_plate(stage="glazed", cobalt=True, glare=True, crack=True)
    res_crack = detector.inspect(cracked, stage="glazed")
    assert res_crack.verdict == "REJECT"
    assert any(d.defect_class == DefectClass.CRACK for d in res_crack.defects)


def test_chip_does_not_double_trigger_shape_warp_and_warp_rejects() -> None:
    """Verify localized rim chip fires CHIP only while global warp fires SHAPE_WARP."""
    detector = DefectDetector()
    chipped = _make_plate(stage="bisque", chip=True)
    res_chip = detector.inspect(chipped, stage="bisque")
    assert res_chip.verdict == "REJECT"
    assert any(d.defect_class == DefectClass.CHIP for d in res_chip.defects)
    assert not any(d.defect_class == DefectClass.SHAPE_WARP for d in res_chip.defects)

    warped = _make_plate(stage="bisque", warp=True)
    res_warp = detector.inspect(warped, stage="bisque")
    assert res_warp.verdict == "REJECT"
    assert any(d.defect_class == DefectClass.SHAPE_WARP for d in res_warp.defects)


def test_gost_triage_pinhole_group_8_vs_9() -> None:
    """Verify <= 8 Накол pinholes count as 1 defect group while 9 triggers REJECT."""
    cfg = DetectorConfig()
    defects_8 = [
        DefectRecord(
            defect_class=DefectClass.PINHOLE_SPECK,
            gost_name_ru="Накол",
            severity="minor",
            bbox_xywh=(100 + i * 20, 100, 4, 4),
            area_px=12,
            area_mm2=1.0,
            length_mm=1.1,
            confidence=0.8,
            zone="interior",
            source="deterministic",
        )
        for i in range(8)
    ]
    verdict_8, grade_8, _ = _apply_gost_triage(defects_8, "glazed", cfg)
    assert verdict_8 == "PASS"
    assert grade_8 == "1 сорт"

    defects_9 = [*defects_8, defects_8[0]]
    verdict_9, grade_9, _ = _apply_gost_triage(defects_9, "glazed", cfg)
    assert verdict_9 == "REJECT"
    assert grade_9 == "Брак (REJECT)"


def test_onnx_weights_yolov8_yolov5_and_fail_fast(tmp_path: Path) -> None:
    """Verify ONNX YOLOv8/YOLOv5 decoding and fail-fast errors on invalid weights."""
    missing = tmp_path / "missing.onnx"
    with pytest.raises(FileNotFoundError):
        DefectDetector(weights_path=missing)

    bad_ext = tmp_path / "bad.txt"
    bad_ext.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        DefectDetector(weights_path=bad_ext)

    corrupt = tmp_path / "corrupt.onnx"
    corrupt.write_bytes(b"corrupt-bytes")
    with pytest.raises(ValueError):
        DefectDetector(weights_path=corrupt)

    # YOLOv8 (1, 10, 4) -> class 0 (CRACK)
    v8_arr = np.zeros((1, 10, 4), dtype=np.float32)
    v8_arr[0, 0:4, 0] = [320.0, 320.0, 120.0, 18.0]
    v8_arr[0, 4, 0] = 0.90
    v8_path = tmp_path / "yolov8.onnx"
    v8_path.write_bytes(_build_constant_onnx_bytes(v8_arr))
    res_v8 = DefectDetector(weights_path=v8_path).inspect(_make_plate(), stage="glazed")
    assert res_v8.model_mode == "onnx"
    assert res_v8.verdict == "REJECT"
    assert any(d.source == "onnx" and d.defect_class == DefectClass.CRACK for d in res_v8.defects)

    # YOLOv5 (1, 4, 11) -> overlapping class 2 (GLAZE_BARE) deduplicated by NMSBoxes
    v5_arr = np.zeros((1, 4, 11), dtype=np.float32)
    v5_arr[0, 0, :5] = [280.0, 280.0, 80.0, 80.0, 0.95]
    v5_arr[0, 0, 5 + 2] = 0.95
    v5_arr[0, 1, :5] = [282.0, 281.0, 80.0, 80.0, 0.88]
    v5_arr[0, 1, 5 + 2] = 0.88
    v5_path = tmp_path / "yolov5.onnx"
    v5_path.write_bytes(_build_constant_onnx_bytes(v5_arr))
    res_v5 = DefectDetector(weights_path=v5_path).inspect(_make_plate(), stage="glazed")
    assert res_v5.model_mode == "onnx"
    bare_hits = [d for d in res_v5.defects if d.defect_class == DefectClass.GLAZE_BARE]
    assert len(bare_hits) == 1


def test_fingerprint_genuine_counterfeit_wrong_pigment_and_blank() -> None:
    """Verify 289-FZ matcher distinguishes genuine tilt, wrong pigment, and blank frames."""
    genuine = _make_plate(stage="glazed", cobalt=True, seed=10)
    ref_frame = preprocess_image(genuine, stage="glazed")
    sig = extract_fingerprint(ref_frame)
    assert sig.orb_count >= 12

    rot = cv2.getRotationMatrix2D((320.0, 320.0), 4.0, 1.0)
    tilted = cv2.warpAffine(genuine, rot, (640, 640), borderValue=(30, 32, 36))
    res_ok = verify_fingerprint(sig, preprocess_image(tilted, stage="glazed"))
    assert res_ok.verdict == VERDICT_AUTHENTIC
    assert res_ok.match_percent >= 62.0

    wrong_pigment = _make_plate(stage="glazed", green_pigment=True, seed=10)
    res_green = verify_fingerprint(sig, preprocess_image(wrong_pigment, stage="glazed"))
    assert res_green.verdict == VERDICT_COUNTERFEIT
    assert res_green.color_score == 0.0

    blank = np.full((640, 640, 3), 185, dtype=np.uint8)
    blank_frame = preprocess_image(blank, stage="glazed")
    blank_sig = extract_fingerprint(blank_frame)
    assert blank_sig.orb_count == 0
    assert blank_sig.descriptors_b64 == ""
    res_blank = verify_fingerprint(sig, blank_frame)
    assert res_blank.verdict == VERDICT_COUNTERFEIT

    # Custom threshold guard
    strict_cfg = FingerprintConfig(authentic_threshold_pct=99.9, min_matched_kp=500)
    assert verify_fingerprint(sig, preprocess_image(tilted, stage="glazed"), strict_cfg).verdict == VERDICT_COUNTERFEIT


def test_storage_wal_security_duplicate_and_analytics_filter(tmp_path: Path) -> None:
    """Verify SQLite WAL, path containment, duplicate error, and scoped analytics filter."""
    db_path = tmp_path / "ceramscan.db"
    qr_dir = tmp_path / "qr"
    ref_dir = tmp_path / "passports"
    storage = CeramStorage(db_path=db_path, qr_dir=qr_dir, passport_ref_dir=ref_dir)
    assert storage.health_check() is True

    with sqlite3.connect(db_path) as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(mode).lower() == "wal"

    det = DefectDetector()
    r_bisque = det.inspect(_make_plate(stage="bisque", crack=True), stage="bisque")
    r_glazed = det.inspect(_make_plate(stage="glazed", cobalt=True), stage="glazed")
    storage.save_inspection(r_bisque, overlay_path="ov1.png", item_id="ITEM-01", shift="S1")
    storage.save_inspection(r_glazed, overlay_path="ov2.png", item_id="ITEM-02", shift="S2")

    with pytest.raises(InvalidItemIdError):
        storage.save_inspection(r_glazed, overlay_path="ov.png", item_id="../evil")

    analytics = storage.get_analytics(limit=10, stage="bisque", verdict=None)
    assert analytics["summary"]["total_inspections"] == 2
    assert len(analytics["recent_inspections"]) == 1
    assert analytics["recent_inspections"][0]["stage"] == "bisque"

    cobalt_img = _make_plate(stage="glazed", cobalt=True)
    frame = preprocess_image(cobalt_img, stage="glazed")
    sig = extract_fingerprint(frame)
    reg = storage.register_passport(
        sig,
        item_id="GZHEL-TEST-01",
        sku="SKU-9",
        master_name="Мастер Петров",
        batch_note="Смена-1",
        stage="glazed",
        reference_image_bgr=cobalt_img,
    )
    assert reg["qr_payload"] == "CERAMSCAN:GZHEL-TEST-01"
    assert (qr_dir / "GZHEL-TEST-01.png").is_file()
    assert (ref_dir / "GZHEL-TEST-01.png").is_file()

    with pytest.raises(DuplicatePassportError):
        storage.register_passport(sig, item_id="GZHEL-TEST-01")

    with pytest.raises(PassportNotFoundError):
        storage.get_passport("UNKNOWN-999")

    loaded = storage.get_passport("GZHEL-TEST-01")
    assert loaded["sku"] == "SKU-9"
    ver = verify_fingerprint(loaded["signature"], frame)
    ver_id = storage.record_verification("GZHEL-TEST-01", ver)
    assert ver_id >= 1


def _make_rotated_calibration_scene(
    with_plate: bool = True,
    border_inset: int | None = 8,
    border_thick: int = 8,
    crack: bool = False,
    chip: bool = False,
) -> np.ndarray:
    card = np.full((640, 640, 3), (242, 242, 244), dtype=np.uint8)
    if border_inset is not None:
        cv2.rectangle(
            card,
            (border_inset, border_inset),
            (639 - border_inset, 639 - border_inset),
            (24, 24, 28),
            border_thick,
        )
    if with_plate:
        cx, cy, r = 320, 320, 215
        plate_mask = np.zeros((640, 640), dtype=np.uint8)
        cv2.circle(plate_mask, (cx, cy), r, 255, -1)
        if chip:
            cv2.circle(plate_mask, (cx + r - 4, cy), 18, 0, -1)
        card[plate_mask > 0] = (218, 220, 224)
    if crack:
        pts = np.array([[230, 300], [300, 315], [410, 330]], dtype=np.int32)
        cv2.polylines(card, [pts], False, (18, 18, 22), 4, cv2.LINE_AA)

    scene = np.full((900, 900, 3), (32, 34, 38), dtype=np.uint8)
    scene[130:770, 130:770] = card
    rot = cv2.getRotationMatrix2D((450.0, 450.0), 15.0, 1.0)
    return cv2.warpAffine(scene, rot, (900, 900), borderValue=(32, 34, 38))


def test_calibration_quad_inner_border_case_a_and_case_b_tile() -> None:
    """Verify hollow inner card borders do not hide plates or trigger false tile warps."""
    detector = DefectDetector()

    for inset, thick in ((8, 8), (16, 4)):
        scene_a = _make_rotated_calibration_scene(
            with_plate=True, border_inset=inset, border_thick=thick
        )
        frame_a = preprocess_image(scene_a, stage="glazed")
        assert frame_a.perspective_applied is True
        assert frame_a.geometry.scale_calibrated is True
        res_a = detector.inspect(scene_a, stage="glazed")
        assert res_a.verdict == "PASS"
        assert res_a.defect_count == 0

    scene_crack = _make_rotated_calibration_scene(
        with_plate=True, border_inset=8, border_thick=8, crack=True
    )
    res_crack = detector.inspect(scene_crack, stage="glazed")
    assert res_crack.verdict == "REJECT"
    assert any(d.defect_class == DefectClass.CRACK for d in res_crack.defects)

    scene_chip = _make_rotated_calibration_scene(
        with_plate=True, border_inset=8, border_thick=8, chip=True
    )
    res_chip = detector.inspect(scene_chip, stage="glazed")
    assert res_chip.verdict == "REJECT"
    assert any(d.defect_class == DefectClass.CHIP for d in res_chip.defects)

    for inset in (None, 8):
        tile_clean = _make_rotated_calibration_scene(
            with_plate=False, border_inset=inset, border_thick=8
        )
        frame_b = preprocess_image(tile_clean, stage="glazed")
        assert frame_b.perspective_applied is True
        assert frame_b.geometry.scale_calibrated is False
        assert int(np.count_nonzero(frame_b.m_rim)) == 0
        res_b = detector.inspect(tile_clean, stage="glazed")
        assert res_b.verdict == "PASS"
        assert res_b.defect_count == 0

    tile_crack = _make_rotated_calibration_scene(
        with_plate=False, border_inset=8, border_thick=8, crack=True
    )
    res_tile_crack = detector.inspect(tile_crack, stage="glazed")
    assert res_tile_crack.verdict == "REJECT"
    assert any(d.defect_class == DefectClass.CRACK for d in res_tile_crack.defects)


def test_gost_triage_glaze_drip_rework_vs_reject() -> None:
    """Verify single localized GLAZE_DRIP <= 100 mm2 triages to REWORK while >100 or >1 rejects."""
    cfg = DetectorConfig()

    def _make_drip(area_mm2: float, x: int = 200) -> DefectRecord:
        return DefectRecord(
            defect_class=DefectClass.GLAZE_DRIP,
            gost_name_ru="Натек / Прыщ",
            severity="rework",
            bbox_xywh=(x, 200, 16, 56),
            area_px=620,
            area_mm2=area_mm2,
            length_mm=20.0,
            confidence=0.76,
            zone="interior",
            source="deterministic",
        )

    v_ok, g_ok, _ = _apply_gost_triage([_make_drip(85.0)], "glazed", cfg)
    assert v_ok == "REWORK"
    assert g_ok == "Доработка (REWORK)"

    v_big, g_big, _ = _apply_gost_triage([_make_drip(105.0)], "glazed", cfg)
    assert v_big == "REJECT"
    assert g_big == "Брак (REJECT)"

    v_multi, g_multi, _ = _apply_gost_triage(
        [_make_drip(40.0, 180), _make_drip(40.0, 320)], "glazed", cfg
    )
    assert v_multi == "REJECT"
    assert g_multi == "Брак (REJECT)"

    drip_img = _make_plate(stage="glazed", cobalt=False, glare=False)
    drip_pts = np.array([[355, 278], [359, 302], [357, 324]], dtype=np.int32)
    cv2.polylines(drip_img, [drip_pts], False, (172, 184, 202), 8, cv2.LINE_AA)
    res_drip = DefectDetector().inspect(drip_img, stage="glazed")
    assert res_drip.verdict == "REWORK"
    assert any(d.defect_class == DefectClass.GLAZE_DRIP for d in res_drip.defects)


def test_onnx_yolov8_disambiguation_when_n_equals_5_plus_c(tmp_path: Path) -> None:
    """Verify (1, 10, 11) tensors disambiguate YOLOv8 vs YOLOv5 without dead branch."""
    # Single-candidate (1, 10, 11) YOLOv8
    v8_single = np.zeros((1, 10, 11), dtype=np.float32)
    v8_single[0, 0:4, 0] = [320.0, 320.0, 120.0, 18.0]
    v8_single[0, 4, 0] = 0.90
    p1 = tmp_path / "v8_10x11_single.onnx"
    p1.write_bytes(_build_constant_onnx_bytes(v8_single))
    res1 = DefectDetector(weights_path=p1).inspect(_make_plate(), stage="glazed")
    assert res1.verdict == "REJECT"
    assert any(d.defect_class == DefectClass.CRACK for d in res1.defects)

    # Multi-candidate (1, 10, 11) YOLOv8 with candidate >= 4 populated
    v8_multi = np.zeros((1, 10, 11), dtype=np.float32)
    v8_multi[0, 0:4, 5] = [280.0, 280.0, 110.0, 16.0]
    v8_multi[0, 4, 5] = 0.88
    p2 = tmp_path / "v8_10x11_multi.onnx"
    p2.write_bytes(_build_constant_onnx_bytes(v8_multi))
    res2 = DefectDetector(weights_path=p2).inspect(_make_plate(), stage="glazed")
    assert res2.verdict == "REJECT"
    assert any(d.defect_class == DefectClass.CRACK for d in res2.defects)

    # (1, 10, 11) YOLOv5 tensor
    v5_arr = np.zeros((1, 10, 11), dtype=np.float32)
    v5_arr[0, 5, :5] = [300.0, 300.0, 80.0, 80.0, 0.95]
    v5_arr[0, 5, 5 + 2] = 0.92
    p3 = tmp_path / "v5_10x11.onnx"
    p3.write_bytes(_build_constant_onnx_bytes(v5_arr))
    res3 = DefectDetector(weights_path=p3).inspect(_make_plate(), stage="glazed")
    assert any(d.defect_class == DefectClass.GLAZE_BARE for d in res3.defects)


def test_onnx_small_pinhole_nakol_parity(tmp_path: Path) -> None:
    """Verify <= 8 small ONNX pinholes in glazed stage map to Накол and PASS, 9 REJECT."""
    plate = _make_plate(stage="glazed")

    v8_5 = np.zeros((1, 10, 8), dtype=np.float32)
    for i in range(5):
        v8_5[0, 0:4, i] = [200.0 + i * 35.0, 260.0, 4.0, 4.0]
        v8_5[0, 4 + 3, i] = 0.90
    p5 = tmp_path / "pinhole_5.onnx"
    p5.write_bytes(_build_constant_onnx_bytes(v8_5))
    res5 = DefectDetector(weights_path=p5).inspect(plate, stage="glazed")
    assert res5.verdict == "PASS"
    assert len(res5.defects) == 5
    assert all(d.gost_name_ru == "Накол" for d in res5.defects)

    v8_9 = np.zeros((1, 10, 12), dtype=np.float32)
    for i in range(9):
        v8_9[0, 0:4, i] = [180.0 + (i % 5) * 40.0, 220.0 + (i // 5) * 60.0, 4.0, 4.0]
        v8_9[0, 4 + 3, i] = 0.90
    p9 = tmp_path / "pinhole_9.onnx"
    p9.write_bytes(_build_constant_onnx_bytes(v8_9))
    res9 = DefectDetector(weights_path=p9).inspect(plate, stage="glazed")
    assert res9.verdict == "REJECT"
    assert len(res9.defects) == 9

    v8_large = np.zeros((1, 10, 4), dtype=np.float32)
    v8_large[0, 0:4, 0] = [300.0, 300.0, 10.0, 10.0]
    v8_large[0, 4 + 3, 0] = 0.90
    p_lg = tmp_path / "speck_large.onnx"
    p_lg.write_bytes(_build_constant_onnx_bytes(v8_large))
    res_lg = DefectDetector(weights_path=p_lg).inspect(plate, stage="glazed")
    assert res_lg.verdict == "REJECT"
    assert res_lg.defects[0].gost_name_ru == "Накол / Мушка"


def test_storage_whitespace_item_id_consistency(tmp_path: Path) -> None:
    """Verify whitespace item_id behaves consistently across inspection and passport APIs."""
    db_path = tmp_path / "ws.db"
    qr_dir = tmp_path / "qr"
    ref_dir = tmp_path / "passports"
    storage = CeramStorage(db_path=db_path, qr_dir=qr_dir, passport_ref_dir=ref_dir)

    det = DefectDetector()
    cobalt_img = _make_plate(stage="glazed", cobalt=True)
    res = det.inspect(cobalt_img, stage="glazed")

    row_id = storage.save_inspection(res, overlay_path="ov.png", item_id="   ")
    assert row_id >= 1
    with sqlite3.connect(db_path) as conn:
        saved = conn.execute("SELECT item_id FROM inspections WHERE id = ?;", (row_id,)).fetchone()
        assert saved is not None and saved[0] is None

    with pytest.raises(InvalidItemIdError):
        storage.save_inspection(res, overlay_path="ov2.png", item_id="  ITEM-01  ")

    frame = preprocess_image(cobalt_img, stage="glazed")
    sig = extract_fingerprint(frame)
    reg = storage.register_passport(sig, item_id="   ", reference_image_bgr=cobalt_img)
    assert re.match(r"^GZHEL-\d{8}-[0-9a-f]{6}$", reg["item_id"])
    assert (qr_dir / f"{reg['item_id']}.png").is_file()
    assert (ref_dir / f"{reg['item_id']}.png").is_file()

    with pytest.raises(InvalidItemIdError):
        storage.get_passport("   ")
    ver = verify_fingerprint(sig, frame)
    with pytest.raises(InvalidItemIdError):
        storage.record_verification("   ", ver)


def test_storage_register_passport_invalid_ref_image_no_orphan_qr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify invalid reference_image_bgr leaves zero orphan QR PNGs and zero DB rows."""
    db_path = tmp_path / "orphan.db"
    qr_dir = tmp_path / "qr"
    ref_dir = tmp_path / "passports"
    storage = CeramStorage(db_path=db_path, qr_dir=qr_dir, passport_ref_dir=ref_dir)

    cobalt_img = _make_plate(stage="glazed", cobalt=True)
    sig = extract_fingerprint(preprocess_image(cobalt_img, stage="glazed"))

    for bad_img in (
        np.empty((0, 0, 3), dtype=np.uint8),
        np.zeros((10, 10, 5), dtype=np.uint8),
    ):
        with pytest.raises(ValueError):
            storage.register_passport(sig, item_id="GZHEL-ORPHAN-01", reference_image_bgr=bad_img)
        assert not (qr_dir / "GZHEL-ORPHAN-01.png").exists()
        assert not (ref_dir / "GZHEL-ORPHAN-01.png").exists()

    monkeypatch.setattr(cv2, "imencode", lambda *args, **kwargs: (False, np.empty(0, dtype=np.uint8)))
    with pytest.raises(RuntimeError):
        storage.register_passport(sig, item_id="GZHEL-ORPHAN-02", reference_image_bgr=cobalt_img)
    assert not (qr_dir / "GZHEL-ORPHAN-02.png").exists()
    assert not (ref_dir / "GZHEL-ORPHAN-02.png").exists()

    with sqlite3.connect(db_path) as conn:
        cnt = conn.execute("SELECT COUNT(*) FROM passports;").fetchone()[0]
        assert cnt == 0

    monkeypatch.undo()
    ok_reg = storage.register_passport(sig, item_id="GZHEL-ORPHAN-01", reference_image_bgr=cobalt_img)
    assert ok_reg["item_id"] == "GZHEL-ORPHAN-01"
    assert (qr_dir / "GZHEL-ORPHAN-01.png").is_file()
    assert (ref_dir / "GZHEL-ORPHAN-01.png").is_file()
