"""Deterministic CPU ceramic defect detector, --weights ONNX hook, and GOST 28390-89 triage."""

from __future__ import annotations

import ast
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
import onnxruntime.capi.onnxruntime_pybind11_state as ort_state
from PIL import Image, ImageDraw
from scipy.ndimage import median_filter
from skimage.filters import sato
from skimage.morphology import skeletonize

from .preprocessing import (
    DefectClass,
    PreprocessConfig,
    PreprocessedFrame,
    StageType,
    VerdictType,
    get_cyrillic_font,
    preprocess_image,
)

_CLASS_BY_INDEX: dict[int, DefectClass] = {
    0: DefectClass.CRACK,
    1: DefectClass.CHIP,
    2: DefectClass.GLAZE_BARE,
    3: DefectClass.PINHOLE_SPECK,
    4: DefectClass.GLAZE_DRIP,
    5: DefectClass.SHAPE_WARP,
}

_GOST_NAME_BY_CLASS: dict[DefectClass, str] = {
    DefectClass.CRACK: "Трещина / Цек",
    DefectClass.CHIP: "Откол / Скол кромки",
    DefectClass.GLAZE_BARE: "Плешина, сборка глазури",
    DefectClass.PINHOLE_SPECK: "Накол / Мушка",
    DefectClass.GLAZE_DRIP: "Натек / Прыщ",
    DefectClass.SHAPE_WARP: "Деформация от характерного размера",
}

_SHORT_LABEL_BY_CLASS: dict[DefectClass, str] = {
    DefectClass.CRACK: "Трещина/цек",
    DefectClass.CHIP: "Скол кромки",
    DefectClass.GLAZE_BARE: "Плешина/выгорка",
    DefectClass.PINHOLE_SPECK: "Накол/мушка",
    DefectClass.GLAZE_DRIP: "Натек глазури",
    DefectClass.SHAPE_WARP: "Перекос контура",
}

_SEVERITY_RANK: dict[str, int] = {
    "critical": 0,
    "rework": 1,
    "minor": 2,
}


def _build_line_kernels(length: int = 15) -> list[np.ndarray]:
    """Create 4 directional morphological line structuring elements (0, 45, 90, 135 deg)."""
    kernels: list[np.ndarray] = []
    half = length // 2
    cx, cy = half, half
    for deg in (0, 45, 90, 135):
        k = np.zeros((length, length), dtype=np.uint8)
        rad = math.radians(deg)
        dx = round(half * math.cos(rad))
        dy = round(half * math.sin(rad))
        cv2.line(k, (cx - dx, cy - dy), (cx + dx, cy + dy), 1, thickness=1)
        kernels.append(k)
    return kernels


_LINE_KERNELS_15 = _build_line_kernels(15)
_ELLIPSE_9X9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_ELLIPSE_17X17 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
_ELLIPSE_41X41 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
_MORPH_3X3 = np.ones((3, 3), dtype=np.uint8)
_MORPH_5X5 = np.ones((5, 5), dtype=np.uint8)

_ORT_LOAD_ERRORS = (
    RuntimeError,
    ValueError,
    OSError,
    ort_state.InvalidProtobuf,
    ort_state.InvalidGraph,
    ort_state.InvalidArgument,
    ort_state.NoSuchFile,
    ort_state.Fail,
    ort_state.RuntimeException,
)


@dataclass(slots=True)
class DefectRecord:
    defect_class: DefectClass
    gost_name_ru: str
    severity: Literal["minor", "rework", "critical"]
    bbox_xywh: tuple[int, int, int, int]
    area_px: int
    area_mm2: float
    length_mm: float
    confidence: float
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


def _map_name_to_class(raw_name: str, fallback_idx: int) -> DefectClass:
    norm = raw_name.strip().lower().replace("-", "_").replace(" ", "_")
    for cls in DefectClass:
        if cls.value == norm or cls.name.lower() == norm:
            return cls
    aliases: dict[str, DefectClass] = {
        "crazing": DefectClass.CRACK,
        "crack": DefectClass.CRACK,
        "chip": DefectClass.CHIP,
        "glaze_bare": DefectClass.GLAZE_BARE,
        "bare": DefectClass.GLAZE_BARE,
        "pinhole": DefectClass.PINHOLE_SPECK,
        "speck": DefectClass.PINHOLE_SPECK,
        "pinhole_speck": DefectClass.PINHOLE_SPECK,
        "drip": DefectClass.GLAZE_DRIP,
        "glaze_drip": DefectClass.GLAZE_DRIP,
        "warp": DefectClass.SHAPE_WARP,
        "shape_warp": DefectClass.SHAPE_WARP,
    }
    if norm in aliases:
        return aliases[norm]
    return _CLASS_BY_INDEX.get(fallback_idx, DefectClass.CRACK)


def _letterbox_bgr(
    img_bgr: np.ndarray,
    new_shape: tuple[int, int],
) -> tuple[np.ndarray, float, tuple[int, int]]:
    ih, iw = img_bgr.shape[:2]
    nh, nw = new_shape
    scale = min(nw / float(iw), nh / float(ih))
    rw = max(1, round(iw * scale))
    rh = max(1, round(ih * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img_bgr, (rw, rh), interpolation=interp)
    canvas = np.full((nh, nw, 3), 114, dtype=np.uint8)
    pad_x = (nw - rw) // 2
    pad_y = (nh - rh) // 2
    canvas[pad_y : pad_y + rh, pad_x : pad_x + rw] = resized
    return canvas, scale, (pad_x, pad_y)


def _detect_rim_chip_and_warp(
    frame: PreprocessedFrame,
    config: DetectorConfig,
    source: Literal["deterministic", "onnx"],
) -> tuple[list[DefectRecord], float]:
    """Detect localized rim chips via circular median radius filter and global warp."""
    if not frame.geometry.scale_calibrated or not np.any(frame.m_rim):
        return [], frame.geometry.contour_deviation_pct

    contours, _ = cv2.findContours(frame.m_item, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return [], frame.geometry.contour_deviation_pct

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 30:
        return [], frame.geometry.contour_deviation_pct

    pts = cnt.reshape(-1, 2).astype(np.float64)
    cx, cy = frame.geometry.center
    major_px, minor_px = frame.geometry.axes
    angle_deg = frame.geometry.angle_deg
    px_per_mm = max(0.5, frame.geometry.px_per_mm)

    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    rx = max(1.0, major_px / 2.0)
    ry = max(1.0, minor_px / 2.0)
    phi_raw = np.arctan2(dy, dx)
    order = np.argsort(phi_raw)
    pts_ord = pts[order]
    r_actual = np.hypot(pts_ord[:, 0] - cx, pts_ord[:, 1] - cy)

    l_dx = pts_ord[:, 0] - cx
    l_dy = pts_ord[:, 1] - cy
    loc_x = l_dx * cos_t + l_dy * sin_t
    loc_y = -l_dx * sin_t + l_dy * cos_t
    loc_phi = np.arctan2(loc_y, loc_x)
    r_opt_a = (rx * ry) / np.maximum(1e-6, np.hypot(ry * np.cos(loc_phi), rx * np.sin(loc_phi)))
    r_opt_b = (rx * ry) / np.maximum(1e-6, np.hypot(rx * np.cos(loc_phi), ry * np.sin(loc_phi)))
    err_a = float(np.median(np.abs(r_actual - r_opt_a)))
    err_b = float(np.median(np.abs(r_actual - r_opt_b)))
    r_ellipse = r_opt_a if err_a <= err_b else r_opt_b

    n_pts = len(r_actual)
    win = max(31, (n_pts // 5) | 1)
    r_med = median_filter(r_actual, size=win, mode="wrap")
    inward = r_med - r_actual
    if frame.geometry.axis_ratio_error_pct <= 2.0:
        inward = np.maximum(inward, r_ellipse - r_actual)
    notch_thresh = max(3.5, 0.9 * px_per_mm)
    raw_notch = inward > notch_thresh
    # Bridge tiny 1..8 pt contour quantization gaps around a single rim chip
    notch_mask = cv2.morphologyEx(
        raw_notch.astype(np.uint8).reshape(1, -1),
        cv2.MORPH_CLOSE,
        np.ones((1, 9), dtype=np.uint8),
    ).ravel() > 0

    defects: list[DefectRecord] = []
    is_chip_pt = np.zeros(n_pts, dtype=bool)

    if np.any(notch_mask) and not np.all(notch_mask) and frame.geometry.axis_ratio_error_pct <= 2.5:
        padded = np.concatenate([[False], notch_mask, [False]])
        edges = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        segments: list[list[int]] = [
            list(range(int(s), int(e))) for s, e in zip(starts, ends, strict=True)
        ]
        if len(segments) >= 2 and segments[0][0] == 0 and segments[-1][-1] == n_pts - 1:
            merged = segments[-1] + segments[0]
            segments = [merged, *segments[1:-1]]

        for seg in segments:
            if not (6 <= len(seg) < 0.14 * n_pts):
                continue
            seg_idx = np.asarray(seg, dtype=np.intp)
            if float(np.max(inward[seg_idx])) <= notch_thresh + 0.5:
                continue
            is_chip_pt[seg_idx] = True
            seg_pts = pts_ord[seg_idx].astype(np.int32)
            x, y, bw, bh = cv2.boundingRect(seg_pts)
            pad = 4
            bx = max(0, x - pad)
            by = max(0, y - pad)
            bw_p = max(4, min(frame.roi_bgr.shape[1] - bx, bw + 2 * pad))
            bh_p = max(4, min(frame.roi_bgr.shape[0] - by, bh + 2 * pad))

            max_depth_px = float(np.max(inward[seg_idx]))
            depth_mm = max_depth_px / px_per_mm
            arc_mm = max(depth_mm, float(len(seg_idx)) / px_per_mm)
            area_px = max(1, int(round(float(np.sum(inward[seg_idx])))))
            area_mm2 = float(area_px) / (px_per_mm * px_per_mm)

            defects.append(
                DefectRecord(
                    defect_class=DefectClass.CHIP,
                    gost_name_ru=_GOST_NAME_BY_CLASS[DefectClass.CHIP],
                    severity="critical",
                    bbox_xywh=(int(bx), int(by), int(bw_p), int(bh_p)),
                    area_px=area_px,
                    area_mm2=round(area_mm2, 2),
                    length_mm=round(max(depth_mm, arc_mm * 0.5), 2),
                    confidence=round(min(0.98, 0.72 + max_depth_px / 30.0), 3),
                    zone="rim",
                    source=source,
                )
            )

    mean_radius = max(1.0, float(np.mean(r_ellipse)))
    valid_pts = ~is_chip_pt
    if int(np.count_nonzero(valid_pts)) >= 20:
        non_chip_dev_pct = float(
            np.percentile(np.abs(r_actual[valid_pts] - r_ellipse[valid_pts]), 95)
            / mean_radius
            * 100.0
        )
    else:
        non_chip_dev_pct = frame.geometry.contour_deviation_pct

    reported_dev = max(non_chip_dev_pct, frame.geometry.axis_ratio_error_pct)
    is_warped = (
        non_chip_dev_pct > config.max_contour_deviation_pct
        or frame.geometry.axis_ratio_error_pct > config.max_contour_deviation_pct
    )
    if is_warped:
        xs = pts[:, 0]
        ys = pts[:, 1]
        bx = max(0, int(np.min(xs)))
        by = max(0, int(np.min(ys)))
        bw = max(8, int(np.max(xs) - bx))
        bh = max(8, int(np.max(ys) - by))
        defects.append(
            DefectRecord(
                defect_class=DefectClass.SHAPE_WARP,
                gost_name_ru=_GOST_NAME_BY_CLASS[DefectClass.SHAPE_WARP],
                severity="critical",
                bbox_xywh=(bx, by, bw, bh),
                area_px=int(cv2.contourArea(cnt)),
                area_mm2=round(float(cv2.contourArea(cnt)) / (px_per_mm * px_per_mm), 2),
                length_mm=round(major_px / px_per_mm, 2),
                confidence=round(min(0.99, 0.75 + reported_dev / 20.0), 3),
                zone="rim",
                source=source,
            )
        )

    effective_dev = (
        round(reported_dev, 4)
        if is_warped
        else round(frame.geometry.contour_deviation_pct, 4)
    )
    return defects, effective_dev


def _detect_deterministic_surface_defects(
    frame: PreprocessedFrame,
    config: DetectorConfig,
    rim_defects: list[DefectRecord] | None = None,
) -> tuple[list[DefectRecord], np.ndarray]:
    """Run calibrated 320px Sato + 4-angle Black-Hat cracks and compact glaze/bisque defects."""
    h, w = frame.roi_bgr.shape[:2]
    px_per_mm = max(0.5, frame.geometry.px_per_mm)
    px_per_mm2 = px_per_mm * px_per_mm

    m_int_eff = frame.m_interior.copy()
    if rim_defects:
        for rd in rim_defects:
            if rd.defect_class == DefectClass.CHIP:
                bx, by, bw, bh = rd.bbox_xywh
                pad_c = max(48, bw + 20, bh + 20)
                m_int_eff[
                    max(0, by - pad_c) : min(h, by + bh + pad_c),
                    max(0, bx - pad_c) : min(w, bx + bw + pad_c),
                ] = 0
    if not frame.geometry.scale_calibrated:
        inset = max(24, round(min(h, w) * 0.075))
        m_int_eff[:inset, :] = 0
        m_int_eff[-inset:, :] = 0
        m_int_eff[:, :inset] = 0
        m_int_eff[:, -inset:] = 0

    lab_roi = cv2.cvtColor(frame.roi_bgr, cv2.COLOR_BGR2LAB)
    l_raw = lab_roi[:, :, 0]
    s_raw = frame.roi_hsv[:, :, 1]

    d_size = 320
    scale_inv = float(w) / float(d_size)
    l320 = cv2.resize(l_raw, (d_size, d_size), interpolation=cv2.INTER_AREA)
    s320 = cv2.resize(s_raw, (d_size, d_size), interpolation=cv2.INTER_AREA)
    int320 = cv2.resize(m_int_eff, (d_size, d_size), interpolation=cv2.INTER_NEAREST) > 0
    cob320 = cv2.resize(frame.m_cobalt, (d_size, d_size), interpolation=cv2.INTER_NEAREST)
    spec320 = cv2.resize(frame.m_spec, (d_size, d_size), interpolation=cv2.INTER_NEAREST)

    cob_dil320 = cv2.dilate(cob320, _MORPH_5X5) > 0
    bg_mask320 = int320 & (~cob_dil320) & (spec320 == 0)
    if not np.any(bg_mask320):
        bg_mask320 = int320 & (~cob_dil320)
    if not np.any(bg_mask320):
        bg_mask320 = int320
    bg_l320 = float(np.median(l320[bg_mask320])) if np.any(bg_mask320) else 180.0

    dark_neutral = (l320 < 55) & (s320 < 55)
    l320_clean = np.where(cob_dil320 & (~dark_neutral), int(round(bg_l320)), l320).astype(
        np.uint8
    )

    bh_responses = [cv2.morphologyEx(l320_clean, cv2.MORPH_BLACKHAT, k) for k in _LINE_KERNELS_15]
    r_max = np.maximum.reduce(bh_responses)

    int_vals = r_max[int320]
    raw_med = float(np.median(int_vals)) if int_vals.size > 0 else 0.0
    striated = raw_med > 15.0

    spec_border320 = (cv2.dilate(spec320, _MORPH_5X5) > 0) & (l320 > 190)
    valid_crack_zone320 = int320 & (~spec_border320)

    ridge_map = sato(l320_clean.astype(np.float32) / 255.0, sigmas=[1.0, 1.5], black_ridges=True)
    ridge_map = np.where(valid_crack_zone320, ridge_map, 0.0)

    # Use full interior int320 for noise baseline so high-glare defect frames do not skew threshold
    bh_med = raw_med
    bh_mad = float(np.median(np.abs(int_vals.astype(np.float32) - bh_med))) if int_vals.size > 0 else 1.0
    bh_thresh = max(16.0 if not striated else 48.0, bh_med + 2.8 * max(2.0, bh_mad))

    ridge_int = ridge_map[int320]
    ridge_med = float(np.median(ridge_int)) if ridge_int.size > 0 else 0.01
    ridge_mad = float(np.median(np.abs(ridge_int - ridge_med))) if ridge_int.size > 0 else 0.005
    ridge_thresh = max(0.036 if not striated else 0.095, ridge_med + 4.0 * max(0.004, ridge_mad))

    crack_cand320 = (
        ((ridge_map > ridge_thresh) & (r_max > bh_thresh * 0.60))
        | (r_max > bh_thresh * 1.45)
    ) & valid_crack_zone320

    crack_cand320_u8 = cv2.morphologyEx(
        crack_cand320.astype(np.uint8) * 255,
        cv2.MORPH_CLOSE,
        _MORPH_3X3,
    )

    defects: list[DefectRecord] = []
    crack_mask640 = np.zeros((h, w), dtype=np.uint8)

    skel = skeletonize(crack_cand320_u8 > 0)
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
        skel.astype(np.uint8),
        connectivity=8,
    )
    min_skel320 = 50 if striated else max(22, config.min_crack_skeleton_px)

    for lbl_idx in range(1, num_lbl):
        skel_px = int(stats[lbl_idx, cv2.CC_STAT_AREA])
        bx320 = int(stats[lbl_idx, cv2.CC_STAT_LEFT])
        by320 = int(stats[lbl_idx, cv2.CC_STAT_TOP])
        bw320 = int(stats[lbl_idx, cv2.CC_STAT_WIDTH])
        bh320 = int(stats[lbl_idx, cv2.CC_STAT_HEIGHT])
        span320 = math.hypot(bw320, bh320)
        if skel_px < min_skel320 and span320 < (min_skel320 * 1.15):
            continue

        comp_mask320 = labels == lbl_idx
        ys320, xs320 = np.nonzero(comp_mask320)
        if len(xs320) < 10:
            continue

        coords = np.column_stack([xs320.astype(np.float64), ys320.astype(np.float64)])
        cov = np.cov(coords, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        lam_min = max(1e-4, float(eigvals[0]))
        lam_max = max(lam_min, float(eigvals[1]))
        ecc = math.sqrt(max(0.0, 1.0 - lam_min / lam_max))
        aspect = math.sqrt(lam_max / lam_min)
        if ecc < config.min_crack_eccentricity and aspect < 3.5:
            continue

        peak_bh = float(np.percentile(r_max[comp_mask320], 90))
        min_l = float(np.min(l320_clean[comp_mask320]))
        if min_l >= 120.0:
            continue
        if peak_bh < (18.0 if not striated else 50.0) and (bg_l320 - min_l) < 30.0:
            continue

        pad320 = 3
        rx0 = max(0, round((bx320 - pad320) * scale_inv))
        ry0 = max(0, round((by320 - pad320) * scale_inv))
        rx1 = min(w, round((bx320 + bw320 + pad320) * scale_inv))
        ry1 = min(h, round((by320 + bh320 + pad320) * scale_inv))
        bw640 = max(4, rx1 - rx0)
        bh640 = max(4, ry1 - ry0)

        comp_dil320 = cv2.dilate(comp_mask320.astype(np.uint8) * 255, _MORPH_5X5)
        comp_640 = cv2.resize(comp_dil320, (w, h), interpolation=cv2.INTER_NEAREST)
        crack_mask640 = cv2.bitwise_or(crack_mask640, comp_640)

        length_px = max(float(skel_px) * scale_inv, span320 * scale_inv)
        area_px = max(skel_px * 3, int(round(length_px * 2.5)))
        defects.append(
            DefectRecord(
                defect_class=DefectClass.CRACK,
                gost_name_ru="Трещина" if frame.stage == "bisque" else "Трещина / Цек",
                severity="critical",
                bbox_xywh=(rx0, ry0, bw640, bh640),
                area_px=area_px,
                area_mm2=round(float(area_px) / px_per_mm2, 2),
                length_mm=round(length_px / px_per_mm, 2),
                confidence=round(min(0.99, 0.70 + peak_bh / 120.0), 3),
                zone="interior",
                source="deterministic",
            )
        )

    # Compact interior defect detection (bare glaze / cavity, pinhole / speck, drip)
    cob_dil640 = cv2.dilate(frame.m_cobalt, _MORPH_5X5) > 0
    crack_dil640 = cv2.dilate(crack_mask640, _MORPH_5X5) > 0
    compact_zone = (m_int_eff > 0) & (~cob_dil640) & (~crack_dil640)

    non_glare_vals = l_raw[compact_zone & (l_raw < 248)]
    if non_glare_vals.size > 0:
        bg_l640 = float(np.median(non_glare_vals))
    elif np.any(compact_zone):
        bg_l640 = float(np.median(l_raw[compact_zone]))
    else:
        bg_l640 = 180.0

    # 1. Glaze bare / deep cavity (Выгорка / Плешина) using local wide Black-Hat at 320px
    bh_wide320 = cv2.morphologyEx(l320_clean, cv2.MORPH_BLACKHAT, _ELLIPSE_41X41)
    bh_wide640 = cv2.resize(bh_wide320, (w, h), interpolation=cv2.INTER_LINEAR)
    bh_pit = cv2.morphologyEx(l_raw, cv2.MORPH_BLACKHAT, _ELLIPSE_17X17)
    bh_pit_comb = np.maximum(bh_pit, bh_wide640)

    deep_pit_mask = (l_raw < 38) & (bh_pit_comb > (75 if striated else 55)) & compact_zone
    if frame.stage == "glazed" and not striated:
        bare_patch_mask = (
            (bh_wide640 > 38)
            & ((bg_l640 - l_raw.astype(np.float32)) > 40.0)
            & (s_raw < 50)
            & (l_raw < 170)
        ) & compact_zone
        bare_cand = deep_pit_mask | bare_patch_mask
    else:
        bare_cand = deep_pit_mask

    bare_cand_u8 = cv2.morphologyEx(bare_cand.astype(np.uint8) * 255, cv2.MORPH_CLOSE, _MORPH_3X3)
    bare_cand_u8 = cv2.morphologyEx(bare_cand_u8, cv2.MORPH_OPEN, _MORPH_3X3)

    bare_used640 = np.zeros((h, w), dtype=np.uint8)
    n_bare, lbl_bare, st_bare, _ = cv2.connectedComponentsWithStats(bare_cand_u8, connectivity=8)
    min_bare_area = 280 if striated else 240

    for idx in range(1, n_bare):
        area_px = int(st_bare[idx, cv2.CC_STAT_AREA])
        if area_px < min_bare_area:
            continue
        bx = int(st_bare[idx, cv2.CC_STAT_LEFT])
        by = int(st_bare[idx, cv2.CC_STAT_TOP])
        bw = int(st_bare[idx, cv2.CC_STAT_WIDTH])
        bh = int(st_bare[idx, cv2.CC_STAT_HEIGHT])
        comp = lbl_bare == idx
        ys_c, xs_c = np.nonzero(comp)
        if xs_c.size < 10:
            continue

        pts_c = np.column_stack([xs_c, ys_c]).astype(np.int32)
        hull_area = max(1.0, float(cv2.contourArea(cv2.convexHull(pts_c))))
        solidity = float(area_px) / hull_area
        l_min_c = float(np.min(l_raw[comp]))
        l_mean_c = float(np.mean(l_raw[comp]))
        peak_pit = float(np.percentile(bh_pit_comb[comp], 90))
        is_deep_pit = l_min_c < 35.0 and peak_pit > (55.0 if not striated else 75.0)
        if not is_deep_pit and frame.stage != "glazed":
            continue
        if not is_deep_pit and (solidity < 0.65 or max(bw, bh) / max(1, min(bw, bh)) > 2.6):
            continue

        bare_used640[comp] = 255
        area_mm2 = float(area_px) / px_per_mm2
        gost_name = "Выгорка" if is_deep_pit else "Плешина, сборка глазури"
        defects.append(
            DefectRecord(
                defect_class=DefectClass.GLAZE_BARE,
                gost_name_ru=gost_name,
                severity="critical" if is_deep_pit else "rework",
                bbox_xywh=(bx, by, bw, bh),
                area_px=area_px,
                area_mm2=round(area_mm2, 2),
                length_mm=round(max(bw, bh) / px_per_mm, 2),
                confidence=round(min(0.97, 0.68 + (bg_l640 - l_mean_c) / 140.0), 3),
                zone="interior",
                source="deterministic",
            )
        )

    # 2. Pinhole / dark speck (Накол / Мушка) with adaptive texture floor
    compact_no_bare = compact_zone & (cv2.dilate(bare_used640, _MORPH_5X5) == 0)
    bh_pin = cv2.morphologyEx(l_raw, cv2.MORPH_BLACKHAT, _ELLIPSE_9X9)
    pin_vals = bh_pin[compact_no_bare]
    pin_med = float(np.median(pin_vals)) if pin_vals.size > 0 else 4.0
    pin_mad = float(np.median(np.abs(pin_vals.astype(np.float32) - pin_med))) if pin_vals.size > 0 else 2.0
    pin_thresh = max(45.0 if not striated else 120.0, pin_med + 8.0 * max(2.5, pin_mad))
    pin_cand = (bh_pin > pin_thresh) & (l_raw < (95 if not striated else 45)) & compact_no_bare
    n_pin, lbl_pin, st_pin, _ = cv2.connectedComponentsWithStats(
        pin_cand.astype(np.uint8),
        connectivity=8,
    )
    pinhole_used640 = np.zeros((h, w), dtype=np.uint8)
    min_pin_area = 10 if striated else 4
    for idx in range(1, n_pin):
        area_px = int(st_pin[idx, cv2.CC_STAT_AREA])
        if not (min_pin_area <= area_px < min_bare_area):
            continue
        bx = int(st_pin[idx, cv2.CC_STAT_LEFT])
        by = int(st_pin[idx, cv2.CC_STAT_TOP])
        bw = int(st_pin[idx, cv2.CC_STAT_WIDTH])
        bh = int(st_pin[idx, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) / max(1, min(bw, bh)) > 3.5:
            continue
        comp = lbl_pin == idx
        pinhole_used640[comp] = 255
        area_mm2 = float(area_px) / px_per_mm2
        equiv_diam_mm = 2.0 * math.sqrt(area_mm2 / math.pi)
        gost_name = (
            "Накол"
            if (frame.stage == "glazed" and equiv_diam_mm <= 1.85 and area_px <= 22)
            else "Мушка"
        )
        defects.append(
            DefectRecord(
                defect_class=DefectClass.PINHOLE_SPECK,
                gost_name_ru=gost_name,
                severity="minor",
                bbox_xywh=(bx, by, bw, bh),
                area_px=area_px,
                area_mm2=round(area_mm2, 2),
                length_mm=round(equiv_diam_mm, 2),
                confidence=round(min(0.95, 0.66 + float(np.max(bh_pin[comp])) / 150.0), 3),
                zone="interior",
                source="deterministic",
            )
        )

    # 3. Glaze drip / thickened ridge (Натек / Прыщ, glazed stage only)
    if frame.stage == "glazed" and not striated and frame.geometry.scale_calibrated:
        compact_no_prev = compact_no_bare & (cv2.dilate(pinhole_used640, _MORPH_5X5) == 0)
        l_drop = bg_l640 - l_raw.astype(np.float32)
        b_lab = lab_roi[:, :, 2].astype(np.float32)
        bg_b = float(np.median(b_lab[compact_no_prev])) if np.any(compact_no_prev) else 128.0
        drip_cand = (
            (
                ((l_drop >= 14.0) & (l_drop <= 65.0) & (np.abs(b_lab - bg_b) >= 3.0))
                | ((bh_wide640 >= 18) & (l_drop >= 14.0) & (l_drop <= 55.0))
            )
            & (l_raw >= 100)
            & compact_no_prev
        )
        drip_u8 = cv2.morphologyEx(drip_cand.astype(np.uint8) * 255, cv2.MORPH_CLOSE, _MORPH_5X5)
        drip_u8 = cv2.morphologyEx(drip_u8, cv2.MORPH_OPEN, _MORPH_3X3)
        n_drip, lbl_drip, st_drip, _ = cv2.connectedComponentsWithStats(drip_u8, connectivity=8)
        for idx in range(1, n_drip):
            area_px = int(st_drip[idx, cv2.CC_STAT_AREA])
            if not (140 <= area_px <= 8000):
                continue
            bx = int(st_drip[idx, cv2.CC_STAT_LEFT])
            by = int(st_drip[idx, cv2.CC_STAT_TOP])
            bw = int(st_drip[idx, cv2.CC_STAT_WIDTH])
            bh = int(st_drip[idx, cv2.CC_STAT_HEIGHT])
            if max(bw, bh) / max(1, min(bw, bh)) < 1.35:
                continue
            area_mm2 = float(area_px) / px_per_mm2
            defects.append(
                DefectRecord(
                    defect_class=DefectClass.GLAZE_DRIP,
                    gost_name_ru=_GOST_NAME_BY_CLASS[DefectClass.GLAZE_DRIP],
                    severity="rework",
                    bbox_xywh=(bx, by, bw, bh),
                    area_px=area_px,
                    area_mm2=round(area_mm2, 2),
                    length_mm=round(max(bw, bh) / px_per_mm, 2),
                    confidence=0.76,
                    zone="interior",
                    source="deterministic",
                )
            )

    return defects, crack_mask640


def _apply_gost_triage(
    defects: list[DefectRecord],
    stage: StageType,
    config: DetectorConfig,
) -> tuple[VerdictType, Literal["1 сорт", "Доработка (REWORK)", "Брак (REJECT)"], list[str]]:
    """Evaluate GOST 28390-89 1st-grade limits and CeramScan operational triage."""
    rules: list[str] = []
    chip_count = sum(1 for d in defects if d.defect_class == DefectClass.CHIP)
    nakol_items = [
        d
        for d in defects
        if d.defect_class == DefectClass.PINHOLE_SPECK and d.gost_name_ru == "Накол"
    ]
    mushka_items = [
        d
        for d in defects
        if d.defect_class == DefectClass.PINHOLE_SPECK and d.gost_name_ru != "Накол"
    ]
    bare_items = [
        d
        for d in defects
        if d.defect_class == DefectClass.GLAZE_BARE and d.gost_name_ru != "Выгорка"
    ]
    drip_items = [d for d in defects if d.defect_class == DefectClass.GLAZE_DRIP]

    for d in defects:
        if d.defect_class == DefectClass.CRACK:
            d.severity = "critical"
            rules.append("ГОСТ 28390-89: трещина / цек не допускаются (REJECT)")
        elif d.defect_class == DefectClass.SHAPE_WARP:
            d.severity = "critical"
            rules.append("ГОСТ 28390-89: деформация формы > 1.5% (REJECT)")
        elif d.defect_class == DefectClass.CHIP:
            if stage == "bisque" and chip_count == 1 and d.length_mm <= 1.5:
                d.severity = "rework"
                rules.append("Утиль: мелкий скол <= 1.5 мм направлен на подшлифовку (REWORK)")
            else:
                d.severity = "critical"
                rules.append("ГОСТ 28390-89: скол / откол кромки (REJECT)")
        elif d.defect_class == DefectClass.GLAZE_BARE:
            if d.gost_name_ru == "Выгорка":
                if stage == "bisque" and d.area_mm2 <= 10.0 and len(defects) == 1:
                    d.severity = "rework"
                    rules.append("Утиль: мелкая выгорка <= 10 мм2 на зачистку (REWORK)")
                else:
                    d.severity = "critical"
                    rules.append("ГОСТ 28390-89: выгорка не допускается (REJECT)")
            else:
                if len(bare_items) == 1 and d.area_mm2 <= config.max_glaze_bare_mm2:
                    d.severity = "minor"
                    rules.append("ГОСТ 28390-89: плешина <= 20 мм2 в допуске 1 сорта")
                elif len(bare_items) == 1 and d.area_mm2 <= 100.0:
                    d.severity = "rework"
                    rules.append("Плешина > 20 мм2: повторное глазурование (REWORK)")
                else:
                    d.severity = "critical"
                    rules.append("ГОСТ 28390-89: множественная или крупная плешина (REJECT)")
        elif d.defect_class == DefectClass.PINHOLE_SPECK:
            if d.gost_name_ru == "Накол":
                if len(nakol_items) > config.max_pinhole_count:
                    d.severity = "critical"
                    rules.append("ГОСТ 28390-89: накол > 8 шт. (REJECT)")
                else:
                    d.severity = "minor"
                    rules.append("ГОСТ 28390-89: накол <= 8 шт. (1 группа недочетов)")
            else:
                if stage == "bisque":
                    if len(mushka_items) <= 2:
                        d.severity = "rework"
                        rules.append("Утиль: мушка <= 2 шт. на зачистку до обжига (REWORK)")
                    else:
                        d.severity = "critical"
                        rules.append("Утиль: > 2 темных включений (REJECT)")
                else:
                    if len(mushka_items) <= 2 and d.length_mm <= 0.5:
                        d.severity = "minor"
                        rules.append("ГОСТ 28390-89: мушка <= 0.5 мм (<= 2 шт.) в допуске")
                    elif len(mushka_items) <= 2 and d.length_mm <= 1.5:
                        d.severity = "rework"
                        rules.append("Мушка 0.5-1.5 мм: доработка / 2 сорт (REWORK)")
                    else:
                        d.severity = "critical"
                        rules.append("ГОСТ 28390-89: мушка > 2 шт. или > 1.5 мм (REJECT)")
        elif d.defect_class == DefectClass.GLAZE_DRIP:
            if len(drip_items) == 1 and d.area_mm2 <= 100.0:
                d.severity = "rework"
                rules.append("Локальный натек глазури: зашлифовка (REWORK)")
            else:
                d.severity = "critical"
                rules.append("ГОСТ 28390-89: множественный или грубый натек (REJECT)")

    non_nakol_groups = sum(
        1
        for d in defects
        if not (d.defect_class == DefectClass.PINHOLE_SPECK and d.gost_name_ru == "Накол")
    )
    gost_groups = non_nakol_groups + (1 if nakol_items else 0)

    has_critical = any(d.severity == "critical" for d in defects)
    has_rework = any(d.severity == "rework" for d in defects)

    if has_critical:
        verdict: VerdictType = "REJECT"
    elif gost_groups > config.max_gost_defects_grade1:
        rules.append(
            f"ГОСТ 28390-89: превышен лимит <= {config.max_gost_defects_grade1} недочетов"
        )
        sandable_classes = {DefectClass.CHIP, DefectClass.PINHOLE_SPECK}
        if stage == "bisque" and all(d.defect_class in sandable_classes for d in defects):
            verdict = "REWORK"
        else:
            verdict = "REJECT"
    elif has_rework:
        verdict = "REWORK"
    else:
        verdict = "PASS"

    if verdict == "PASS" and not rules:
        rules.append("ГОСТ 28390-89: дефекты 1 сорта не выявлены")

    dedup_rules = list(dict.fromkeys(rules))
    grade_map: dict[VerdictType, Literal["1 сорт", "Доработка (REWORK)", "Брак (REJECT)"]] = {
        "PASS": "1 сорт",
        "REWORK": "Доработка (REWORK)",
        "REJECT": "Брак (REJECT)",
    }
    return verdict, grade_map[verdict], dedup_rules


def _render_overlay(
    frame: PreprocessedFrame,
    defects: list[DefectRecord],
    verdict: VerdictType,
    gost_grade: str,
    crack_mask: np.ndarray | None,
) -> tuple[np.ndarray, bytes]:
    """Render annotated overlay BGR image with Cyrillic Pillow labels and PNG bytes."""
    overlay_bgr = frame.roi_bgr.copy()
    h, w = overlay_bgr.shape[:2]

    if crack_mask is not None and np.any(crack_mask):
        crack_bool = crack_mask > 0
        overlay_bgr[crack_bool] = (
            0.45 * overlay_bgr[crack_bool].astype(np.float32)
            + 0.55 * np.array([38.0, 38.0, 220.0], dtype=np.float32)
        ).astype(np.uint8)

    if frame.geometry.scale_calibrated:
        contours, _ = cv2.findContours(frame.m_item, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, contours, -1, (216, 78, 29), 1, cv2.LINE_AA)

    color_by_sev: dict[str, tuple[int, int, int]] = {
        "critical": (38, 38, 220),
        "rework": (9, 83, 180),
        "minor": (216, 78, 29),
    }
    rgb_by_sev: dict[str, tuple[int, int, int]] = {
        "critical": (220, 38, 38),
        "rework": (180, 83, 9),
        "minor": (29, 78, 216),
    }

    for d in defects:
        bx, by, bw, bh = d.bbox_xywh
        bgr = color_by_sev[d.severity]
        cv2.rectangle(overlay_bgr, (bx, by), (bx + bw, by + bh), bgr, 2, cv2.LINE_AA)

    pil_img = Image.fromarray(cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    banner_font = get_cyrillic_font(size=16, bold=True)
    tag_font = get_cyrillic_font(size=13, bold=True)

    verdict_rgb = {
        "PASS": (6, 95, 70),
        "REWORK": (146, 64, 14),
        "REJECT": (153, 27, 27),
    }[verdict]
    stage_label = "Утиль 900C" if frame.stage == "bisque" else "Глазурь 1350C"
    header_text = f"CeramScan | {stage_label} | {verdict}: {gost_grade} | Дефекты: {len(defects)}"
    draw.rectangle([(0, 0), (w, 34)], fill=(248, 250, 252))
    draw.line([(0, 34), (w, 34)], fill=(203, 213, 225), width=1)
    draw.text((12, 8), header_text, font=banner_font, fill=verdict_rgb)

    for idx, d in enumerate(defects[:8]):
        bx, by, _, _ = d.bbox_xywh
        tag_text = f"{idx + 1}. {_SHORT_LABEL_BY_CLASS[d.defect_class]}"
        t_box = draw.textbbox((0, 0), tag_text, font=tag_font)
        tw = t_box[2] - t_box[0] + 10
        th = t_box[3] - t_box[1] + 6
        tx = max(4, min(w - tw - 4, bx))
        ty = max(38, by - th - 2)
        fill_rgb = rgb_by_sev[d.severity]
        draw.rectangle([(tx, ty), (tx + tw, ty + th)], fill=fill_rgb)
        draw.text((tx + 5, ty + 2), tag_text, font=tag_font, fill=(255, 255, 255))

    annotated_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", annotated_bgr)
    if not ok:
        raise RuntimeError("Failed to encode inspection overlay PNG.")
    return annotated_bgr, bytes(buf)


class DefectDetector:
    """Two-stage ceramic defect detector with deterministic CPU core and ONNX/YOLO hook."""

    def __init__(
        self,
        weights_path: Path | None = None,
        config: DetectorConfig | None = None,
        preprocess_config: PreprocessConfig | None = None,
    ) -> None:
        self.config = config if config is not None else DetectorConfig()
        self.preprocess_config = (
            preprocess_config if preprocess_config is not None else PreprocessConfig()
        )
        self.weights_path: Path | None = Path(weights_path) if weights_path is not None else None
        self._session: ort.InferenceSession | None = None
        self._input_name: str = ""
        self._input_hw: tuple[int, int] = (640, 640)
        self._class_map: dict[int, DefectClass] = dict(_CLASS_BY_INDEX)

        if self.weights_path is not None:
            if not self.weights_path.is_file():
                raise FileNotFoundError(f"ONNX weights file not found: {self.weights_path}")
            if self.weights_path.suffix.lower() != ".onnx":
                raise ValueError(f"ONNX weights file must end with .onnx: {self.weights_path}")
            try:
                self._session = ort.InferenceSession(
                    str(self.weights_path),
                    providers=["CPUExecutionProvider"],
                )
            except _ORT_LOAD_ERRORS as exc:
                raise ValueError(f"Failed to load ONNX weights {self.weights_path}: {exc}") from exc

            inputs = self._session.get_inputs()
            if not inputs:
                raise ValueError("ONNX model exposes zero inputs.")
            inp = inputs[0]
            self._input_name = inp.name
            if len(inp.shape) == 4:
                h_dim, w_dim = inp.shape[2], inp.shape[3]
                if isinstance(h_dim, int) and h_dim > 0 and isinstance(w_dim, int) and w_dim > 0:
                    self._input_hw = (h_dim, w_dim)

            meta = self._session.get_modelmeta()
            raw_names = meta.custom_metadata_map.get("names")
            if raw_names:
                try:
                    parsed = ast.literal_eval(raw_names)
                except (SyntaxError, ValueError):
                    parsed = None
                parsed_map: dict[int, DefectClass] = {}
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        try:
                            idx = int(k)
                        except (TypeError, ValueError):
                            continue
                        if idx >= 0:
                            parsed_map[idx] = _map_name_to_class(str(v), idx)
                elif isinstance(parsed, (list, tuple)):
                    for idx, name in enumerate(parsed):
                        parsed_map[idx] = _map_name_to_class(str(name), idx)
                if parsed_map:
                    self._class_map = parsed_map

    def _infer_onnx(self, frame: PreprocessedFrame) -> list[DefectRecord]:
        if self._session is None:
            raise RuntimeError("ONNX session is not initialized.")
        in_h, in_w = self._input_hw
        roi_h, roi_w = frame.roi_bgr.shape[:2]
        lb_bgr, scale, (pad_x, pad_y) = _letterbox_bgr(frame.roi_bgr, (in_h, in_w))
        rgb = cv2.cvtColor(lb_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None, :, :, :])

        raw_out = np.asarray(self._session.run(None, {self._input_name: tensor})[0])
        if raw_out.ndim != 3 or raw_out.shape[0] != 1:
            raise ValueError(f"Unsupported ONNX output shape: {raw_out.shape}")

        num_classes = len(self._class_map)
        d1, d2 = raw_out.shape[1], raw_out.shape[2]
        if d1 == 4 + num_classes and d2 == 5 + num_classes:
            v8_slice = raw_out[0, 4:, :]
            v5_slice = raw_out[0, :, 4:]
            v8_fin = v8_slice[np.isfinite(v8_slice)]
            v5_fin = v5_slice[np.isfinite(v5_slice)]
            v8_scores_unit = bool(
                v8_fin.size > 0 and np.all((v8_fin >= 0.0) & (v8_fin <= 1.0))
            )
            v5_scores_unit = bool(
                v5_fin.size > 0 and np.all((v5_fin >= 0.0) & (v5_fin <= 1.0))
            )
            if v8_scores_unit and not v5_scores_unit:
                use_v8 = True
            elif v5_scores_unit and not v8_scores_unit:
                use_v8 = False
            else:
                v8_sub = raw_out[0, 4:, :4]
                v5_sub = raw_out[0, :4, 4:]
                v8_sub_fin = v8_sub[np.isfinite(v8_sub)]
                v5_sub_fin = v5_sub[np.isfinite(v5_sub)]
                v8_only_active = float(np.max(v8_sub_fin)) if v8_sub_fin.size > 0 else 0.0
                v5_only_active = float(np.max(v5_sub_fin)) if v5_sub_fin.size > 0 else 0.0
                use_v8 = v8_scores_unit and (v8_only_active > v5_only_active)
            if use_v8:
                rows = raw_out[0].T
                boxes_xywh = rows[:, :4]
                scores_mat = rows[:, 4:]
            else:
                rows = raw_out[0]
                boxes_xywh = rows[:, :4]
                scores_mat = rows[:, 4:5] * rows[:, 5:]
        elif d1 == 4 + num_classes:
            rows = raw_out[0].T
            boxes_xywh = rows[:, :4]
            scores_mat = rows[:, 4:]
        elif d2 == 5 + num_classes:
            rows = raw_out[0]
            boxes_xywh = rows[:, :4]
            scores_mat = rows[:, 4:5] * rows[:, 5:]
        else:
            raise ValueError(
                f"Unsupported YOLO channel layout {raw_out.shape} for {num_classes} classes."
            )

        if rows.shape[0] == 0:
            return []

        finite_box_mask = np.all(np.isfinite(boxes_xywh), axis=1)
        if not np.any(finite_box_mask):
            return []
        boxes_xywh = boxes_xywh[finite_box_mask]
        scores_mat = scores_mat[finite_box_mask]

        clean_scores = np.where(np.isfinite(scores_mat), scores_mat, -1.0)
        class_ids = np.argmax(clean_scores, axis=1)
        raw_confs = clean_scores[np.arange(clean_scores.shape[0]), class_ids]
        confidences = np.where(raw_confs > 1.0, 1.0, raw_confs)
        min_conf = max(0.0, float(self.config.conf_threshold))
        keep_mask = np.isfinite(confidences) & (confidences >= min_conf) & (confidences <= 1.0)
        if not np.any(keep_mask):
            return []

        filt_boxes = boxes_xywh[keep_mask]
        filt_confs = confidences[keep_mask]
        filt_cls = class_ids[keep_mask]

        nms_boxes: list[list[float]] = []
        xywh_canonical: list[tuple[int, int, int, int]] = []
        for xc, yc, bw, bh in filt_boxes:
            x1 = max(0.0, min(float(roi_w - 1), (float(xc) - float(bw) / 2.0 - pad_x) / scale))
            y1 = max(0.0, min(float(roi_h - 1), (float(yc) - float(bh) / 2.0 - pad_y) / scale))
            cw = max(1.0, min(float(roi_w) - x1, float(bw) / scale))
            ch = max(1.0, min(float(roi_h) - y1, float(bh) / scale))
            nms_boxes.append([x1, y1, cw, ch])
            xywh_canonical.append((int(round(x1)), int(round(y1)), int(round(cw)), int(round(ch))))

        raw_idxs = cv2.dnn.NMSBoxes(
            nms_boxes,
            [float(c) for c in filt_confs],
            self.config.conf_threshold,
            self.config.nms_iou_threshold,
        )
        kept_idxs = np.asarray(raw_idxs, dtype=np.int64).ravel().tolist()

        px_per_mm = max(0.5, frame.geometry.px_per_mm)
        px_per_mm2 = px_per_mm * px_per_mm
        ordered_cls_keys = sorted(self._class_map.keys())
        defects: list[DefectRecord] = []
        for k_idx in kept_idxs:
            cls_id = int(filt_cls[k_idx])
            if 0 <= cls_id < len(ordered_cls_keys):
                d_cls = self._class_map[ordered_cls_keys[cls_id]]
            elif cls_id in self._class_map:
                d_cls = self._class_map[cls_id]
            else:
                d_cls = _CLASS_BY_INDEX.get(cls_id, DefectClass.CRACK)
            bx, by, bw, bh = xywh_canonical[k_idx]
            area_px = max(1, bw * bh)
            area_mm2 = round(float(area_px) / px_per_mm2, 2)
            length_mm = round(max(bw, bh) / px_per_mm, 2)
            gost_name_ru = (
                "Накол"
                if (
                    d_cls == DefectClass.PINHOLE_SPECK
                    and frame.stage == "glazed"
                    and length_mm <= 1.5
                    and area_mm2 <= 2.5
                )
                else _GOST_NAME_BY_CLASS[d_cls]
            )
            zone: Literal["rim", "interior"] = (
                "rim" if d_cls in (DefectClass.CHIP, DefectClass.SHAPE_WARP) else "interior"
            )
            conf_clamped = round(float(min(1.0, max(0.0, filt_confs[k_idx]))), 3)
            defects.append(
                DefectRecord(
                    defect_class=d_cls,
                    gost_name_ru=gost_name_ru,
                    severity="critical",
                    bbox_xywh=(bx, by, bw, bh),
                    area_px=area_px,
                    area_mm2=area_mm2,
                    length_mm=length_mm,
                    confidence=conf_clamped,
                    zone=zone,
                    source="onnx",
                )
            )
        return defects

    def inspect(
        self,
        image_bgr: np.ndarray,
        stage: StageType = "glazed",
        reference_diameter_mm: float = 200.0,
    ) -> InspectionResult:
        """Execute preprocessing, defect detection, GOST 28390-89 triage, and overlay render."""
        if not math.isfinite(reference_diameter_mm) or reference_diameter_mm <= 0.0:
            raise ValueError(
                f"reference_diameter_mm must be finite and > 0, got {reference_diameter_mm!r}."
            )
        t0 = time.perf_counter()
        pp_cfg = self.preprocess_config
        if reference_diameter_mm != pp_cfg.reference_diameter_mm:
            pp_cfg = PreprocessConfig(
                target_size=pp_cfg.target_size,
                clahe_clip_bisque=pp_cfg.clahe_clip_bisque,
                clahe_clip_glazed=pp_cfg.clahe_clip_glazed,
                clahe_grid=pp_cfg.clahe_grid,
                glare_l_thresh=pp_cfg.glare_l_thresh,
                glare_s_thresh=pp_cfg.glare_s_thresh,
                rim_band_ratio=pp_cfg.rim_band_ratio,
                cobalt_s_min=pp_cfg.cobalt_s_min,
                cobalt_h_range=pp_cfg.cobalt_h_range,
                cobalt_b_max=pp_cfg.cobalt_b_max,
                reference_diameter_mm=reference_diameter_mm,
                min_item_area_ratio=pp_cfg.min_item_area_ratio,
            )

        frame = preprocess_image(image_bgr, stage=stage, config=pp_cfg)
        model_mode: Literal["deterministic", "onnx"] = (
            "onnx" if self._session is not None else "deterministic"
        )

        crack_mask: np.ndarray | None = None
        if model_mode == "onnx":
            defects = self._infer_onnx(frame)
            rim_defects, contour_dev_pct = _detect_rim_chip_and_warp(
                frame,
                self.config,
                source="onnx",
            )
            has_onnx_warp = any(d.defect_class == DefectClass.SHAPE_WARP for d in defects)
            for rd in rim_defects:
                if rd.defect_class == DefectClass.SHAPE_WARP and not has_onnx_warp:
                    defects.append(rd)
        else:
            rim_defects, contour_dev_pct = _detect_rim_chip_and_warp(
                frame,
                self.config,
                source="deterministic",
            )
            surface_defects, crack_mask = _detect_deterministic_surface_defects(
                frame,
                self.config,
                rim_defects=rim_defects,
            )
            defects = [*surface_defects, *rim_defects]

        verdict, gost_grade, gost_rules = _apply_gost_triage(defects, stage, self.config)
        defects.sort(
            key=lambda d: (_SEVERITY_RANK.get(d.severity, 3), -d.area_px, -d.confidence),
        )
        overlay_bgr, overlay_png_bytes = _render_overlay(
            frame,
            defects,
            verdict,
            gost_grade,
            crack_mask,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return InspectionResult(
            stage=stage,
            verdict=verdict,
            gost_grade=gost_grade,
            model_mode=model_mode,
            contour_deviation_pct=contour_dev_pct,
            defect_count=len(defects),
            defects=defects,
            gost_rules_triggered=gost_rules,
            glare_ratio=frame.glare_ratio,
            cobalt_ratio=frame.cobalt_ratio,
            scale_calibrated=frame.geometry.scale_calibrated,
            latency_ms=latency_ms,
            overlay_bgr=overlay_bgr,
            overlay_png_bytes=overlay_png_bytes,
        )
