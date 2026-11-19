"""Ceramic ROI segmentation, perspective rectification, LAB CLAHE, and pigment/glare masking."""

from __future__ import annotations

import functools
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import ImageFont

StageType = Literal["bisque", "glazed"]
VerdictType = Literal["PASS", "REWORK", "REJECT"]
AuthVerdictType = Literal["Подлинник", "Подозрение на контрафакт или штамповку"]

VALID_STAGES: frozenset[str] = frozenset({"bisque", "glazed"})
_MORPH_3X3 = np.ones((3, 3), dtype=np.uint8)
_MORPH_7X7_ELLIPSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


class DefectClass(str, Enum):
    CRACK = "crack"
    CHIP = "chip"
    GLAZE_BARE = "glaze_bare"
    PINHOLE_SPECK = "pinhole_speck"
    GLAZE_DRIP = "glaze_drip"
    SHAPE_WARP = "shape_warp"


@dataclass(slots=True)
class PreprocessConfig:
    target_size: int = 640
    clahe_clip_bisque: float = 2.0
    clahe_clip_glazed: float = 1.8
    clahe_grid: tuple[int, int] = (8, 8)
    glare_l_thresh: int = 235
    glare_s_thresh: int = 30
    rim_band_ratio: float = 0.055
    cobalt_s_min: int = 45
    cobalt_h_range: tuple[int, int] = (90, 135)
    cobalt_b_max: int = 118
    reference_diameter_mm: float = 200.0
    min_item_area_ratio: float = 0.12


@dataclass(slots=True)
class EllipseGeometry:
    center: tuple[float, float]
    axes: tuple[float, float]
    angle_deg: float
    contour_deviation_pct: float
    axis_ratio_error_pct: float
    px_per_mm: float
    scale_calibrated: bool


@dataclass(slots=True)
class PreprocessedFrame:
    stage: StageType
    original_bgr: np.ndarray
    roi_bgr: np.ndarray
    roi_lab_l_clahe: np.ndarray
    roi_hsv: np.ndarray
    m_item: np.ndarray
    m_rim: np.ndarray
    m_interior: np.ndarray
    m_spec: np.ndarray
    m_cobalt: np.ndarray
    geometry: EllipseGeometry
    perspective_applied: bool
    glare_ratio: float
    cobalt_ratio: float
    latency_ms: float


@functools.lru_cache(maxsize=24)
def get_cyrillic_font(
    size: int = 18,
    bold: bool = False,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load and cache a TrueType Cyrillic-capable font across macOS and Linux."""
    candidates: tuple[tuple[str, int], ...]
    if bold:
        candidates = (
            ("/System/Library/Fonts/Supplemental/PTSans.ttc", 7),
            ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 0),
            ("/System/Library/Fonts/Supplemental/PTSans.ttc", 0),
            ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 0),
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 0),
            ("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 0),
            ("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 0),
        )
    else:
        candidates = (
            ("/System/Library/Fonts/Supplemental/PTSans.ttc", 0),
            ("/System/Library/Fonts/Supplemental/Arial.ttf", 0),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
            ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 0),
            ("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 0),
        )

    for font_path, index in candidates:
        if Path(font_path).is_file():
            try:
                return ImageFont.truetype(font_path, size=size, index=index)
            except OSError:
                continue
    return ImageFont.load_default()


def _order_quad_clockwise(pts: np.ndarray) -> np.ndarray:
    """Sort 4 quad vertices into [top-left, top-right, bottom-right, bottom-left]."""
    pts_f = pts.reshape(4, 2).astype(np.float32)
    centroid = np.mean(pts_f, axis=0)
    angles = np.arctan2(pts_f[:, 1] - centroid[1], pts_f[:, 0] - centroid[0])
    ordered = pts_f[np.argsort(angles)]
    tl_idx = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -tl_idx, axis=0)


def _detect_calibration_quad(det_bgr: np.ndarray) -> np.ndarray | None:
    """Detect a convex 4-vertex calibration card covering 20%..90% of the proxy frame."""
    det_h, det_w = det_bgr.shape[:2]
    det_area = float(det_h * det_w)
    gray = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(blur, 40, 130), _MORPH_3X3)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    for cnt in top_contours:
        area = float(cv2.contourArea(cnt))
        if not (0.20 * det_area <= area <= 0.90 * det_area):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        _, _, bw, bh = cv2.boundingRect(approx)
        rect_fill = area / float(max(1, bw * bh))
        if rect_fill <= 0.55:
            continue
        return _order_quad_clockwise(approx)
    return None


def _segment_bounded_item(
    det_bgr: np.ndarray,
    min_item_area_ratio: float,
    border_inset_px: int = 0,
) -> tuple[bool, np.ndarray, tuple[int, int, int, int], np.ndarray]:
    """Run perimeter-MAD LAB L1 + low-spread rim Canny segmentation with bounded-item gate."""
    det_h, det_w = det_bgr.shape[:2]
    det_area = float(det_h * det_w)

    lab_det = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
    p = max(4, min(det_h, det_w) // 40)
    ins = min(border_inset_px, min(det_h, det_w) // 8)
    s_off = max(6, ins * 3) if ins > 0 else 0
    border_pts = np.concatenate(
        [
            lab_det[s_off : s_off + p, s_off : det_w - s_off, :].reshape(-1, 3),
            lab_det[det_h - s_off - p : det_h - s_off, s_off : det_w - s_off, :].reshape(-1, 3),
            lab_det[s_off : det_h - s_off, s_off : s_off + p, :].reshape(-1, 3),
            lab_det[s_off : det_h - s_off, det_w - s_off - p : det_w - s_off, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_med = np.median(border_pts, axis=0)
    bg_spread = float(np.sum(np.median(np.abs(border_pts - bg_med), axis=0)))
    dist_l1 = np.sum(np.abs(lab_det - bg_med), axis=2)
    thresh = max(12.0, 2.8 * bg_spread + 8.0)
    fg = np.where(dist_l1 > thresh, 255, 0).astype(np.uint8)
    if ins > 0:
        fg[:ins, :] = 0
        fg[-ins:, :] = 0
        fg[:, :ins] = 0
        fg[:, -ins:] = 0

    if bg_spread < 8.0 and ins == 0:
        gray_det = cv2.cvtColor(det_bgr, cv2.COLOR_BGR2GRAY)
        rim_edges = cv2.Canny(cv2.GaussianBlur(gray_det, (5, 5), 0), 20, 60)
        fg = cv2.bitwise_or(fg, cv2.dilate(rim_edges, _MORPH_3X3))

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _MORPH_7X7_ELLIPSE)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _MORPH_3X3)

    contours, hierarchy = cv2.findContours(fg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        empty = np.zeros((det_h, det_w), dtype=np.uint8)
        return False, empty, (0, 0, det_w, det_h), border_pts

    h_arr = hierarchy[0]
    cb = max(4, min(det_h, det_w) // 25)
    best_area = -1.0
    best_mask: np.ndarray | None = None
    best_bbox: tuple[int, int, int, int] = (0, 0, det_w, det_h)

    for idx, cnt in enumerate(contours):
        depth = 0
        parent = int(h_arr[idx][3])
        while parent >= 0:
            depth += 1
            parent = int(h_arr[parent][3])
        if depth % 2 != 0 or len(cnt) < 5:
            continue

        area = float(cv2.contourArea(cnt))
        area_ratio = area / det_area
        if not (min_item_area_ratio <= area_ratio <= 0.92):
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        touches = (
            int(bx <= 1)
            + int(by <= 1)
            + int(bx + bw >= det_w - 1)
            + int(by + bh >= det_h - 1)
        )
        if touches >= 3:
            continue

        cand_mask = np.zeros((det_h, det_w), dtype=np.uint8)
        cv2.drawContours(cand_mask, [cnt], -1, 255, thickness=-1)

        rect_fill = area / float(max(1, bw * bh))
        if ins > 0 and area_ratio > 0.75 and rect_fill >= 0.88:
            has_large_hole = False
            child = int(h_arr[idx][2])
            while child >= 0:
                if float(cv2.contourArea(contours[child])) >= 0.50 * area:
                    has_large_hole = True
                    break
                child = int(h_arr[child][0])
            if has_large_hole:
                fg_fill = float(np.count_nonzero(fg & cand_mask)) / max(1.0, area)
                if fg_fill < 0.85:
                    continue

        corners_fg = (
            int(float(np.mean(cand_mask[:cb, :cb])) > 127.0)
            + int(float(np.mean(cand_mask[:cb, -cb:])) > 127.0)
            + int(float(np.mean(cand_mask[-cb:, :cb])) > 127.0)
            + int(float(np.mean(cand_mask[-cb:, -cb:])) > 127.0)
        )
        if corners_fg > 1:
            continue

        hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
        solidity = area / max(1.0, hull_area)
        if solidity < 0.78:
            continue

        if area > best_area:
            best_area = area
            best_mask = cand_mask
            best_bbox = (bx, by, bw, bh)

    if best_mask is None:
        empty = np.zeros((det_h, det_w), dtype=np.uint8)
        return False, empty, (0, 0, det_w, det_h), border_pts
    return True, best_mask, best_bbox, border_pts


def _crop_square_with_pad(
    img: np.ndarray,
    x0: int,
    y0: int,
    side: int,
    pad_bgr: tuple[int, int, int],
) -> np.ndarray:
    """Crop a square region [x0:x0+side, y0:y0+side] padding out-of-bounds areas safely."""
    h, w = img.shape[:2]
    x1, y1 = x0 + side, y0 + side
    if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
        return img[y0:y1, x0:x1]

    canvas = np.full((side, side, 3), pad_bgr, dtype=img.dtype)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = img[sy0:sy1, sx0:sx1]
    return canvas


def _crop_square_mask_with_pad(
    mask: np.ndarray,
    x0: int,
    y0: int,
    side: int,
) -> np.ndarray:
    """Crop a single-channel square mask padding out-of-bounds regions with 0."""
    h, w = mask.shape[:2]
    x1, y1 = x0 + side, y0 + side
    if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
        return mask[y0:y1, x0:x1]

    canvas = np.zeros((side, side), dtype=mask.dtype)
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        canvas[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = mask[sy0:sy1, sx0:sx1]
    return canvas


def _compute_ellipse_geometry(
    m_item: np.ndarray,
    reference_diameter_mm: float,
    target_size: int,
) -> tuple[EllipseGeometry, float]:
    """Fit ellipse to canonical m_item and compute radial deviation + axis ratio error."""
    contours, _ = cv2.findContours(m_item, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        half = target_size / 2.0
        geom = EllipseGeometry(
            center=(half, half),
            axes=(float(target_size), float(target_size)),
            angle_deg=0.0,
            contour_deviation_pct=0.0,
            axis_ratio_error_pct=0.0,
            px_per_mm=float(target_size) / reference_diameter_mm,
            scale_calibrated=True,
        )
        return geom, float(target_size)

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        x, y, bw, bh = cv2.boundingRect(cnt)
        major_px = float(max(bw, bh, 1))
        minor_px = float(min(bw, bh, 1))
        geom = EllipseGeometry(
            center=(x + bw / 2.0, y + bh / 2.0),
            axes=(major_px, minor_px),
            angle_deg=0.0,
            contour_deviation_pct=0.0,
            axis_ratio_error_pct=abs(1.0 - major_px / max(1.0, minor_px)) * 100.0,
            px_per_mm=major_px / reference_diameter_mm,
            scale_calibrated=True,
        )
        return geom, minor_px

    (cx, cy), (w_ax, h_ax), angle_deg = cv2.fitEllipse(cnt)
    major_px = float(max(w_ax, h_ax, 1.0))
    minor_px = float(max(1.0, min(w_ax, h_ax)))

    pts = cnt.reshape(-1, 2).astype(np.float64)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    theta = np.radians(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    lx = dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t

    rx = max(1.0, w_ax / 2.0)
    ry = max(1.0, h_ax / 2.0)
    phi = np.arctan2(ly, lx)
    r_actual = np.hypot(lx, ly)
    r_ellipse = (rx * ry) / np.maximum(1e-6, np.hypot(ry * np.cos(phi), rx * np.sin(phi)))
    mean_radius = max(1.0, float(np.mean(r_ellipse)))
    contour_deviation_pct = float(np.max(np.abs(r_actual - r_ellipse)) / mean_radius * 100.0)
    axis_ratio_error_pct = float(abs(1.0 - major_px / minor_px) * 100.0)

    geom = EllipseGeometry(
        center=(float(cx), float(cy)),
        axes=(major_px, minor_px),
        angle_deg=float(angle_deg),
        contour_deviation_pct=round(contour_deviation_pct, 4),
        axis_ratio_error_pct=round(axis_ratio_error_pct, 4),
        px_per_mm=major_px / reference_diameter_mm,
        scale_calibrated=True,
    )
    return geom, minor_px


def _build_item_rim_and_interior(
    m_item: np.ndarray,
    minor_px: float,
    rim_band_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute disjoint m_rim and m_interior masks in a single distanceTransform pass."""
    dist = cv2.distanceTransform(m_item, cv2.DIST_L2, 3)
    rim_px = max(7.0, minor_px * rim_band_ratio)
    m_interior = np.where(dist >= rim_px, 255, 0).astype(np.uint8)
    m_rim = np.where((dist > 0.0) & (dist < rim_px), 255, 0).astype(np.uint8)
    return m_rim, m_interior


def _macro_fallback(
    work_bgr: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, EllipseGeometry]:
    """Resize aspect-preservingly onto zero-padded canvas with 5% interior inset."""
    wh, ww = work_bgr.shape[:2]
    scale = target_size / float(max(wh, ww))
    nw = max(1, min(target_size, round(ww * scale)))
    nh = max(1, min(target_size, round(wh * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(work_bgr, (nw, nh), interpolation=interp)

    roi_bgr = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x0 = (target_size - nw) // 2
    y0 = (target_size - nh) // 2
    roi_bgr[y0 : y0 + nh, x0 : x0 + nw] = resized

    m_item = np.zeros((target_size, target_size), dtype=np.uint8)
    m_item[y0 : y0 + nh, x0 : x0 + nw] = 255

    ins = max(16, round(min(nw, nh) * 0.05))
    m_interior = np.zeros((target_size, target_size), dtype=np.uint8)
    ix0 = min(x0 + nw, x0 + ins)
    ix1 = max(ix0, x0 + nw - ins)
    iy0 = min(y0 + nh, y0 + ins)
    iy1 = max(iy0, y0 + nh - ins)
    if ix1 > ix0 and iy1 > iy0:
        m_interior[iy0:iy1, ix0:ix1] = 255
    else:
        m_interior[y0 : y0 + nh, x0 : x0 + nw] = 255

    m_rim = np.zeros((target_size, target_size), dtype=np.uint8)
    geom = EllipseGeometry(
        center=(target_size / 2.0, target_size / 2.0),
        axes=(float(nw), float(nh)),
        angle_deg=0.0,
        contour_deviation_pct=0.0,
        axis_ratio_error_pct=0.0,
        px_per_mm=float(max(nw, nh)) / 160.0,
        scale_calibrated=False,
    )
    return roi_bgr, m_item, m_rim, m_interior, geom


def preprocess_image(
    image_bgr: np.ndarray,
    stage: StageType = "glazed",
    config: PreprocessConfig | None = None,
) -> PreprocessedFrame:
    """Locate ceramic ROI, normalize geometry/lighting, and build semantic masks."""
    if stage not in VALID_STAGES:
        raise ValueError(f"Unsupported stage '{stage}'. Expected 'bisque' or 'glazed'.")
    if (
        image_bgr is None
        or not isinstance(image_bgr, np.ndarray)
        or image_bgr.size == 0
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
    ):
        raise ValueError("image_bgr must be a non-empty HxWx3 BGR numpy array.")
    if image_bgr.dtype != np.uint8:
        raise ValueError(f"image_bgr must have dtype uint8, got {image_bgr.dtype}.")

    h, w = image_bgr.shape[:2]
    if min(h, w) < 32:
        raise ValueError(f"Image dimensions too small ({w}x{h}); minimum is 32x32.")

    t0 = time.perf_counter()
    cfg = config if config is not None else PreprocessConfig()
    target_size = cfg.target_size
    if (
        not isinstance(target_size, int)
        or isinstance(target_size, bool)
        or target_size < 64
    ):
        raise ValueError(f"target_size must be an integer >= 64, got {target_size!r}.")
    if not math.isfinite(cfg.reference_diameter_mm) or cfg.reference_diameter_mm <= 0.0:
        raise ValueError(
            f"reference_diameter_mm must be finite and > 0, got {cfg.reference_diameter_mm!r}."
        )

    if max(h, w) > 1600:
        cap_scale = 1600.0 / float(max(h, w))
        original_bgr = cv2.resize(
            image_bgr,
            (max(32, round(w * cap_scale)), max(32, round(h * cap_scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        original_bgr = image_bgr.copy()

    wh, ww = original_bgr.shape[:2]
    det_scale = min(1.0, 320.0 / float(max(wh, ww)))
    if det_scale < 1.0:
        det_bgr = cv2.resize(
            original_bgr,
            (max(32, round(ww * det_scale)), max(32, round(wh * det_scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        det_bgr = original_bgr

    quad = _detect_calibration_quad(det_bgr)
    perspective_applied = False

    if quad is not None:
        src_pts = (quad / det_scale).astype(np.float32)
        dst_pts = np.array(
            [
                [0.0, 0.0],
                [float(target_size - 1), 0.0],
                [float(target_size - 1), float(target_size - 1)],
                [0.0, float(target_size - 1)],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        roi_bgr = cv2.warpPerspective(
            original_bgr,
            matrix,
            (target_size, target_size),
            flags=cv2.INTER_LINEAR,
        )
        perspective_applied = True

        warp_det_scale = min(1.0, 320.0 / float(target_size))
        warp_det = cv2.resize(
            roi_bgr,
            (round(target_size * warp_det_scale), round(target_size * warp_det_scale)),
            interpolation=cv2.INTER_AREA,
        )
        inner_ok, inner_filled_det, _, _ = _segment_bounded_item(
            warp_det,
            cfg.min_item_area_ratio,
            border_inset_px=max(4, round(warp_det.shape[0] * 0.015)),
        )
        if inner_ok:
            up_mask = cv2.resize(
                inner_filled_det,
                (target_size, target_size),
                interpolation=cv2.INTER_LINEAR,
            )
            m_item = np.where(cv2.GaussianBlur(up_mask, (5, 5), 0) > 127, 255, 0).astype(
                np.uint8
            )
            geom, minor_px = _compute_ellipse_geometry(
                m_item,
                cfg.reference_diameter_mm,
                target_size,
            )
            m_rim, m_interior = _build_item_rim_and_interior(
                m_item,
                minor_px,
                cfg.rim_band_ratio,
            )
        else:
            m_item = np.full((target_size, target_size), 255, dtype=np.uint8)
            ins = max(16, round(target_size * 0.05))
            m_interior = np.zeros((target_size, target_size), dtype=np.uint8)
            m_interior[ins : target_size - ins, ins : target_size - ins] = 255
            m_rim = np.zeros((target_size, target_size), dtype=np.uint8)
            geom = EllipseGeometry(
                center=(target_size / 2.0, target_size / 2.0),
                axes=(float(target_size), float(target_size)),
                angle_deg=0.0,
                contour_deviation_pct=0.0,
                axis_ratio_error_pct=0.0,
                px_per_mm=float(target_size) / cfg.reference_diameter_mm,
                scale_calibrated=False,
            )
    else:
        is_item, filled_det, (bx, by, bw, bh), _ = _segment_bounded_item(
            det_bgr,
            cfg.min_item_area_ratio,
        )
        if is_item:
            fx = bx / det_scale
            fy = by / det_scale
            fw = bw / det_scale
            fh = bh / det_scale
            side = max(32, round(max(fw, fh) * 1.16))
            side = max(side, round(max(fw, fh)) + 24)
            cx_full = fx + fw / 2.0
            cy_full = fy + fh / 2.0
            crop_x0 = round(cx_full - side / 2.0)
            crop_y0 = round(cy_full - side / 2.0)

            border_sample = np.concatenate(
                [
                    original_bgr[:4, :, :].reshape(-1, 3),
                    original_bgr[-4:, :, :].reshape(-1, 3),
                    original_bgr[:, :4, :].reshape(-1, 3),
                    original_bgr[:, -4:, :].reshape(-1, 3),
                ],
                axis=0,
            )
            pad_vals = np.median(border_sample, axis=0).astype(np.uint8)
            pad_bgr = (int(pad_vals[0]), int(pad_vals[1]), int(pad_vals[2]))

            cropped = _crop_square_with_pad(original_bgr, crop_x0, crop_y0, side, pad_bgr)
            interp = cv2.INTER_AREA if side > target_size else cv2.INTER_LINEAR
            roi_bgr = cv2.resize(cropped, (target_size, target_size), interpolation=interp)

            det_side = max(8, round(side * det_scale))
            det_x0 = round(crop_x0 * det_scale)
            det_y0 = round(crop_y0 * det_scale)
            cropped_mask = _crop_square_mask_with_pad(filled_det, det_x0, det_y0, det_side)
            up_mask = cv2.resize(
                cropped_mask,
                (target_size, target_size),
                interpolation=cv2.INTER_LINEAR,
            )
            m_item = np.where(cv2.GaussianBlur(up_mask, (5, 5), 0) > 127, 255, 0).astype(
                np.uint8
            )
            geom, minor_px = _compute_ellipse_geometry(
                m_item,
                cfg.reference_diameter_mm,
                target_size,
            )
            m_rim, m_interior = _build_item_rim_and_interior(
                m_item,
                minor_px,
                cfg.rim_band_ratio,
            )
        else:
            roi_bgr, m_item, m_rim, m_interior, geom = _macro_fallback(
                original_bgr,
                target_size,
            )

    roi_lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    l_chan, _, b_chan = cv2.split(roi_lab)
    clip_limit = cfg.clahe_clip_bisque if stage == "bisque" else cfg.clahe_clip_glazed
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=cfg.clahe_grid)
    roi_lab_l_clahe = clahe.apply(l_chan)

    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, _ = cv2.split(roi_hsv)

    item_bool = m_item > 0
    if stage == "bisque":
        m_spec = np.zeros((target_size, target_size), dtype=np.uint8)
    else:
        m_spec = np.where(
            (l_chan > cfg.glare_l_thresh) & (s_chan < cfg.glare_s_thresh) & item_bool,
            255,
            0,
        ).astype(np.uint8)

    h_lo, h_hi = cfg.cobalt_h_range
    cobalt_bool = (
        ((s_chan > cfg.cobalt_s_min) & (h_chan >= h_lo) & (h_chan <= h_hi))
        | (b_chan < cfg.cobalt_b_max)
    ) & item_bool
    m_cobalt = np.where(cobalt_bool, 255, 0).astype(np.uint8)

    item_area = max(1, int(np.count_nonzero(item_bool)))
    glare_ratio = float(np.count_nonzero(m_spec)) / float(item_area)
    cobalt_ratio = float(np.count_nonzero(m_cobalt)) / float(item_area)
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return PreprocessedFrame(
        stage=stage,
        original_bgr=original_bgr,
        roi_bgr=roi_bgr,
        roi_lab_l_clahe=roi_lab_l_clahe,
        roi_hsv=roi_hsv,
        m_item=m_item,
        m_rim=m_rim,
        m_interior=m_interior,
        m_spec=m_spec,
        m_cobalt=m_cobalt,
        geometry=geom,
        perspective_applied=perspective_applied,
        glare_ratio=glare_ratio,
        cobalt_ratio=cobalt_ratio,
        latency_ms=latency_ms,
    )
