"""289-FZ anti-counterfeit Gzhel cobalt fingerprint extraction and verification."""

from __future__ import annotations

import base64
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import AuthVerdictType, PreprocessedFrame

VERDICT_AUTHENTIC: AuthVerdictType = "Подлинник"
VERDICT_COUNTERFEIT: AuthVerdictType = "Подозрение на контрафакт или штамповку"
_ORB_DILATE_5X5 = np.ones((5, 5), dtype=np.uint8)


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
    min_cobalt_ratio: float = 0.015


@dataclass(slots=True)
class FingerprintSignature:
    phash_hex: str
    cobalt_hist: list[float]
    cobalt_ratio: float
    keypoints_xy: list[list[float]]
    descriptors_b64: str
    orb_count: int
    version: str = "1.0"


@dataclass(slots=True)
class VerificationResult:
    match_percent: float
    orb_score: float
    phash_score: float
    color_score: float
    matched_kp: int
    verdict: AuthVerdictType
    latency_ms: float


def _compute_phash_hex(roi_bgr: np.ndarray, m_item: np.ndarray) -> str:
    """Compute 64-bit DCT pHash inside ceramic mask with neutral background fill."""
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    masked_gray = np.where(m_item > 0, gray, 220).astype(np.uint8)
    small = cv2.resize(masked_gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)
    block = dct[:8, :8].ravel()
    med = float(np.median(block[1:]))
    bits = block > med
    val = 0
    for bit in bits:
        val = (val << 1) | int(bool(bit))
    return f"{val:016x}"


def _compute_cobalt_hist(
    roi_hsv: np.ndarray,
    m_cobalt: np.ndarray,
    m_interior: np.ndarray,
    cobalt_ratio: float,
    min_cobalt_ratio: float,
) -> list[float]:
    """Compute L1-normalized 18x16 HSV histogram for cobalt pigment or interior fallback."""
    mask = m_cobalt if cobalt_ratio >= min_cobalt_ratio else m_interior
    hist = cv2.calcHist([roi_hsv], [0, 1], mask, [18, 16], [85, 140, 35, 256]).astype(
        np.float32
    )
    total = float(hist.sum())
    if total > 0.0:
        hist /= total
    return [float(v) for v in hist.ravel()]


def extract_fingerprint(
    frame: PreprocessedFrame,
    config: FingerprintConfig | None = None,
) -> FingerprintSignature:
    """Extract ORB-500, 64-bit DCT pHash, and 288-bin cobalt HSV signature."""
    cfg = config if config is not None else FingerprintConfig()
    item_area = max(1, int(np.count_nonzero(frame.m_item)))
    cobalt_ratio = float(np.count_nonzero(frame.m_cobalt)) / float(item_area)

    phash_hex = _compute_phash_hex(frame.roi_bgr, frame.m_item)
    cobalt_hist = _compute_cobalt_hist(
        frame.roi_hsv,
        frame.m_cobalt,
        frame.m_interior,
        cobalt_ratio,
        cfg.min_cobalt_ratio,
    )

    gray = cv2.cvtColor(frame.roi_bgr, cv2.COLOR_BGR2GRAY)
    orb_mask = cv2.bitwise_and(
        cv2.dilate(frame.m_item, _ORB_DILATE_5X5),
        cv2.bitwise_not(frame.m_spec),
    )
    orb = cv2.ORB_create(nfeatures=cfg.orb_nfeatures)
    keypoints, descriptors = orb.detectAndCompute(gray, orb_mask)

    if not keypoints or descriptors is None or len(keypoints) == 0:
        return FingerprintSignature(
            phash_hex=phash_hex,
            cobalt_hist=cobalt_hist,
            cobalt_ratio=cobalt_ratio,
            keypoints_xy=[],
            descriptors_b64="",
            orb_count=0,
        )

    kp_xy = [[round(float(kp.pt[0]), 2), round(float(kp.pt[1]), 2)] for kp in keypoints]
    des_u8 = np.ascontiguousarray(descriptors, dtype=np.uint8)
    desc_b64 = base64.b64encode(des_u8.tobytes()).decode("ascii")
    return FingerprintSignature(
        phash_hex=phash_hex,
        cobalt_hist=cobalt_hist,
        cobalt_ratio=cobalt_ratio,
        keypoints_xy=kp_xy,
        descriptors_b64=desc_b64,
        orb_count=len(kp_xy),
    )


def _decode_descriptors(signature: FingerprintSignature) -> np.ndarray | None:
    if signature.orb_count < 2 or not signature.descriptors_b64:
        return None
    raw = base64.b64decode(signature.descriptors_b64.encode("ascii"))
    expected_len = signature.orb_count * 32
    if len(raw) != expected_len:
        raise ValueError(
            f"Corrupt ORB descriptor payload: expected {expected_len} bytes, got {len(raw)}."
        )
    return np.frombuffer(raw, dtype=np.uint8).reshape(signature.orb_count, 32)


def verify_fingerprint(
    reference: FingerprintSignature,
    probe_frame: PreprocessedFrame,
    config: FingerprintConfig | None = None,
) -> VerificationResult:
    """Compare probe frame against registered fingerprint signature under 289-FZ rules."""
    t0 = time.perf_counter()
    cfg = config if config is not None else FingerprintConfig()
    probe_sig = extract_fingerprint(probe_frame, cfg)

    hamming = (int(reference.phash_hex, 16) ^ int(probe_sig.phash_hex, 16)).bit_count()
    phash_score = float(np.clip(1.0 - (hamming / 64.0), 0.0, 1.0))

    if len(reference.cobalt_hist) != 288:
        raise ValueError(f"Expected 288-bin cobalt_hist, got {len(reference.cobalt_hist)}.")
    h1 = np.asarray(reference.cobalt_hist, dtype=np.float32).reshape(18, 16)
    if not np.all(np.isfinite(h1)):
        raise ValueError("Reference cobalt_hist contains non-finite values.")
    if not math.isfinite(reference.cobalt_ratio):
        raise ValueError("Reference cobalt_ratio must be finite.")
    h2 = np.asarray(probe_sig.cobalt_hist, dtype=np.float32).reshape(18, 16)
    bhatt = float(cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA))
    if not math.isfinite(bhatt):
        raise ValueError("Histogram comparison produced non-finite Bhattacharyya distance.")

    r1 = reference.cobalt_ratio
    r2 = probe_sig.cobalt_ratio
    r1_has_cobalt = r1 >= cfg.min_cobalt_ratio
    r2_has_cobalt = r2 >= cfg.min_cobalt_ratio
    if r1_has_cobalt and r2_has_cobalt:
        cov = min(r1, r2) / max(r1, r2, 1e-6)
        color_score = float(np.clip((1.0 - bhatt) * (0.4 + 0.6 * cov), 0.0, 1.0))
    elif not r1_has_cobalt and not r2_has_cobalt:
        color_score = float(np.clip((1.0 - bhatt) * 0.5, 0.0, 1.0))
    else:
        color_score = 0.0

    if len(reference.keypoints_xy) != reference.orb_count:
        raise ValueError(
            f"Mismatched keypoints_xy ({len(reference.keypoints_xy)}) and orb_count ({reference.orb_count})."
        )
    matched_kp = 0
    orb_score = 0.0
    ref_des = _decode_descriptors(reference)
    if ref_des is not None and len(reference.keypoints_xy) != len(ref_des):
        raise ValueError(
            f"Mismatched keypoints_xy ({len(reference.keypoints_xy)}) and ORB descriptors ({len(ref_des)})."
        )
    probe_des = _decode_descriptors(probe_sig)

    if ref_des is not None and probe_des is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(ref_des, probe_des, k=2)
        good = [
            pair[0]
            for pair in raw_matches
            if len(pair) == 2 and pair[0].distance < cfg.lowe_ratio * pair[1].distance
        ]
        if len(good) >= 8 and len(reference.keypoints_xy) >= len(ref_des):
            src_pts = np.float32([reference.keypoints_xy[m.queryIdx] for m in good]).reshape(
                -1, 1, 2
            )
            dst_pts = np.float32([probe_sig.keypoints_xy[m.trainIdx] for m in good]).reshape(
                -1, 1, 2
            )
            homography, mask = cv2.findHomography(
                src_pts,
                dst_pts,
                cv2.RANSAC,
                cfg.ransac_reproj_thresh,
            )
            if homography is not None and mask is not None:
                matched_kp = int(mask.ravel().sum())
                orb_score = min(1.0, matched_kp / 35.0)
            else:
                matched_kp = 0
                orb_score = 0.0
        else:
            matched_kp = len(good)
            orb_score = min(1.0, len(good) / 45.0)

    match_percent = round(
        100.0
        * (
            cfg.weight_orb * orb_score
            + cfg.weight_phash * phash_score
            + cfg.weight_color * color_score
        ),
        2,
    )
    cobalt_consistent = r1_has_cobalt == r2_has_cobalt
    is_authentic = (
        cobalt_consistent
        and match_percent >= cfg.authentic_threshold_pct
        and matched_kp >= cfg.min_matched_kp
    )
    verdict: AuthVerdictType = VERDICT_AUTHENTIC if is_authentic else VERDICT_COUNTERFEIT
    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return VerificationResult(
        match_percent=match_percent,
        orb_score=round(orb_score, 4),
        phash_score=round(phash_score, 4),
        color_score=round(color_score, 4),
        matched_kp=matched_kp,
        verdict=verdict,
        latency_ms=latency_ms,
    )
