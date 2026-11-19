#!/usr/bin/env python3
"""Generate the CeramScan synthetic validation stand v0.1 (>= 24 labeled ceramic plates)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

STAND_LABEL = "синтетический валидационный стенд v0.1"


def _render_gzhel_cobalt(
    layer: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    rng: np.random.Generator,
    motif: str = "standard",
) -> None:
    """Paint feathered hand-brushed Gzhel cobalt or counterfeit motif onto plate layer."""
    if motif == "green_pigment":
        dark_color = (58, 138, 66)
        mid_color = (82, 168, 92)
    else:
        dark_color = (176, 54, 20)
        mid_color = (206, 92, 46)

    if motif == "diamond_grid":
        step = max(34, radius // 5)
        for offset in range(-radius, radius + 1, step):
            cv2.line(
                layer,
                (cx + offset, cy - radius),
                (cx + offset + 35, cy + radius),
                dark_color,
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                layer,
                (cx - radius, cy + offset),
                (cx + radius, cy + offset - 35),
                mid_color,
                2,
                cv2.LINE_AA,
            )
        cv2.rectangle(
            layer,
            (cx - int(radius * 0.42), cy - int(radius * 0.42)),
            (cx + int(radius * 0.42), cy + int(radius * 0.42)),
            dark_color,
            4,
        )
        return

    if motif == "halftone":
        dot_step = 20
        for dy in range(-int(radius * 0.7), int(radius * 0.7) + 1, dot_step):
            for dx in range(-int(radius * 0.7), int(radius * 0.7) + 1, dot_step):
                if dx * dx + dy * dy <= (radius * 0.68) ** 2:
                    cv2.circle(layer, (cx + dx, cy + dy), 5, mid_color, -1, cv2.LINE_AA)
        blurred = cv2.GaussianBlur(layer, (9, 9), 3.0)
        layer[:] = blurred
        return

    if motif == "alt_floral":
        # Stamped 4-branch angular motif without the hand-painted center ring
        for idx in range(4):
            ang = idx * (math.pi / 2.0) + math.pi / 4.0
            rx = int(cx + radius * 0.48 * math.cos(ang))
            ry = int(cy + radius * 0.48 * math.sin(ang))
            pts_tri = np.array(
                [
                    [cx, cy],
                    [int(rx - 22 * math.sin(ang)), int(ry + 22 * math.cos(ang))],
                    [int(cx + radius * 0.72 * math.cos(ang)), int(cy + radius * 0.72 * math.sin(ang))],
                    [int(rx + 22 * math.sin(ang)), int(ry - 22 * math.cos(ang))],
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(layer, pts_tri, dark_color, cv2.LINE_AA)
        cv2.rectangle(
            layer,
            (cx - int(radius * 0.16), cy - int(radius * 0.16)),
            (cx + int(radius * 0.16), cy + int(radius * 0.16)),
            mid_color,
            3,
        )
        return

    petal_count = 8
    cv2.ellipse(
        layer,
        (cx, cy),
        (int(radius * 0.73), int(radius * 0.73)),
        0,
        0,
        360,
        mid_color,
        3,
        cv2.LINE_AA,
    )
    phase = float(rng.uniform(-0.12, 0.12)) if motif != "alt_floral" else 0.45
    for idx in range(petal_count):
        theta = idx * (2.0 * math.pi / float(petal_count)) + phase
        px = int(cx + radius * 0.41 * math.cos(theta))
        py = int(cy + radius * 0.41 * math.sin(theta))
        deg = int(round(math.degrees(theta)))
        cv2.ellipse(layer, (px, py), (34, 15), deg, 0, 360, dark_color, -1, cv2.LINE_AA)

        vx = int(cx + radius * 0.49 * math.cos(theta))
        vy = int(cy + radius * 0.49 * math.sin(theta))
        cv2.line(layer, (cx, cy), (vx, vy), (218, 222, 226), 1, cv2.LINE_AA)

        dx = int(cx + radius * 0.63 * math.cos(theta + 0.16))
        dy = int(cy + radius * 0.63 * math.sin(theta + 0.16))
        cv2.circle(layer, (dx, dy), 6, mid_color, -1, cv2.LINE_AA)

        lx = int(cx + radius * 0.24 * math.cos(theta + 0.25))
        ly = int(cy + radius * 0.24 * math.sin(theta + 0.25))
        cv2.ellipse(layer, (lx, ly), (16, 6), deg + 35, 0, 360, dark_color, -1, cv2.LINE_AA)

    cv2.circle(layer, (cx, cy), int(radius * 0.13), dark_color, 4, cv2.LINE_AA)
    for k in range(8):
        ang = k * (math.pi / 4.0)
        sx = int(cx + radius * 0.08 * math.cos(ang))
        sy = int(cy + radius * 0.08 * math.sin(ang))
        cv2.circle(layer, (sx, sy), 3, mid_color, -1, cv2.LINE_AA)


def _render_plate(
    stage: str,
    defect_class: str,
    seed: int,
    cobalt: bool = False,
    motif: str = "standard",
    glare: bool = False,
) -> np.ndarray:
    """Render a deterministic 640x640 synthetic ceramic plate with natural shading."""
    rng = np.random.default_rng(seed)
    size = 640
    img = np.full((size, size, 3), (30, 32, 36), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = 240

    ax_x = radius
    ax_y = int(radius * 0.85) if defect_class == "shape_warp" else radius

    plate_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(plate_mask, (cx, cy), (ax_x, ax_y), 0, 0, 360, 255, -1)

    if defect_class == "chip":
        chip_angle = float(rng.uniform(-0.6, 0.6))
        notch_cx = int(cx + (ax_x - 4) * math.cos(chip_angle))
        notch_cy = int(cy + (ax_y - 4) * math.sin(chip_angle))
        cv2.circle(plate_mask, (notch_cx, notch_cy), 16, 0, -1)

    base_bgr = np.array([178.0, 180.0, 184.0] if stage == "bisque" else [218.0, 222.0, 226.0])
    yy, xx = np.mgrid[0:size, 0:size]
    radial = np.hypot(xx - cx, yy - cy) / float(radius)
    shade = (1.0 - 0.05 * np.clip(radial, 0.0, 1.2))[:, :, None]
    noise = rng.integers(-2, 3, size=(size, size, 3))
    plate_layer = np.clip(base_bgr * shade + noise, 0, 255).astype(np.uint8)

    rim_tone = (168, 170, 174) if stage == "bisque" else (206, 210, 215)
    cv2.ellipse(
        plate_layer,
        (cx, cy),
        (int(ax_x * 0.95), int(ax_y * 0.95)),
        0,
        0,
        360,
        rim_tone,
        2,
        cv2.LINE_AA,
    )

    if cobalt:
        _render_gzhel_cobalt(plate_layer, cx, cy, radius, rng, motif=motif)

    if defect_class == "crack":
        pts = np.array(
            [
                [cx - 95, cy - 28],
                [cx - 42, cy - 8],
                [cx + 12, cy + 16],
                [cx + 88, cy + 4],
            ],
            dtype=np.int32,
        )
        jitter = rng.integers(-6, 7, size=pts.shape)
        pts = pts + jitter
        cv2.polylines(plate_layer, [pts], False, (18, 18, 20), 3, cv2.LINE_AA)

    if defect_class == "glaze_bare":
        bx = cx - int(rng.integers(55, 85))
        by = cy + int(rng.integers(40, 70))
        cv2.ellipse(plate_layer, (bx, by), (15, 11), 18, 0, 360, (142, 136, 128), -1, cv2.LINE_AA)

    if defect_class == "pinhole_speck":
        offsets = [(-65, -48), (18, -62), (-22, 54), (62, 36)]
        for dx, dy in offsets:
            cv2.circle(plate_layer, (cx + dx, cy + dy), 4, (22, 22, 24), -1, cv2.LINE_AA)

    if defect_class == "glaze_drip":
        drip_pts = np.array(
            [
                [cx + 35, cy - 42],
                [cx + 39, cy - 18],
                [cx + 37, cy + 4],
            ],
            dtype=np.int32,
        )
        cv2.polylines(plate_layer, [drip_pts], False, (172, 184, 202), 8, cv2.LINE_AA)

    if glare and stage == "glazed":
        glare_mask = np.zeros((size, size), dtype=np.float32)
        cv2.ellipse(
            glare_mask,
            (cx - 48, cy - 62),
            (64, 34),
            -24,
            0,
            360,
            1.0,
            -1,
        )
        glare_mask = cv2.GaussianBlur(glare_mask, (15, 15), 4.0)
        white = np.full_like(plate_layer, 254.0, dtype=np.float32)
        alpha = glare_mask[:, :, None]
        plate_layer = np.clip(
            plate_layer.astype(np.float32) * (1.0 - alpha) + white * alpha,
            0,
            255,
        ).astype(np.uint8)

    img[plate_mask > 0] = plate_layer[plate_mask > 0]
    return img


def _build_sample_specs(count: int) -> list[dict[str, Any]]:
    base_specs: list[dict[str, Any]] = [
        {"id": "SYN-BISQUE-PASS-01", "stage": "bisque", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-BISQUE-PASS-02", "stage": "bisque", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-BISQUE-PASS-03", "stage": "bisque", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-BISQUE-CRACK-01", "stage": "bisque", "defect_class": "crack", "expected_verdict": "REJECT", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-BISQUE-CRACK-02", "stage": "bisque", "defect_class": "crack", "expected_verdict": "REJECT", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-BISQUE-CRACK-03", "stage": "bisque", "defect_class": "crack", "expected_verdict": "REJECT", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-BISQUE-CHIP-01", "stage": "bisque", "defect_class": "chip", "expected_verdict": "REJECT", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-BISQUE-CHIP-02", "stage": "bisque", "defect_class": "chip", "expected_verdict": "REJECT", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-BISQUE-WARP-01", "stage": "bisque", "defect_class": "shape_warp", "expected_verdict": "REJECT", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-BISQUE-WARP-02", "stage": "bisque", "defect_class": "shape_warp", "expected_verdict": "REJECT", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-PASS-01", "stage": "glazed", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": True, "motif": "standard", "glare": True, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-PASS-02", "stage": "glazed", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": True, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-PASS-03", "stage": "glazed", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": True, "motif": "standard", "glare": True, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-PASS-04", "stage": "glazed", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": True, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-CRACK-01", "stage": "glazed", "defect_class": "crack", "expected_verdict": "REJECT", "cobalt": True, "motif": "standard", "glare": True, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-CRACK-02", "stage": "glazed", "defect_class": "crack", "expected_verdict": "REJECT", "cobalt": False, "motif": "standard", "glare": True, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-CRACK-03", "stage": "glazed", "defect_class": "crack", "expected_verdict": "REJECT", "cobalt": True, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-BARE-01", "stage": "glazed", "defect_class": "glaze_bare", "expected_verdict": "REWORK", "cobalt": True, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-BARE-02", "stage": "glazed", "defect_class": "glaze_bare", "expected_verdict": "REWORK", "cobalt": False, "motif": "standard", "glare": True, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-PINHOLE-01", "stage": "glazed", "defect_class": "pinhole_speck", "expected_verdict": "REJECT", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-GLAZED-DRIP-01", "stage": "glazed", "defect_class": "glaze_drip", "expected_verdict": "REWORK", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
        {"id": "SYN-FAKE-ALT-01", "stage": "glazed", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": True, "motif": "alt_floral", "glare": False, "auth_pair_id": "SYN-GLAZED-PASS-01", "is_counterfeit": True},
        {"id": "SYN-FAKE-HALFTONE-02", "stage": "glazed", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": True, "motif": "halftone", "glare": False, "auth_pair_id": "SYN-GLAZED-PASS-02", "is_counterfeit": True},
        {"id": "SYN-FAKE-GRID-03", "stage": "glazed", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": True, "motif": "diamond_grid", "glare": False, "auth_pair_id": "SYN-GLAZED-PASS-03", "is_counterfeit": True},
        {"id": "SYN-FAKE-GREEN-04", "stage": "glazed", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": True, "motif": "green_pigment", "glare": False, "auth_pair_id": "SYN-GLAZED-PASS-04", "is_counterfeit": True},
        {"id": "SYN-BISQUE-PASS-04", "stage": "bisque", "defect_class": "pass", "expected_verdict": "PASS", "cobalt": False, "motif": "standard", "glare": False, "auth_pair_id": None, "is_counterfeit": False},
    ]

    if count == 24:
        # Preserve all required classes & 4 counterfeits within exact 24 samples
        return [s for s in base_specs if s["id"] not in {"SYN-BISQUE-PASS-03", "SYN-BISQUE-PASS-04"}]
    if count == 25:
        return [s for s in base_specs if s["id"] != "SYN-BISQUE-PASS-04"]
    if count == 26:
        return base_specs

    specs = list(base_specs)
    extra_idx = 1
    while len(specs) < count:
        template = dict(base_specs[(extra_idx - 1) % 21])
        template["id"] = f"SYN-EXTRA-{extra_idx:02d}"
        specs.append(template)
        extra_idx += 1
    return specs


def generate_stand(output_dir: Path, count: int = 26, seed: int = 42) -> Path:
    """Generate synthetic ceramic dataset images and manifest.json."""
    if count < 24:
        raise ValueError(f"--count must be >= 24, got {count}.")

    out_dir = Path(output_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for stale_png in img_dir.glob("*.png"):
        stale_png.unlink()

    specs = _build_sample_specs(count)
    manifest_samples: list[dict[str, Any]] = []

    for idx, spec in enumerate(specs):
        sample_seed = seed * 100 + idx + 1
        bgr = _render_plate(
            stage=spec["stage"],
            defect_class=spec["defect_class"],
            seed=sample_seed,
            cobalt=spec["cobalt"],
            motif=spec["motif"],
            glare=spec["glare"],
        )
        rel_path = Path("images") / f"{spec['id']}.png"
        abs_path = out_dir / rel_path
        ok = cv2.imwrite(str(abs_path), bgr)
        if not ok:
            raise RuntimeError(f"Failed to write synthetic image: {abs_path}")

        manifest_samples.append(
            {
                "id": spec["id"],
                "image_path": rel_path.as_posix(),
                "stage": spec["stage"],
                "defect_class": spec["defect_class"],
                "expected_verdict": spec["expected_verdict"],
                "auth_pair_id": spec["auth_pair_id"],
                "is_counterfeit": spec["is_counterfeit"],
            }
        )

    payload = {
        "stand_label": STAND_LABEL,
        "seed": seed,
        "count": len(manifest_samples),
        "samples": manifest_samples,
    }
    manifest_path = out_dir / "manifest.json"
    tmp_path = out_dir / ".manifest.json.tmp"
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, manifest_path)
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CeramScan synthetic validation stand.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--count", type=int, default=26)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.count < 24:
        print(f"Error: --count must be >= 24 (got {args.count}).", file=sys.stderr)
        return 2

    manifest_path = generate_stand(args.output_dir, count=args.count, seed=args.seed)
    print(f"Generated {args.count} synthetic ceramic samples at {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
