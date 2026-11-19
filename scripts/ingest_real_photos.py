#!/usr/bin/env python3
"""Ingest real ceramic factory photos, optionally export YOLO labels, and rebuild reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_synthetic_ceramic import generate_stand  # noqa: E402
from scripts.run_benchmark import run_benchmark  # noqa: E402
from src.cv_core.defect_detector import DefectDetector  # noqa: E402
from src.cv_core.preprocessing import DefectClass, preprocess_image  # noqa: E402

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})
CATEGORY_MAP: dict[str, tuple[str, str, bool]] = {
    "pass": ("pass", "PASS", False),
    "chip": ("chip", "REJECT", False),
    "crack": ("crack", "REJECT", False),
    "glaze": ("glaze_bare", "REWORK", False),
    "fake": ("pass", "PASS", True),
}
YOLO_CLASS_IDS: dict[str, int] = {cls.value: idx for idx, cls in enumerate(DefectClass)}


def _infer_stage(category: str, img_path: Path) -> str:
    if category == "glaze":
        return "glazed"
    stem_low = img_path.stem.lower()
    if "bisque" in stem_low or "util" in stem_low:
        return "bisque"
    return "glazed"


def _collect_real_samples(input_dir: Path) -> list[dict[str, Any]]:
    input_resolved = input_dir.resolve()
    if not input_resolved.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    collected_by_cat: dict[str, list[Path]] = {cat: [] for cat in CATEGORY_MAP}
    for cat in CATEGORY_MAP:
        cat_dir = input_resolved / cat
        if not cat_dir.is_dir():
            continue
        for child in sorted(cat_dir.iterdir()):
            if not child.is_file():
                continue
            resolved_child = child.resolve()
            if not resolved_child.is_relative_to(input_resolved):
                raise ValueError(f"Path traversal outside input dir rejected: {child}")
            if resolved_child.suffix.lower() not in IMAGE_EXTS:
                continue
            decoded = cv2.imread(str(resolved_child), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                raise ValueError(f"Failed to decode real photo: {resolved_child}")
            collected_by_cat[cat].append(resolved_child)

    first_any_pass_id: str | None = None
    first_glazed_pass_id: str | None = None
    samples: list[dict[str, Any]] = []
    idx_counter = 1

    for cat in ("pass", "chip", "crack", "glaze", "fake"):
        defect_cls, expected_verdict, is_fake = CATEGORY_MAP[cat]
        for img_path in collected_by_cat[cat]:
            sample_id = f"REAL-{cat.upper()}-{idx_counter:03d}"
            idx_counter += 1
            stage = _infer_stage(cat, img_path)
            if cat == "pass":
                if first_any_pass_id is None:
                    first_any_pass_id = sample_id
                if stage == "glazed" and first_glazed_pass_id is None:
                    first_glazed_pass_id = sample_id
            auth_target_id = first_glazed_pass_id or first_any_pass_id
            samples.append(
                {
                    "id": sample_id,
                    "image_path": str(img_path),
                    "stage": stage,
                    "defect_class": defect_cls,
                    "expected_verdict": expected_verdict,
                    "auth_pair_id": auth_target_id if is_fake else None,
                    "is_counterfeit": is_fake,
                }
            )
    return samples


def _export_yolo_dataset(
    manifest_path: Path,
    export_dir: Path,
    detector: DefectDetector,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    manifest_dir = manifest_path.parent

    img_out_dir = export_dir / "images"
    lbl_out_dir = export_dir / "labels"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        sample_id = str(sample["id"])
        raw_p = Path(str(sample["image_path"]))
        src_p = raw_p if raw_p.is_absolute() else (manifest_dir / raw_p if (manifest_dir / raw_p).is_file() else Path.cwd() / raw_p)
        img_bgr = cv2.imread(str(src_p), cv2.IMREAD_COLOR)
        if img_bgr is None or img_bgr.size == 0:
            raise ValueError(f"Failed to read image for YOLO export: {src_p}")

        stage = str(sample["stage"])
        frame = preprocess_image(img_bgr, stage=stage)  # type: ignore[arg-type]
        result = detector.inspect(img_bgr, stage=stage)  # type: ignore[arg-type]
        h, w = frame.roi_bgr.shape[:2]

        out_img_path = img_out_dir / f"{sample_id}.png"
        ok, encoded = cv2.imencode(".png", frame.roi_bgr)
        if not ok:
            raise RuntimeError(f"Failed to encode YOLO image: {out_img_path}")
        out_img_path.write_bytes(bytes(encoded))

        lines: list[str] = []
        if str(sample["defect_class"]) != "pass" and not sample.get("is_counterfeit"):
            for d in result.defects:
                cls_id = YOLO_CLASS_IDS.get(d.defect_class.value, 0)
                bx, by, bw, bh = d.bbox_xywh
                xc = max(0.0, min(1.0, (bx + bw / 2.0) / float(w)))
                yc = max(0.0, min(1.0, (by + bh / 2.0) / float(h)))
                nw = max(0.001, min(1.0, bw / float(w)))
                nh = max(0.001, min(1.0, bh / float(h)))
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        (lbl_out_dir / f"{sample_id}.txt").write_text("\n".join(lines), encoding="utf-8")

    names_yaml = "\n".join(f"  {idx}: {cls.value}" for idx, cls in enumerate(DefectClass))
    data_yaml = "\n".join(
        [
            f"path: {export_dir.resolve()}",
            "train: images",
            "val: images",
            f"nc: {len(DefectClass)}",
            "names:",
            names_yaml,
            "",
        ]
    )
    (export_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest factory ceramic photos and rebuild reports.")
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data" / "real_photos")
    parser.add_argument("--export-yolo", type=Path, default=None)
    parser.add_argument("--rebuild-all", action="store_true")
    parser.add_argument("--weights", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        detector = DefectDetector(weights_path=args.weights)
        real_samples = _collect_real_samples(args.input)

        if real_samples:
            manifest_path = args.input.resolve() / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "stand_label": f"реальные фото производства ({len(real_samples)} шт.)",
                "count": len(real_samples),
                "samples": real_samples,
            }
            tmp_path = manifest_path.parent / ".manifest.json.tmp"
            tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp_path, manifest_path)
            print(f"Ingested {len(real_samples)} real photos into {manifest_path}")
        else:
            synthetic_dir = REPO_ROOT / "data" / "synthetic"
            manifest_path = synthetic_dir / "manifest.json"
            if not manifest_path.is_file():
                generate_stand(synthetic_dir, count=26, seed=42)
            print(
                f"Fallback: 0 real photos found in {args.input}; "
                f"using synthetic validation stand at {manifest_path}."
            )

        if args.export_yolo is not None:
            _export_yolo_dataset(manifest_path, args.export_yolo, detector)
            print(f"Exported YOLO dataset to {args.export_yolo}")

        if args.rebuild_all:
            report = run_benchmark(
                dataset_path=manifest_path,
                output_json=REPO_ROOT / "reports" / "benchmark_results.json",
                output_md=REPO_ROOT / "reports" / "benchmark_summary.md",
                overlays_dir=REPO_ROOT / "reports" / "overlays",
                weights_path=args.weights,
            )
            print(
                f"Rebuilt benchmark reports: n={report['sample_count']}, "
                f"avg={report['avg_latency_ms']:.2f}ms, F1={report['f1']:.4f}"
            )

            pptx_builder = REPO_ROOT / "grant_case" / "build_presentation.py"
            if pptx_builder.is_file():
                subprocess.run([sys.executable, str(pptx_builder)], check=True)
            else:
                print(
                    f"Warning: {pptx_builder} not found yet (expected in M4); skipping PPTX rebuild."
                )
    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        OSError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"Ingest error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
