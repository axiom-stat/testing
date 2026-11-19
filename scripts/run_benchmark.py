#!/usr/bin/env python3
"""Run CPU latency, GOST defect, and 289-FZ authenticity benchmarks for CeramScan."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cv_core.defect_detector import DefectDetector, InspectionResult  # noqa: E402
from src.cv_core.fingerprint import (  # noqa: E402
    VERDICT_AUTHENTIC,
    VERDICT_COUNTERFEIT,
    extract_fingerprint,
    verify_fingerprint,
)
from src.cv_core.preprocessing import get_cyrillic_font, preprocess_image  # noqa: E402


def _resolve_sample_image(dataset_dir: Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    if (dataset_dir / p).is_file():
        return dataset_dir / p
    return Path.cwd() / p


def _load_bgr(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Sample image not found: {path}")
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise ValueError(f"Failed to decode image file: {path}")
    return img


def _render_side_by_side_overlay(
    src_bgr: np.ndarray,
    result: InspectionResult,
    sample_id: str,
    stage: str,
    out_path: Path,
) -> None:
    """Save side-by-side 'Исходник | Разметка CeramScan' PNG."""
    oh, ow = result.overlay_bgr.shape[:2]
    left = cv2.resize(src_bgr, (ow, oh), interpolation=cv2.INTER_AREA)
    divider = np.full((oh, 4, 3), (203, 213, 225), dtype=np.uint8)
    combined = np.hstack([left, divider, result.overlay_bgr])

    header_h = 44
    canvas = np.full((oh + header_h, combined.shape[1], 3), (248, 250, 252), dtype=np.uint8)
    canvas[header_h:, :] = combined

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = get_cyrillic_font(size=18, bold=True)
    stage_ru = "Утиль 900C" if stage == "bisque" else "Глазурь 1350C"
    title = f"Исходник | Разметка CeramScan ({sample_id}, {stage_ru}) [{result.verdict}]"
    verdict_rgb = {
        "PASS": (6, 95, 70),
        "REWORK": (146, 64, 14),
        "REJECT": (153, 27, 27),
    }[result.verdict]
    draw.line([(0, header_h - 1), (canvas.shape[1], header_h - 1)], fill=(203, 213, 225), width=1)
    draw.text((16, 11), title, font=font, fill=verdict_rgb)

    final_bgr = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", final_bgr)
    if not ok:
        raise RuntimeError(f"Failed to encode side-by-side overlay PNG: {out_path}")
    out_path.write_bytes(bytes(encoded))


def _perturb_genuine_probe(ref_bgr: np.ndarray) -> np.ndarray:
    """Apply deterministic mild smartphone rotation and illumination shift."""
    h, w = ref_bgr.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 4.0, 1.0)
    rotated = cv2.warpAffine(ref_bgr, rot, (w, h), borderValue=(30, 32, 36))
    return np.clip(rotated.astype(np.int16) + 8, 0, 255).astype(np.uint8)


def _atomic_write_text(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, target)


def run_benchmark(
    dataset_path: Path,
    output_json: Path,
    output_md: Path,
    overlays_dir: Path,
    weights_path: Path | None = None,
) -> dict[str, Any]:
    """Execute end-to-end defect and authenticity benchmarks and write reports."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {dataset_path}")

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(f"Dataset manifest has zero samples: {dataset_path}")

    detector = DefectDetector(weights_path=weights_path)
    dataset_dir = dataset_path.parent
    overlays_dir.mkdir(parents=True, exist_ok=True)

    latencies: list[float] = []
    verdict_hits = 0
    y_true: list[str] = []
    y_pred: list[str] = []
    overlay_files: list[str] = []
    written_names: set[str] = set()
    loaded_images: dict[str, np.ndarray] = {}
    sample_by_id: dict[str, dict[str, Any]] = {}

    for sample in samples:
        sample_id = str(sample["id"])
        sample_by_id[sample_id] = sample
        stage = str(sample["stage"])
        img_path = _resolve_sample_image(dataset_dir, str(sample["image_path"]))
        img_bgr = _load_bgr(img_path)
        loaded_images[sample_id] = img_bgr

        t0 = time.perf_counter()
        result = detector.inspect(img_bgr, stage=stage)  # type: ignore[arg-type]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        if result.verdict == sample["expected_verdict"]:
            verdict_hits += 1

        true_cls = str(sample["defect_class"])
        detected_classes = [d.defect_class.value for d in result.defects]
        if not detected_classes:
            pred_cls = "pass"
        elif true_cls != "pass" and true_cls in detected_classes:
            pred_cls = true_cls
        else:
            pred_cls = detected_classes[0]
        y_true.append(true_cls)
        y_pred.append(pred_cls)

        ov_filename = f"{sample_id}.png"
        ov_path = overlays_dir / ov_filename
        _render_side_by_side_overlay(img_bgr, result, sample_id, stage, ov_path)
        written_names.add(ov_filename)
        overlay_files.append(ov_path.as_posix())

    for stale_png in overlays_dir.glob("*.png"):
        if stale_png.is_file() and stale_png.name not in written_names:
            stale_png.unlink()

    all_classes = sorted(set(y_true) | set(y_pred))
    per_class: dict[str, dict[str, Any]] = {}
    macro_p: list[float] = []
    macro_r: list[float] = []
    macro_f1: list[float] = []

    for cls_name in all_classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == cls_name and yp == cls_name)
        fp = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt != cls_name and yp == cls_name)
        fn = sum(1 for yt, yp in zip(y_true, y_pred, strict=True) if yt == cls_name and yp != cls_name)
        support = tp + fn
        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / float(support) if support > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0.0 else 0.0
        per_class[cls_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support > 0:
            macro_p.append(prec)
            macro_r.append(rec)
            macro_f1.append(f1)

    auth_pairs = [
        s
        for s in samples
        if s.get("is_counterfeit") and s.get("auth_pair_id") and s["auth_pair_id"] in sample_by_id
    ]
    if auth_pairs:
        sig_cache: dict[str, Any] = {}
        genuine_scores: list[float] = []
        fake_scores: list[float] = []
        auth_hits = 0

        for fake_sample in auth_pairs:
            ref_id = str(fake_sample["auth_pair_id"])
            if ref_id not in sig_cache:
                ref_stage = str(sample_by_id[ref_id]["stage"])
                ref_frame = preprocess_image(loaded_images[ref_id], stage=ref_stage)  # type: ignore[arg-type]
                sig_cache[ref_id] = extract_fingerprint(ref_frame)
            ref_sig = sig_cache[ref_id]

            gen_probe_bgr = _perturb_genuine_probe(loaded_images[ref_id])
            gen_probe_frame = preprocess_image(gen_probe_bgr, stage="glazed")
            gen_res = verify_fingerprint(ref_sig, gen_probe_frame)
            genuine_scores.append(gen_res.match_percent)
            if gen_res.verdict == VERDICT_AUTHENTIC:
                auth_hits += 1

            fake_id = str(fake_sample["id"])
            fake_frame = preprocess_image(loaded_images[fake_id], stage="glazed")
            fake_res = verify_fingerprint(ref_sig, fake_frame)
            fake_scores.append(fake_res.match_percent)
            if fake_res.verdict == VERDICT_COUNTERFEIT:
                auth_hits += 1

        auth_benchmark = {
            "pair_count": len(auth_pairs),
            "auth_accuracy": round(float(auth_hits) / float(2 * len(auth_pairs)), 4),
            "genuine_mean_match": round(float(np.mean(genuine_scores)), 2),
            "fake_mean_match": round(float(np.mean(fake_scores)), 2),
        }
    else:
        auth_benchmark = {
            "pair_count": 0,
            "auth_accuracy": 0.0,
            "genuine_mean_match": 0.0,
            "fake_mean_match": 0.0,
        }

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_label": str(payload.get("stand_label", dataset_path.stem)),
        "model_mode": "onnx" if weights_path is not None else "deterministic",
        "sample_count": len(samples),
        "avg_latency_ms": round(float(np.mean(latencies)), 2),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        "precision": round(float(np.mean(macro_p)) if macro_p else 0.0, 4),
        "recall": round(float(np.mean(macro_r)) if macro_r else 0.0, 4),
        "f1": round(float(np.mean(macro_f1)) if macro_f1 else 0.0, 4),
        "verdict_accuracy": round(float(verdict_hits) / float(len(samples)), 4),
        "per_class": per_class,
        "auth_benchmark": auth_benchmark,
        "overlay_files": overlay_files,
    }

    _atomic_write_text(output_json, json.dumps(report, ensure_ascii=False, indent=2))

    md_lines = [
        f"# CeramScan Benchmark Summary ({report['dataset_label']})",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Model mode: `{report['model_mode']}`",
        f"- Sample count: {report['sample_count']}",
        f"- Average latency: {report['avg_latency_ms']:.2f} ms/frame",
        f"- p95 latency: {report['p95_latency_ms']:.2f} ms/frame",
        f"- Macro Precision: {report['precision']:.4f}",
        f"- Macro Recall: {report['recall']:.4f}",
        f"- Macro F1: {report['f1']:.4f}",
        f"- GOST Verdict Accuracy: {report['verdict_accuracy']:.4f}",
        "",
        "## Per-Class Defect Metrics",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
    ]
    for cls_name, metrics in per_class.items():
        md_lines.append(
            f"| `{cls_name}` | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['support']} |"
        )
    md_lines.extend(
        [
            "",
            "## 289-FZ Authenticity Benchmark",
            "",
            f"- Pair count: {auth_benchmark['pair_count']}",
            f"- Auth accuracy: {auth_benchmark['auth_accuracy']:.4f}",
            f"- Genuine mean match: {auth_benchmark['genuine_mean_match']:.2f}%",
            f"- Counterfeit mean match: {auth_benchmark['fake_mean_match']:.2f}%",
            "",
        ]
    )
    _atomic_write_text(output_md, "\n".join(md_lines))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark CeramScan CV core and 289-FZ matcher.")
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic/manifest.json"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("reports/benchmark_results.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/benchmark_summary.md"))
    parser.add_argument("--overlays-dir", type=Path, default=Path("reports/overlays"))
    args = parser.parse_args(argv)

    try:
        report = run_benchmark(
            dataset_path=args.dataset,
            output_json=args.output_json,
            output_md=args.output_md,
            overlays_dir=args.overlays_dir,
            weights_path=args.weights,
        )
    except (FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Benchmark complete ({report['model_mode']}): "
        f"n={report['sample_count']}, avg={report['avg_latency_ms']:.2f}ms, "
        f"p95={report['p95_latency_ms']:.2f}ms, F1={report['f1']:.4f}, "
        f"verdict_acc={report['verdict_accuracy']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
