"""CLI integration tests for synthetic stand generation, benchmarking, and real photo ingest."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def _run_script(cwd: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_generate_synthetic_and_run_benchmark_cli(tmp_path: Path) -> None:
    """Verify generate_synthetic_ceramic.py and run_benchmark.py produce valid outputs."""
    repo_root = Path(__file__).resolve().parents[1]
    syn_dir = tmp_path / "syn"
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    overlays_dir = tmp_path / "overlays"

    gen_proc = _run_script(
        repo_root,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--count", "26", "--seed", "42"],
    )
    assert gen_proc.returncode == 0, gen_proc.stderr
    manifest = json.loads((syn_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stand_label"] == "синтетический валидационный стенд v0.1"
    assert manifest["count"] == 26

    bench_proc = _run_script(
        repo_root,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(syn_dir / "manifest.json"),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--overlays-dir",
            str(overlays_dir),
        ],
    )
    assert bench_proc.returncode == 0, bench_proc.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["sample_count"] == 26
    assert 0.0 < report["avg_latency_ms"] < 300.0
    assert report["p95_latency_ms"] < 300.0
    assert report["f1"] >= 0.85
    assert report["auth_benchmark"]["pair_count"] == 4
    assert len(report["overlay_files"]) == 26
    assert out_md.is_file()


def test_ingest_real_photos_export_yolo_and_rebuild_all(tmp_path: Path) -> None:
    """Verify ingest_real_photos.py handles populated folders, YOLO export, and --rebuild-all."""
    repo_root = Path(__file__).resolve().parents[1]
    ws = tmp_path / "ws"
    shutil.copytree(repo_root / "src", ws / "src")
    shutil.copytree(repo_root / "scripts", ws / "scripts")

    photos_dir = ws / "data" / "real_photos"
    for cat in ("pass", "chip", "crack", "glaze", "fake"):
        (photos_dir / cat).mkdir(parents=True, exist_ok=True)

    # Create simple valid BGR plates in pass/ and crack/
    clean = np.full((640, 640, 3), (30, 32, 36), dtype=np.uint8)
    cv2.circle(clean, (320, 320), 240, (218, 222, 226), -1)
    cv2.circle(clean, (320, 320), 90, (176, 54, 20), 6)
    cv2.imwrite(str(photos_dir / "pass" / "bisque_sample.jpg"), clean)

    cracked = clean.copy()
    cv2.line(cracked, (220, 320), (420, 330), (15, 15, 18), 4)
    cv2.imwrite(str(photos_dir / "crack" / "crack_sample.jpg"), cracked)

    yolo_dir = tmp_path / "yolo_export"
    proc = _run_script(
        ws,
        [
            "scripts/ingest_real_photos.py",
            "--input",
            str(photos_dir),
            "--export-yolo",
            str(yolo_dir),
            "--rebuild-all",
        ],
    )
    assert proc.returncode == 0, proc.stderr
    assert (yolo_dir / "data.yaml").is_file()
    assert len(list((yolo_dir / "images").glob("*.png"))) == 2
    assert len(list((yolo_dir / "labels").glob("*.txt"))) == 2

    report = json.loads((ws / "reports" / "benchmark_results.json").read_text(encoding="utf-8"))
    assert report["sample_count"] == 2
    assert len(report["overlay_files"]) == 2


def _make_cobalt_plate(green: bool = False) -> np.ndarray:
    img = np.full((640, 640, 3), (30, 32, 36), dtype=np.uint8)
    mask = np.zeros((640, 640), dtype=np.uint8)
    cv2.circle(mask, (320, 320), 240, 255, -1)
    plate = np.full((640, 640, 3), (218, 222, 226), dtype=np.uint8)
    dark = (58, 138, 66) if green else (176, 54, 20)
    mid = (82, 168, 92) if green else (206, 92, 46)
    cv2.circle(plate, (320, 320), 175, mid, 3, cv2.LINE_AA)
    for idx in range(8):
        ang = idx * (2.0 * math.pi / 8.0)
        px = int(320 + 100 * math.cos(ang))
        py = int(320 + 100 * math.sin(ang))
        cv2.ellipse(plate, (px, py), (34, 15), int(round(math.degrees(ang))), 0, 360, dark, -1)
        cv2.line(
            plate,
            (320, 320),
            (int(320 + 118 * math.cos(ang)), int(320 + 118 * math.sin(ang))),
            (218, 222, 226),
            1,
            cv2.LINE_AA,
        )
    cv2.circle(plate, (320, 320), 34, dark, 4, cv2.LINE_AA)
    img[mask > 0] = plate[mask > 0]
    return img


def test_run_benchmark_prunes_stale_top_level_overlays_preserves_runtime(tmp_path: Path) -> None:
    """Verify run_benchmark removes stale top-level PNGs while leaving runtime/ intact."""
    repo_root = Path(__file__).resolve().parents[1]
    img_dir = tmp_path / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    clean_img = _make_cobalt_plate(green=False)
    cv2.imwrite(str(img_dir / "P1.png"), clean_img)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stand_label": "prune-test",
                "count": 1,
                "samples": [
                    {
                        "id": "KEEP-01",
                        "image_path": str(img_dir / "P1.png"),
                        "stage": "glazed",
                        "defect_class": "pass",
                        "expected_verdict": "PASS",
                        "auth_pair_id": None,
                        "is_counterfeit": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    overlays_dir = tmp_path / "overlays"
    runtime_dir = overlays_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stale_file = overlays_dir / "STALE-OLD-01.png"
    stale_file.write_bytes(b"stale")
    runtime_keep = runtime_dir / "keep.png"
    runtime_keep.write_bytes(b"runtime-keep")

    proc = _run_script(
        repo_root,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(manifest_path),
            "--output-json",
            str(tmp_path / "bench.json"),
            "--output-md",
            str(tmp_path / "bench.md"),
            "--overlays-dir",
            str(overlays_dir),
        ],
    )
    assert proc.returncode == 0, proc.stderr
    assert not stale_file.exists()
    assert (overlays_dir / "KEEP-01.png").is_file()
    assert runtime_keep.is_file()


def test_ingest_real_photos_custom_input_isolation_and_glazed_auth_pair(tmp_path: Path) -> None:
    """Verify --input isolation writes manifest inside input dir and pairs fake to glazed pass."""
    repo_root = Path(__file__).resolve().parents[1]
    ws = tmp_path / "ws"
    shutil.copytree(repo_root / "src", ws / "src")
    shutil.copytree(repo_root / "scripts", ws / "scripts")
    (ws / "data" / "real_photos").mkdir(parents=True, exist_ok=True)

    ext_dir = tmp_path / "external_photos"
    (ext_dir / "pass").mkdir(parents=True, exist_ok=True)
    (ext_dir / "fake").mkdir(parents=True, exist_ok=True)

    bisque = np.full((640, 640, 3), (30, 32, 36), dtype=np.uint8)
    cv2.circle(bisque, (320, 320), 240, (178, 180, 184), -1)
    cv2.imwrite(str(ext_dir / "pass" / "bisque_01.jpg"), bisque)

    glazed_cobalt = _make_cobalt_plate(green=False)
    cv2.imwrite(str(ext_dir / "pass" / "glazed_02.jpg"), glazed_cobalt)

    fake_green = _make_cobalt_plate(green=True)
    cv2.imwrite(str(ext_dir / "fake" / "fake_01.jpg"), fake_green)

    proc = _run_script(
        ws,
        ["scripts/ingest_real_photos.py", "--input", str(ext_dir), "--rebuild-all"],
    )
    assert proc.returncode == 0, proc.stderr
    assert not (ws / "data" / "real_photos" / "manifest.json").exists()
    ext_manifest_path = ext_dir / "manifest.json"
    assert ext_manifest_path.is_file()

    manifest = json.loads(ext_manifest_path.read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in manifest["samples"]}
    assert by_id["REAL-PASS-001"]["stage"] == "bisque"
    assert by_id["REAL-PASS-002"]["stage"] == "glazed"
    assert by_id["REAL-FAKE-003"]["auth_pair_id"] == "REAL-PASS-002"

    report = json.loads((ws / "reports" / "benchmark_results.json").read_text(encoding="utf-8"))
    assert report["sample_count"] == 3
    assert report["auth_benchmark"]["pair_count"] == 1
    assert report["auth_benchmark"]["auth_accuracy"] == 1.0


def test_ingest_real_photos_fake_falls_back_to_bisque_pass_when_no_glazed(tmp_path: Path) -> None:
    """Verify fake/ falls back to bisque pass when no glazed pass photo exists."""
    repo_root = Path(__file__).resolve().parents[1]
    ext_dir = tmp_path / "bisque_only"
    (ext_dir / "pass").mkdir(parents=True, exist_ok=True)
    (ext_dir / "fake").mkdir(parents=True, exist_ok=True)

    bisque = np.full((640, 640, 3), (30, 32, 36), dtype=np.uint8)
    cv2.circle(bisque, (320, 320), 240, (178, 180, 184), -1)
    cv2.imwrite(str(ext_dir / "pass" / "bisque_01.jpg"), bisque)
    cv2.imwrite(str(ext_dir / "fake" / "fake_01.jpg"), _make_cobalt_plate(green=True))

    proc = _run_script(
        repo_root,
        ["scripts/ingest_real_photos.py", "--input", str(ext_dir)],
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((ext_dir / "manifest.json").read_text(encoding="utf-8"))
    by_id = {s["id"]: s for s in manifest["samples"]}
    assert by_id["REAL-FAKE-002"]["auth_pair_id"] == "REAL-PASS-001"
