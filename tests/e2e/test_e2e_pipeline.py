"""Opaque-box 4-tier E2E test suite for CeramScan pipeline features F1..F9."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest

FORBIDDEN_REPOS = (
    "avito" + "_bot",
    "ducknoir" + "-tg",
    "facevs" + "-landing",
    "calccore" + "-platform",
)

AUTH_OK_VERDICT = "Подлинник"
AUTH_FAKE_VERDICT = "Подозрение на контрафакт или штамповку"


def _run_cli(
    venv_python: Path,
    args: list[str],
    cwd: Path,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    return subprocess.run(
        [str(venv_python), *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _find_emitted_manifests(*workspaces: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for workspace in workspaces:
        for path in sorted(workspace.rglob("*manifest*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
                manifests.append(payload)
    return manifests


def _copy_image_to(src_bgr: np.ndarray, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dest), src_bgr)
    assert ok, f"Failed to write image to {dest}"
    return dest


# ===========================================================================
# TIER 1: Feature Coverage (Happy-Path Equivalence Classes, F1..F9)
# ===========================================================================


@pytest.mark.tier1
@pytest.mark.f1
def test_t1_f1_01_venv_python_312(venv_python: Path) -> None:
    """Verify .venv Python interpreter runs and reports Python 3.12."""
    proc = subprocess.run(
        [str(venv_python), "-c", "import sys; print(tuple(sys.version_info[:2]))"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "(3, 12)"


@pytest.mark.tier1
@pytest.mark.f1
def test_t1_f1_02_core_packages_importable(venv_python: Path) -> None:
    """Verify all core CV, web, office, ONNX, and test modules import cleanly."""
    modules = (
        "cv2",
        "numpy",
        "skimage",
        "fastapi",
        "uvicorn",
        "qrcode",
        "PIL",
        "pptx",
        "docx",
        "pytest",
        "onnxruntime",
        "pypdf",
        "playwright",
        "httpx",
    )
    code = "; ".join(f"import {mod}" for mod in modules)
    proc = subprocess.run(
        [str(venv_python), "-c", code],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.tier1
@pytest.mark.f1
def test_t1_f1_03_requirements_pinned_complete(repo_root: Path) -> None:
    """Verify requirements.txt pins exact versions for all required libraries."""
    req_path = repo_root / "requirements.txt"
    assert req_path.is_file()
    lines = [
        ln.strip()
        for ln in req_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert len(lines) >= 15
    pinned_re = re.compile(r"^[A-Za-z0-9_.-]+==[^\s]+$")
    pkgs: set[str] = set()
    for line in lines:
        assert pinned_re.match(line), f"Unpinned requirement line: {line}"
        pkgs.add(line.split("==", 1)[0].lower().replace("_", "-"))

    required = {
        "opencv-python",
        "numpy",
        "scikit-image",
        "fastapi",
        "uvicorn",
        "qrcode",
        "pillow",
        "python-pptx",
        "python-docx",
        "pytest",
        "onnxruntime",
        "pypdf",
        "playwright",
    }
    missing = required - pkgs
    assert not missing, f"Missing packages in requirements.txt: {missing}"


@pytest.mark.tier1
@pytest.mark.f1
def test_t1_f1_04_gitignore_rules_present(repo_root: Path) -> None:
    """Verify .gitignore includes mandatory Python, DB, runtime, and photo exclusions."""
    gi_path = repo_root / ".gitignore"
    assert gi_path.is_file()
    content = gi_path.read_text(encoding="utf-8")
    expected_rules = (
        ".venv/",
        "__pycache__/",
        "*.db",
        "*.db-wal",
        "*.db-shm",
        ".agents/",
        "reports/overlays/runtime/",
        "reports/passports/",
        "reports/qr/",
        "data/real_photos/**",
    )
    for rule in expected_rules:
        assert rule in content, f"Missing rule in .gitignore: {rule}"


@pytest.mark.tier1
@pytest.mark.f1
def test_t1_f1_05_real_photos_scaffolding(repo_root: Path) -> None:
    """Verify data/real_photos/ exists with all 5 category folders and .gitkeep files."""
    base = repo_root / "data" / "real_photos"
    assert base.is_dir(), f"Missing {base}"
    assert (base / ".gitkeep").is_file(), "Missing data/real_photos/.gitkeep"
    for category in ("pass", "chip", "crack", "glaze", "fake"):
        sub = base / category
        assert sub.is_dir(), f"Missing category directory: {sub}"
        assert (sub / ".gitkeep").is_file(), f"Missing .gitkeep in {sub}"


@pytest.mark.tier1
@pytest.mark.f2
def test_t1_f2_01_foundation_docs_exist_and_headings(repo_root: Path) -> None:
    """Verify PROJECT.md, ARCHITECTURE.md, and DESIGN_SPEC.md exist and have core headings."""
    expected = (
        ("PROJECT.md", "# Project:"),
        ("ARCHITECTURE.md", "# CeramScan Architecture"),
        ("DESIGN_SPEC.md", "# CeramScan Design System"),
    )
    for filename, heading in expected:
        doc = repo_root / filename
        assert doc.is_file(), f"Missing {filename}"
        text = doc.read_text(encoding="utf-8")
        assert len(text.encode("utf-8")) > 1000
        assert heading in text


@pytest.mark.tier1
@pytest.mark.f2
def test_t1_f2_02_project_feature_and_milestone_inventory(repo_root: Path) -> None:
    """Verify PROJECT.md documents features F1..F16 and milestones M1..Final."""
    text = (repo_root / "PROJECT.md").read_text(encoding="utf-8")
    for idx in range(1, 17):
        assert f"| F{idx} " in text, f"Missing F{idx} in PROJECT.md"
    for milestone in ("| M1 |", "| M2 |", "| M3 |", "| M4 |", "| E2E |", "| Final |"):
        assert milestone in text, f"Missing milestone {milestone}"


@pytest.mark.tier1
@pytest.mark.f2
def test_t1_f2_03_architecture_contracts_locked(repo_root: Path) -> None:
    """Verify ARCHITECTURE.md locks DefectClass, GOST limits, 289-FZ weights, DDL, and CLI."""
    text = (repo_root / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for member in ("CRACK", "CHIP", "GLAZE_BARE", "PINHOLE_SPECK", "GLAZE_DRIP", "SHAPE_WARP"):
        assert member in text
    for token in (
        "1.5%",
        "20 mm2",
        "<= 8",
        "<= 3",
        "0.50",
        "0.25",
        "62.0",
        "12",
        "CREATE TABLE IF NOT EXISTS inspections",
        "CREATE TABLE IF NOT EXISTS passports",
        "CREATE TABLE IF NOT EXISTS passport_verifications",
        "generate_synthetic_ceramic.py",
        "run_benchmark.py",
        "ingest_real_photos.py",
    ):
        assert token in text, f"Missing architecture contract token: {token}"


@pytest.mark.tier1
@pytest.mark.f2
def test_t1_f2_04_design_spec_tokens_and_dom_ids(repo_root: Path) -> None:
    """Verify DESIGN_SPEC.md defines normative palette tokens, fonts, and DOM selectors."""
    text = (repo_root / "DESIGN_SPEC.md").read_text(encoding="utf-8")
    for token in (
        "#1D4ED8",
        "#1E3A8A",
        "#F8FAFC",
        "#0F172A",
        "IBM Plex Sans",
        "JetBrains Mono",
        "#tab-inspect",
        "#tab-passport",
        "#tab-dashboard",
        "#compare-viewport",
        "#passport-meter-fill",
        "#ledger-table",
    ):
        assert token in text, f"Missing DESIGN_SPEC token: {token}"


@pytest.mark.tier1
@pytest.mark.f2
def test_t1_f2_05_cross_doc_endpoint_parity(repo_root: Path) -> None:
    """Verify PROJECT.md and ARCHITECTURE.md agree on all HTTP routes and CLI scripts."""
    proj = (repo_root / "PROJECT.md").read_text(encoding="utf-8")
    arch = (repo_root / "ARCHITECTURE.md").read_text(encoding="utf-8")
    shared_tokens = (
        "/health",
        "/api/demo-samples",
        "/api/inspect",
        "/api/passport/register",
        "/api/passport/verify/{item_id}",
        "/api/analytics",
        "generate_synthetic_ceramic.py",
        "run_benchmark.py",
        "ingest_real_photos.py",
    )
    for token in shared_tokens:
        assert token in proj, f"{token} missing from PROJECT.md"
        assert token in arch, f"{token} missing from ARCHITECTURE.md"


@pytest.mark.tier1
@pytest.mark.f3
def test_t1_f3_01_full_item_glazed_preprocessing(
    ceramic_image_factory: Any,
) -> None:
    """Verify full-item glazed preprocessing outputs canonical 640x640 tensors and geometry."""
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    frame = preprocess_image(img, stage="glazed")
    assert frame.stage == "glazed"
    assert frame.roi_bgr.shape == (640, 640, 3)
    assert frame.roi_lab_l_clahe.shape == (640, 640)
    assert frame.roi_hsv.shape == (640, 640, 3)
    assert frame.geometry.scale_calibrated is True
    assert frame.perspective_applied is False
    assert frame.geometry.contour_deviation_pct < 1.5
    assert frame.latency_ms < 150.0


@pytest.mark.tier1
@pytest.mark.f3
def test_t1_f3_02_bisque_mode_disables_glare_mask(
    ceramic_image_factory: Any,
) -> None:
    """Verify bisque mode disables specular glare masking."""
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", glare=True)
    frame = preprocess_image(img, stage="bisque")
    assert frame.stage == "bisque"
    assert int(np.count_nonzero(frame.m_spec)) == 0
    assert frame.glare_ratio == 0.0
    assert frame.roi_lab_l_clahe.dtype == np.uint8


@pytest.mark.tier1
@pytest.mark.f3
def test_t1_f3_03_calibration_quad_perspective_warp(
    ceramic_image_factory: Any,
) -> None:
    """Verify skewed calibration quad triggers perspective warp and scale calibration."""
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", quad_frame=True)
    frame = preprocess_image(img, stage="glazed")
    assert frame.perspective_applied is True
    assert frame.geometry.scale_calibrated is True
    assert frame.roi_bgr.shape == (640, 640, 3)


@pytest.mark.tier1
@pytest.mark.f3
def test_t1_f3_04_cobalt_mask_isolation(ceramic_image_factory: Any) -> None:
    """Verify cobalt mask captures Gzhel blue pigment and ignores neutral bisque."""
    from src.cv_core.preprocessing import preprocess_image

    cobalt_img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    plain_img = ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", cobalt=False)
    cobalt_frame = preprocess_image(cobalt_img, stage="glazed")
    plain_frame = preprocess_image(plain_img, stage="bisque")
    assert cobalt_frame.cobalt_ratio > 0.015
    assert int(np.count_nonzero(cobalt_frame.m_cobalt)) > 0
    assert plain_frame.cobalt_ratio < 0.005


@pytest.mark.tier1
@pytest.mark.f3
def test_t1_f3_05_macro_patch_fallback_scale() -> None:
    """Verify tight macro crop without outer rim falls back cleanly to uncalibrated scale."""
    from src.cv_core.preprocessing import preprocess_image

    macro = np.full((200, 200, 3), (222, 224, 226), dtype=np.uint8)
    cv2.line(macro, (30, 100), (170, 105), (25, 25, 25), 2)
    frame = preprocess_image(macro, stage="glazed")
    assert frame.geometry.scale_calibrated is False
    assert frame.geometry.px_per_mm == pytest.approx(640.0 / 160.0, rel=0.05)
    assert int(np.count_nonzero(frame.m_interior)) > 0


@pytest.mark.tier1
@pytest.mark.f3
def test_t1_f3_06_custom_preprocess_config(ceramic_image_factory: Any) -> None:
    """Verify PreprocessConfig overrides target output dimensions and diameter scale."""
    from src.cv_core.preprocessing import PreprocessConfig, preprocess_image

    cfg = PreprocessConfig(target_size=512, reference_diameter_mm=250.0)
    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed")
    frame = preprocess_image(img, stage="glazed", config=cfg)
    assert frame.roi_bgr.shape == (512, 512, 3)
    assert frame.m_item.shape == (512, 512)
    assert frame.geometry.px_per_mm > 0.0


@pytest.mark.tier1
@pytest.mark.f4
def test_t1_f4_01_clean_bisque_pass(ceramic_image_factory: Any) -> None:
    """Verify clean bisque sample passes inspection under 300 ms CPU budget."""
    from src.cv_core.defect_detector import DefectDetector

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque")
    res = DefectDetector().inspect(img, stage="bisque")
    assert res.verdict == "PASS"
    assert res.gost_grade == "1 сорт"
    assert res.model_mode == "deterministic"
    assert res.defect_count == 0
    assert res.latency_ms < 300.0
    assert res.overlay_png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.tier1
@pytest.mark.f4
def test_t1_f4_02_clean_glazed_cobalt_no_false_positives(
    ceramic_image_factory: Any,
) -> None:
    """Verify intact hand-painted cobalt plate yields PASS with zero false defects."""
    from src.cv_core.defect_detector import DefectDetector

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    res = DefectDetector().inspect(img, stage="glazed")
    assert res.verdict == "PASS"
    assert res.cobalt_ratio > 0.015
    assert res.defect_count == 0
    assert res.latency_ms < 300.0


@pytest.mark.tier1
@pytest.mark.f4
@pytest.mark.parametrize("stage", ["bisque", "glazed"])
def test_t1_f4_03_crack_triggers_reject(
    ceramic_image_factory: Any,
    stage: str,
) -> None:
    """Verify crack triggers unconditional REJECT across both bisque and glazed stages."""
    from src.cv_core.defect_detector import DefectDetector
    from src.cv_core.preprocessing import DefectClass

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage=stage, crack=True)
    res = DefectDetector().inspect(img, stage=stage)  # type: ignore[arg-type]
    assert res.verdict == "REJECT"
    assert res.gost_grade == "Брак (REJECT)"
    assert any(d.defect_class == DefectClass.CRACK for d in res.defects)
    assert res.latency_ms < 300.0


@pytest.mark.tier1
@pytest.mark.f4
def test_t1_f4_04_rim_chip_detection(ceramic_image_factory: Any) -> None:
    """Verify rim chip is detected on the rim zone and triaged as REWORK or REJECT."""
    from src.cv_core.defect_detector import DefectDetector
    from src.cv_core.preprocessing import DefectClass

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", chip=True)
    res = DefectDetector().inspect(img, stage="glazed")
    assert res.verdict in ("REWORK", "REJECT")
    assert any(
        d.defect_class == DefectClass.CHIP and d.zone == "rim"
        for d in res.defects
    )


@pytest.mark.tier1
@pytest.mark.f4
def test_t1_f4_05_glazed_bare_patch_detection(ceramic_image_factory: Any) -> None:
    """Verify glaze bare patch (>20 mm2) triggers GLAZE_BARE and non-PASS verdict."""
    from src.cv_core.defect_detector import DefectDetector
    from src.cv_core.preprocessing import DefectClass

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", glaze_bare=True)
    res = DefectDetector().inspect(img, stage="glazed")
    assert res.verdict in ("REWORK", "REJECT")
    assert any(d.defect_class == DefectClass.GLAZE_BARE for d in res.defects)


@pytest.mark.tier1
@pytest.mark.f4
def test_t1_f4_06_onnx_yolov8_weights_inference(
    ceramic_image_factory: Any,
    onnx_weights_factory: Callable[[str], Path],
) -> None:
    """Verify DefectDetector executes real YOLOv8 ONNX weights via CPUExecutionProvider."""
    from src.cv_core.defect_detector import DefectDetector
    from src.cv_core.preprocessing import DefectClass

    weights = onnx_weights_factory("yolov8")
    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed")
    res = DefectDetector(weights_path=weights).inspect(img, stage="glazed")
    assert res.model_mode == "onnx"
    assert res.verdict == "REJECT"
    assert any(
        d.source == "onnx" and d.defect_class == DefectClass.CRACK
        for d in res.defects
    )


@pytest.mark.tier1
@pytest.mark.f5
def test_t1_f5_01_extract_fingerprint_schema(ceramic_image_factory: Any) -> None:
    """Verify extract_fingerprint outputs valid pHash, 288-bin cobalt hist, and ORB data."""
    from src.cv_core.fingerprint import extract_fingerprint
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    frame = preprocess_image(img, stage="glazed")
    sig = extract_fingerprint(frame)
    assert len(sig.phash_hex) == 16
    assert int(sig.phash_hex, 16) >= 0
    assert len(sig.cobalt_hist) == 288
    assert sum(sig.cobalt_hist) == pytest.approx(1.0, abs=1e-3)
    assert sig.orb_count > 12
    raw_desc = base64.b64decode(sig.descriptors_b64)
    assert len(raw_desc) == sig.orb_count * 32


@pytest.mark.tier1
@pytest.mark.f5
def test_t1_f5_02_self_verification_authentic(ceramic_image_factory: Any) -> None:
    """Verify self-comparison produces authentic verdict with high match scores."""
    from src.cv_core.fingerprint import extract_fingerprint, verify_fingerprint
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    frame = preprocess_image(img, stage="glazed")
    sig = extract_fingerprint(frame)
    res = verify_fingerprint(sig, frame)
    assert res.verdict == AUTH_OK_VERDICT
    assert res.match_percent >= 90.0
    assert res.phash_score == pytest.approx(1.0, abs=1e-4)
    assert res.color_score >= 0.98
    assert res.matched_kp >= 12


@pytest.mark.tier1
@pytest.mark.f5
def test_t1_f5_03_mild_rotation_brightness_still_authentic(
    ceramic_image_factory: Any,
) -> None:
    """Verify mildly rotated and brightness-shifted genuine plate remains authentic."""
    from src.cv_core.fingerprint import extract_fingerprint, verify_fingerprint
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    ref_frame = preprocess_image(img, stage="glazed")
    sig = extract_fingerprint(ref_frame)

    h, w = img.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 5.0, 1.0)
    rotated = cv2.warpAffine(img, rot, (w, h), borderValue=(28, 28, 30))
    shifted = np.clip(rotated.astype(np.int16) + 10, 0, 255).astype(np.uint8)

    probe_frame = preprocess_image(shifted, stage="glazed")
    res = verify_fingerprint(sig, probe_frame)
    assert res.verdict == AUTH_OK_VERDICT
    assert res.match_percent >= 62.0
    assert res.matched_kp >= 12


@pytest.mark.tier1
@pytest.mark.f5
def test_t1_f5_04_counterfeit_pattern_rejected(ceramic_image_factory: Any) -> None:
    """Verify counterfeit pattern is rejected against genuine cobalt signature."""
    from src.cv_core.fingerprint import extract_fingerprint, verify_fingerprint
    from src.cv_core.preprocessing import preprocess_image

    genuine = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    fake = ceramic_image_factory.make_counterfeit_plate_bgr()
    sig = extract_fingerprint(preprocess_image(genuine, stage="glazed"))
    res = verify_fingerprint(sig, preprocess_image(fake, stage="glazed"))
    assert res.verdict == AUTH_FAKE_VERDICT
    assert res.match_percent < 62.0


@pytest.mark.tier1
@pytest.mark.f5
def test_t1_f5_05_weighted_score_formula_exact(ceramic_image_factory: Any) -> None:
    """Verify match_percent follows exact 0.50*ORB + 0.25*pHash + 0.25*Color formula."""
    from src.cv_core.fingerprint import extract_fingerprint, verify_fingerprint
    from src.cv_core.preprocessing import preprocess_image

    genuine = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    fake = ceramic_image_factory.make_counterfeit_plate_bgr()
    g_frame = preprocess_image(genuine, stage="glazed")
    f_frame = preprocess_image(fake, stage="glazed")
    sig = extract_fingerprint(g_frame)

    for probe in (g_frame, f_frame):
        res = verify_fingerprint(sig, probe)
        expected = round(
            100.0 * (0.50 * res.orb_score + 0.25 * res.phash_score + 0.25 * res.color_score),
            2,
        )
        assert res.match_percent == pytest.approx(expected, abs=0.02)


@pytest.mark.tier1
@pytest.mark.f6
def test_t1_f6_01_sqlite_wal_and_exact_ddl(isolated_workspace: dict[str, Path]) -> None:
    """Verify CeramStorage initializes SQLite WAL pragmas and exact DDL schema."""
    from src.cv_core.storage import CeramStorage

    db_path = isolated_workspace["db_path"]
    CeramStorage(
        db_path=db_path,
        qr_dir=isolated_workspace["qr_dir"],
        passport_ref_dir=isolated_workspace["passport_dir"],
    )
    assert db_path.is_file()

    with sqlite3.connect(db_path) as conn:
        journal = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert str(journal).lower() == "wal"

        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()
        }
        assert {"inspections", "passports", "passport_verifications"}.issubset(tables)

        insp_cols = {row[1] for row in conn.execute("PRAGMA table_info(inspections);")}
        assert {
            "id",
            "item_id",
            "stage",
            "shift",
            "verdict",
            "gost_grade",
            "model_mode",
            "defect_count",
            "primary_defect",
            "contour_deviation_pct",
            "glare_ratio",
            "latency_ms",
            "defects_json",
            "gost_rules_json",
            "overlay_path",
            "created_at",
        }.issubset(insp_cols)

        pass_cols = {row[1] for row in conn.execute("PRAGMA table_info(passports);")}
        assert {
            "item_id",
            "sku",
            "master_name",
            "batch_note",
            "stage",
            "qr_payload",
            "qr_image_path",
            "reference_image_path",
            "signature_json",
            "created_at",
        }.issubset(pass_cols)

        ver_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(passport_verifications);")
        }
        assert {
            "id",
            "item_id",
            "match_percent",
            "orb_score",
            "phash_score",
            "color_score",
            "matched_kp",
            "verdict",
            "latency_ms",
            "created_at",
        }.issubset(ver_cols)


@pytest.mark.tier1
@pytest.mark.f6
def test_t1_f6_02_persist_and_query_inspection(
    isolated_api_server: tuple[str, dict[str, Path]],
    ceramic_image_factory: Any,
) -> None:
    """Verify inspection persists end-to-end into SQLite inspections table."""
    base_url, paths = isolated_api_server
    png = ceramic_image_factory.make_bisque_plate_png(with_crack=True)

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        resp = client.post(
            "/api/inspect",
            data={"stage": "bisque", "item_id": "GZHEL-INSP-01", "shift": "S2"},
            files={"file": ("plate.png", png, "image/png")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["inspection_id"] is not None

    with sqlite3.connect(paths["db_path"]) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM inspections WHERE id = ?;",
            (body["inspection_id"],),
        ).fetchone()
        assert row is not None
        assert row["item_id"] == "GZHEL-INSP-01"
        assert row["stage"] == "bisque"
        assert row["shift"] == "S2"
        assert row["verdict"] == body["verdict"]
        assert row["gost_grade"] == body["gost_grade"]
        assert row["model_mode"] == body["model_mode"]
        assert row["defect_count"] == body["defect_count"]
        assert isinstance(json.loads(row["defects_json"]), list)
        assert isinstance(json.loads(row["gost_rules_json"]), list)
        assert row["created_at"]


@pytest.mark.tier1
@pytest.mark.f6
def test_t1_f6_03_register_passport_qr_and_reference_png(
    isolated_api_server: tuple[str, dict[str, Path]],
    ceramic_image_factory: Any,
) -> None:
    """Verify passport registration persists SQLite row, readable QR PNG, and ref PNG."""
    base_url, paths = isolated_api_server
    png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        resp = client.post(
            "/api/passport/register",
            data={
                "item_id": "GZHEL-2026-001",
                "sku": "GZHEL-CUP-7",
                "master_name": "Мастер Сидоров",
                "batch_note": "Печь-1",
                "stage": "glazed",
            },
            files={"file": ("cobalt.png", png, "image/png")},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["qr_payload"] == "CERAMSCAN:GZHEL-2026-001"

    qr_path = paths["qr_dir"] / "GZHEL-2026-001.png"
    ref_path = paths["passport_dir"] / "GZHEL-2026-001.png"
    assert qr_path.is_file()
    assert ref_path.is_file()

    qr_img = cv2.imread(str(qr_path))
    assert qr_img is not None
    decoded_text, _, _ = cv2.QRCodeDetector().detectAndDecode(qr_img)
    assert decoded_text == "CERAMSCAN:GZHEL-2026-001"

    ref_img = cv2.imread(str(ref_path))
    assert ref_img is not None and ref_img.shape[0] >= 32

    with sqlite3.connect(paths["db_path"]) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM passports WHERE item_id = ?;",
            ("GZHEL-2026-001",),
        ).fetchone()
        assert row is not None
        assert row["qr_payload"] == "CERAMSCAN:GZHEL-2026-001"
        assert row["sku"] == "GZHEL-CUP-7"
        sig_data = json.loads(row["signature_json"])
        assert "phash_hex" in sig_data and "cobalt_hist" in sig_data


@pytest.mark.tier1
@pytest.mark.f6
def test_t1_f6_04_persist_passport_verification(
    isolated_api_server: tuple[str, dict[str, Path]],
    ceramic_image_factory: Any,
) -> None:
    """Verify passport verification logs metrics into SQLite passport_verifications table."""
    base_url, paths = isolated_api_server
    png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        reg = client.post(
            "/api/passport/register",
            data={"item_id": "GZHEL-VER-01"},
            files={"file": ("ref.png", png, "image/png")},
        )
        assert reg.status_code == 200
        ver = client.post(
            "/api/passport/verify/GZHEL-VER-01",
            files={"file": ("probe.png", png, "image/png")},
        )
        assert ver.status_code == 200
        vbody = ver.json()

    with sqlite3.connect(paths["db_path"]) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM passport_verifications WHERE item_id = ?;",
            ("GZHEL-VER-01",),
        ).fetchone()
        assert row is not None
        assert row["match_percent"] == pytest.approx(vbody["match_percent"], abs=0.05)
        assert row["orb_score"] == pytest.approx(vbody["orb_score"], abs=0.01)
        assert row["phash_score"] == pytest.approx(vbody["phash_score"], abs=0.01)
        assert row["color_score"] == pytest.approx(vbody["color_score"], abs=0.01)
        assert row["matched_kp"] == vbody["matched_kp"]
        assert row["verdict"] == vbody["verdict"]


@pytest.mark.tier1
@pytest.mark.f6
def test_t1_f6_05_foreign_key_cascade_delete(isolated_workspace: dict[str, Path]) -> None:
    """Verify deleting a parent passport cascades to child passport_verifications rows."""
    from src.cv_core.storage import CeramStorage

    db_path = isolated_workspace["db_path"]
    CeramStorage(
        db_path=db_path,
        qr_dir=isolated_workspace["qr_dir"],
        passport_ref_dir=isolated_workspace["passport_dir"],
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute(
            """
            INSERT INTO passports (
                item_id, sku, master_name, batch_note, stage,
                qr_payload, qr_image_path, reference_image_path, signature_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "GZHEL-CAS-01",
                "SKU-1",
                "Мастер",
                "",
                "glazed",
                "CERAMSCAN:GZHEL-CAS-01",
                "qr.png",
                "ref.png",
                "{}",
                "2026-11-19T00:00:00Z",
            ),
        )
        conn.execute(
            """
            INSERT INTO passport_verifications (
                item_id, match_percent, orb_score, phash_score,
                color_score, matched_kp, verdict, latency_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                "GZHEL-CAS-01",
                88.5,
                0.9,
                0.85,
                0.89,
                25,
                AUTH_OK_VERDICT,
                12.0,
                "2026-11-19T00:01:00Z",
            ),
        )
        conn.commit()
        before = conn.execute(
            "SELECT COUNT(*) FROM passport_verifications WHERE item_id = ?;",
            ("GZHEL-CAS-01",),
        ).fetchone()[0]
        assert before == 1

        conn.execute("DELETE FROM passports WHERE item_id = ?;", ("GZHEL-CAS-01",))
        conn.commit()
        after = conn.execute(
            "SELECT COUNT(*) FROM passport_verifications WHERE item_id = ?;",
            ("GZHEL-CAS-01",),
        ).fetchone()[0]
        assert after == 0


@pytest.mark.tier1
@pytest.mark.f7
def test_t1_f7_01_default_26_sample_generation(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify generate_synthetic_ceramic.py emits 26-sample manifest with stand label."""
    out_dir = tmp_path / "syn26"
    proc = _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(out_dir), "--seed", "42"],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.is_file()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["stand_label"] == "синтетический валидационный стенд v0.1"
    assert data["seed"] == 42
    assert data["count"] == 26
    assert len(data["samples"]) == 26


@pytest.mark.tier1
@pytest.mark.f7
def test_t1_f7_02_manifest_sample_schema_and_files(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify every generated sample record has required keys and readable PNG image."""
    out_dir = tmp_path / "syn_schema"
    proc = _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(out_dir), "--seed", "42"],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))

    for sample in data["samples"]:
        assert {
            "id",
            "image_path",
            "stage",
            "defect_class",
            "expected_verdict",
            "auth_pair_id",
            "is_counterfeit",
        }.issubset(sample.keys())
        assert sample["stage"] in {"bisque", "glazed"}
        assert sample["expected_verdict"] in {"PASS", "REWORK", "REJECT"}
        assert isinstance(sample["is_counterfeit"], bool)
        img_path = Path(sample["image_path"])
        if not img_path.is_absolute():
            img_path = out_dir / img_path if (out_dir / img_path).exists() else repo_root / img_path
        assert img_path.is_file(), f"Missing generated image {sample['image_path']}"
        img = cv2.imread(str(img_path))
        assert img is not None and img.shape[0] >= 32 and img.shape[1] >= 32


@pytest.mark.tier1
@pytest.mark.f7
def test_t1_f7_03_required_stage_and_defect_distribution(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify default 26-sample stand includes bisque/glazed classes and >=4 counterfeits."""
    out_dir = tmp_path / "syn_dist"
    proc = _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(out_dir), "--seed", "42"],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    samples = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["samples"]

    bisque_classes = {s["defect_class"] for s in samples if s["stage"] == "bisque"}
    glazed_classes = {s["defect_class"] for s in samples if s["stage"] == "glazed"}
    fake_count = sum(1 for s in samples if s["is_counterfeit"] is True)

    assert {"pass", "crack", "chip", "shape_warp"}.issubset(bisque_classes)
    assert {"pass", "crack", "glaze_bare"}.issubset(glazed_classes)
    assert glazed_classes & {"pinhole_speck", "glaze_drip"}
    assert fake_count >= 4


@pytest.mark.tier1
@pytest.mark.f7
def test_t1_f7_04_reproducible_seed_determinism(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify identical seed produces identical images while different seed changes pixels."""
    dir_a = tmp_path / "seed42_a"
    dir_b = tmp_path / "seed42_b"
    dir_c = tmp_path / "seed99"
    for target, seed in ((dir_a, "42"), (dir_b, "42"), (dir_c, "99")):
        proc = _run_cli(
            venv_python,
            ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(target), "--seed", seed],
            cwd=repo_root,
        )
        assert proc.returncode == 0, proc.stderr

    man_a = json.loads((dir_a / "manifest.json").read_text(encoding="utf-8"))
    man_b = json.loads((dir_b / "manifest.json").read_text(encoding="utf-8"))
    man_c = json.loads((dir_c / "manifest.json").read_text(encoding="utf-8"))

    assert [(s["id"], s["stage"], s["defect_class"]) for s in man_a["samples"]] == [
        (s["id"], s["stage"], s["defect_class"]) for s in man_b["samples"]
    ]

    def _first_img_hash(root: Path, sample: dict[str, Any]) -> str:
        p = Path(sample["image_path"])
        resolved = p if p.is_absolute() else (root / p if (root / p).exists() else repo_root / p)
        return hashlib.sha256(resolved.read_bytes()).hexdigest()

    hashes_a = [_first_img_hash(dir_a, s) for s in man_a["samples"][:3]]
    hashes_b = [_first_img_hash(dir_b, s) for s in man_b["samples"][:3]]
    hashes_c = [_first_img_hash(dir_c, s) for s in man_c["samples"][:3]]
    assert hashes_a == hashes_b
    assert hashes_a != hashes_c


@pytest.mark.tier1
@pytest.mark.f7
def test_t1_f7_05_custom_count_30(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify --count 30 produces 30 samples with unique IDs."""
    out_dir = tmp_path / "syn30"
    proc = _run_cli(
        venv_python,
        [
            "scripts/generate_synthetic_ceramic.py",
            "--output-dir",
            str(out_dir),
            "--count",
            "30",
            "--seed",
            "7",
        ],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["count"] == 30
    assert len(data["samples"]) == 30
    ids = [s["id"] for s in data["samples"]]
    assert len(set(ids)) == 30


@pytest.mark.tier1
@pytest.mark.f8
def test_t1_f8_01_benchmark_json_exact_schema(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify run_benchmark.py writes benchmark_results.json matching exact contract schema."""
    syn_dir = tmp_path / "syn"
    proc_gen = _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--seed", "42"],
        cwd=repo_root,
    )
    assert proc_gen.returncode == 0, proc_gen.stderr

    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    overlays_dir = tmp_path / "overlays"
    proc_bench = _run_cli(
        venv_python,
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
        cwd=repo_root,
    )
    assert proc_bench.returncode == 0, proc_bench.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert {
        "generated_at",
        "dataset_label",
        "model_mode",
        "sample_count",
        "avg_latency_ms",
        "p95_latency_ms",
        "precision",
        "recall",
        "f1",
        "verdict_accuracy",
        "per_class",
        "auth_benchmark",
        "overlay_files",
    }.issubset(report.keys())
    assert {
        "pair_count",
        "auth_accuracy",
        "genuine_mean_match",
        "fake_mean_match",
    }.issubset(report["auth_benchmark"].keys())


@pytest.mark.tier1
@pytest.mark.f8
def test_t1_f8_02_cpu_latency_sla_and_valid_metrics(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify measured CPU benchmark stays below 300 ms/frame with valid metric bounds."""
    syn_dir = tmp_path / "syn"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--seed", "42"],
        cwd=repo_root,
    )
    out_json = tmp_path / "bench.json"
    proc = _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(syn_dir / "manifest.json"),
            "--output-json",
            str(out_json),
            "--output-md",
            str(tmp_path / "bench.md"),
            "--overlays-dir",
            str(tmp_path / "overlays"),
        ],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["sample_count"] == 26
    assert 0.0 < report["avg_latency_ms"] < 300.0
    assert report["p95_latency_ms"] >= report["avg_latency_ms"]
    for metric in ("precision", "recall", "f1", "verdict_accuracy"):
        assert 0.0 <= report[metric] <= 1.0


@pytest.mark.tier1
@pytest.mark.f8
def test_t1_f8_03_summary_markdown_parity(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify benchmark_summary.md matches dataset_label and precision/recall/F1 metrics."""
    syn_dir = tmp_path / "syn"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--seed", "42"],
        cwd=repo_root,
    )
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    proc = _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(syn_dir / "manifest.json"),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
            "--overlays-dir",
            str(tmp_path / "overlays"),
        ],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    md_text = out_md.read_text(encoding="utf-8")
    assert report["dataset_label"] in md_text
    assert f"{report['avg_latency_ms']:.1f}" in md_text or f"{report['avg_latency_ms']:.2f}" in md_text
    for heading in ("Precision", "Recall", "F1"):
        assert heading.lower() in md_text.lower()


@pytest.mark.tier1
@pytest.mark.f8
def test_t1_f8_04_side_by_side_overlays_generated(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify run_benchmark.py generates 26 side-by-side (width > height) overlay PNGs."""
    syn_dir = tmp_path / "syn"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--seed", "42"],
        cwd=repo_root,
    )
    out_json = tmp_path / "bench.json"
    overlays_dir = tmp_path / "overlays"
    proc = _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(syn_dir / "manifest.json"),
            "--output-json",
            str(out_json),
            "--output-md",
            str(tmp_path / "bench.md"),
            "--overlays-dir",
            str(overlays_dir),
        ],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    overlay_files = report["overlay_files"]
    assert len(overlay_files) == 26
    for raw_path in overlay_files:
        p = Path(raw_path)
        resolved = p if p.is_absolute() else (repo_root / p if (repo_root / p).exists() else overlays_dir / p.name)
        assert resolved.is_file(), f"Missing overlay PNG: {raw_path}"
        img = cv2.imread(str(resolved))
        assert img is not None
        h, w = img.shape[:2]
        assert w > h, f"Expected side-by-side overlay width > height, got {w}x{h}"


@pytest.mark.tier1
@pytest.mark.f8
def test_t1_f8_05_auth_benchmark_discrimination(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify auth_benchmark reports >=4 pairs and higher mean match for genuine items."""
    syn_dir = tmp_path / "syn"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--seed", "42"],
        cwd=repo_root,
    )
    out_json = tmp_path / "bench.json"
    proc = _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(syn_dir / "manifest.json"),
            "--output-json",
            str(out_json),
            "--output-md",
            str(tmp_path / "bench.md"),
            "--overlays-dir",
            str(tmp_path / "overlays"),
        ],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    auth = json.loads(out_json.read_text(encoding="utf-8"))["auth_benchmark"]
    assert auth["pair_count"] >= 4
    assert 0.0 <= auth["auth_accuracy"] <= 1.0
    assert auth["genuine_mean_match"] > auth["fake_mean_match"]


@pytest.mark.tier1
@pytest.mark.f9
def test_t1_f9_01_ingest_folder_mapping(
    isolated_repo_workspace: Path,
    venv_python: Path,
    real_photos_dir_factory: Callable[[bool], Path],
) -> None:
    """Verify ingest_real_photos maps pass/chip/crack/glaze/fake folders per contract."""
    photos_dir = real_photos_dir_factory(True)
    proc = _run_cli(
        venv_python,
        ["scripts/ingest_real_photos.py", "--input", str(photos_dir), "--rebuild-all"],
        cwd=isolated_repo_workspace,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    manifests = _find_emitted_manifests(isolated_repo_workspace, photos_dir)
    assert manifests, "Expected manifest.json emitted during real-photo ingest"
    samples = manifests[0]["samples"]
    by_name = {Path(s["image_path"]).name: s for s in samples}

    assert by_name["glazed_plate_03.jpg"]["defect_class"] == "pass"
    assert by_name["glazed_plate_03.jpg"]["expected_verdict"] == "PASS"
    assert by_name["chip_01.jpg"]["defect_class"] == "chip"
    assert by_name["chip_01.jpg"]["expected_verdict"] == "REJECT"
    assert by_name["crack_01.jpg"]["defect_class"] == "crack"
    assert by_name["crack_01.jpg"]["expected_verdict"] == "REJECT"
    assert by_name["glaze_01.jpg"]["defect_class"] == "glaze_bare"
    assert by_name["glaze_01.jpg"]["expected_verdict"] == "REWORK"
    assert by_name["fake_01.jpg"]["is_counterfeit"] is True
    assert by_name["fake_01.jpg"]["auth_pair_id"] is not None


@pytest.mark.tier1
@pytest.mark.f9
def test_t1_f9_02_bisque_util_filename_stage_inference(
    isolated_repo_workspace: Path,
    venv_python: Path,
    real_photos_dir_factory: Callable[[bool], Path],
) -> None:
    """Verify filenames containing bisque or util infer stage='bisque'."""
    photos_dir = real_photos_dir_factory(True)
    proc = _run_cli(
        venv_python,
        ["scripts/ingest_real_photos.py", "--input", str(photos_dir), "--rebuild-all"],
        cwd=isolated_repo_workspace,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    manifests = _find_emitted_manifests(isolated_repo_workspace, photos_dir)
    assert manifests
    by_name = {Path(s["image_path"]).name: s for s in manifests[0]["samples"]}
    assert by_name["bisque_plate_01.jpg"]["stage"] == "bisque"
    assert by_name["util_cup_02.jpg"]["stage"] == "bisque"
    assert by_name["glazed_plate_03.jpg"]["stage"] == "glazed"


@pytest.mark.tier1
@pytest.mark.f9
def test_t1_f9_03_export_yolo_structure_and_labels(
    isolated_repo_workspace: Path,
    venv_python: Path,
    real_photos_dir_factory: Callable[[bool], Path],
    tmp_path: Path,
) -> None:
    """Verify --export-yolo emits data.yaml (nc: 6), images/, and normalized labels."""
    photos_dir = real_photos_dir_factory(True)
    yolo_dir = tmp_path / "yolo_out"
    proc = _run_cli(
        venv_python,
        [
            "scripts/ingest_real_photos.py",
            "--input",
            str(photos_dir),
            "--export-yolo",
            str(yolo_dir),
        ],
        cwd=isolated_repo_workspace,
    )
    assert proc.returncode == 0, proc.stderr
    yaml_path = yolo_dir / "data.yaml"
    assert yaml_path.is_file()
    yaml_text = yaml_path.read_text(encoding="utf-8")
    assert "nc: 6" in yaml_text
    assert (yolo_dir / "images").is_dir()
    labels = list((yolo_dir / "labels").glob("*.txt"))
    assert len(labels) >= 5
    non_empty = [ln for lbl in labels for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert non_empty, "Expected at least one YOLO annotation line for defect samples"
    box_re = re.compile(r"^[0-5]\s+[01](?:\.\d+)?\s+[01](?:\.\d+)?\s+[01](?:\.\d+)?\s+[01](?:\.\d+)?$")
    for line in non_empty:
        assert box_re.match(line.strip()), f"Invalid YOLO line: {line}"


@pytest.mark.tier1
@pytest.mark.f9
def test_t1_f9_04_rebuild_all_updates_benchmark_and_overlays(
    isolated_repo_workspace: Path,
    venv_python: Path,
    real_photos_dir_factory: Callable[[bool], Path],
) -> None:
    """Verify --rebuild-all refreshes benchmark_results.json, summary.md, and overlays."""
    photos_dir = real_photos_dir_factory(True)
    proc = _run_cli(
        venv_python,
        ["scripts/ingest_real_photos.py", "--input", str(photos_dir), "--rebuild-all"],
        cwd=isolated_repo_workspace,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    bench_json = isolated_repo_workspace / "reports" / "benchmark_results.json"
    bench_md = isolated_repo_workspace / "reports" / "benchmark_summary.md"
    assert bench_json.is_file()
    assert bench_md.is_file()
    report = json.loads(bench_json.read_text(encoding="utf-8"))
    assert report["sample_count"] == 7
    assert len(report["overlay_files"]) == 7


@pytest.mark.tier1
@pytest.mark.f9
def test_t1_f9_05_ingest_with_onnx_weights(
    isolated_repo_workspace: Path,
    venv_python: Path,
    real_photos_dir_factory: Callable[[bool], Path],
    onnx_weights_factory: Callable[[str], Path],
) -> None:
    """Verify ingest_real_photos --weights propagates ONNX model_mode into benchmark JSON."""
    photos_dir = real_photos_dir_factory(True)
    weights = onnx_weights_factory("yolov8")
    proc = _run_cli(
        venv_python,
        [
            "scripts/ingest_real_photos.py",
            "--input",
            str(photos_dir),
            "--weights",
            str(weights),
            "--rebuild-all",
        ],
        cwd=isolated_repo_workspace,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(
        (isolated_repo_workspace / "reports" / "benchmark_results.json").read_text(encoding="utf-8")
    )
    assert report["model_mode"] == "onnx"


# ===========================================================================
# TIER 2: Boundary, Corner & Negative Cases (F1..F9)
# ===========================================================================


@pytest.mark.tier2
@pytest.mark.f1
def test_t2_f1_01_gitignore_blocks_runtime_artifacts(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Verify .gitignore rejects runtime venv, bytecode, SQLite DBs, and outputs."""
    git_repo = tmp_path / "gi_check"
    git_repo.mkdir()
    shutil.copy2(repo_root / ".gitignore", git_repo / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=str(git_repo), check=True)

    ignored_paths = (
        ".venv/bin/python",
        "src/__pycache__/x.pyc",
        "data/ceramscan.db",
        "data/ceramscan.db-wal",
        "reports/overlays/runtime/a.png",
        "reports/passports/a.png",
        "reports/qr/a.png",
        "tmp_out.png",
        ".agents/log.md",
    )
    for rel in ignored_paths:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", rel],
            cwd=str(git_repo),
        )
        assert proc.returncode == 0, f"Expected .gitignore to ignore {rel}"


@pytest.mark.tier2
@pytest.mark.f1
def test_t2_f1_02_gitignore_blocks_real_photos_keeps_gitkeep(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Verify .gitignore ignores real factory photos while preserving .gitkeep files."""
    git_repo = tmp_path / "gi_photos"
    git_repo.mkdir()
    shutil.copy2(repo_root / ".gitignore", git_repo / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=str(git_repo), check=True)

    ign = subprocess.run(
        ["git", "check-ignore", "-q", "data/real_photos/pass/img.jpg"],
        cwd=str(git_repo),
    )
    assert ign.returncode == 0

    for keep in ("data/real_photos/.gitkeep", "data/real_photos/pass/.gitkeep"):
        kept = subprocess.run(["git", "check-ignore", "-q", keep], cwd=str(git_repo))
        assert kept.returncode == 1, f".gitkeep must not be ignored: {keep}"


@pytest.mark.tier2
@pytest.mark.f1
def test_t2_f1_03_gitignore_allows_tracked_source_and_reports(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    """Verify .gitignore never ignores source code, docs, or benchmark reports."""
    git_repo = tmp_path / "gi_src"
    git_repo.mkdir()
    shutil.copy2(repo_root / ".gitignore", git_repo / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=str(git_repo), check=True)

    for tracked in (
        "src/api.py",
        "PROJECT.md",
        "requirements.txt",
        "reports/benchmark_results.json",
    ):
        proc = subprocess.run(["git", "check-ignore", "-q", tracked], cwd=str(git_repo))
        assert proc.returncode == 1, f"Tracked file unexpectedly ignored: {tracked}"


@pytest.mark.tier2
@pytest.mark.f1
def test_t2_f1_04_requirements_no_unpinned_or_duplicate_pkgs(repo_root: Path) -> None:
    """Verify requirements.txt contains zero unpinned specifiers or duplicate packages."""
    lines = [
        ln.strip()
        for ln in (repo_root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    seen: set[str] = set()
    for line in lines:
        for bad_op in (">=", "<=", "~=", "!=", "-e ", "file://"):
            assert bad_op not in line, f"Forbidden unpinned operator {bad_op} in {line}"
        name = line.split("==", 1)[0].strip().lower().replace("_", "-")
        assert name not in seen, f"Duplicate package entry: {name}"
        seen.add(name)


@pytest.mark.tier2
@pytest.mark.f1
def test_t2_f1_05_venv_dependencies_resolved(
    repo_root: Path,
    venv_python: Path,
) -> None:
    """Verify every pinned requirement and its active transitive deps resolve via importlib.metadata."""
    script = "\n".join(
        [
            "import importlib.metadata as im",
            "from pathlib import Path",
            "from packaging.requirements import Requirement",
            "reqs = [l.strip() for l in Path('requirements.txt').read_text().splitlines() if l.strip() and not l.strip().startswith('#')]",
            "for r_line in reqs:",
            "    req = Requirement(r_line)",
            "    dist = im.distribution(req.name)",
            "    assert req.specifier.contains(dist.version), f'{req.name}: {dist.version} != {req.specifier}'",
            "    for dep in (dist.requires or []):",
            "        dreq = Requirement(dep)",
            "        if dreq.marker and not dreq.marker.evaluate():",
            "            continue",
            "        d_ver = im.version(dreq.name)",
            "        assert dreq.specifier.contains(d_ver), f'{req.name}->{dreq}: {d_ver}'",
            "print(len(reqs))",
        ]
    )
    proc = subprocess.run(
        [str(venv_python), "-c", script],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout.strip()) >= 15


@pytest.mark.tier2
@pytest.mark.f2
def test_t2_f2_01_foundation_docs_slop_scan_zero(
    repo_root: Path,
    slop_scanner_cmd: list[str],
) -> None:
    """Verify foundation docs pass slop_scan.py with 0 findings and zero U+2014."""
    docs = [repo_root / "PROJECT.md", repo_root / "ARCHITECTURE.md", repo_root / "DESIGN_SPEC.md"]
    proc = subprocess.run(
        [*slop_scanner_cmd, "--ignore-inline-code", *(str(d) for d in docs)],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for doc in docs:
        assert "\u2014" not in doc.read_text(encoding="utf-8")


@pytest.mark.tier2
@pytest.mark.f2
def test_t2_f2_02_design_spec_no_forbidden_tokens_in_css_block(repo_root: Path) -> None:
    """Verify DESIGN_SPEC.md CSS token blocks exclude banned purple, terracotta, blur, and Inter."""
    text = (repo_root / "DESIGN_SPEC.md").read_text(encoding="utf-8")
    css_blocks = re.findall(r"```css(.*?)```", text, flags=re.DOTALL)
    assert css_blocks
    combined_css = "\n".join(css_blocks)
    for banned in ("#7c3aed", "#d97757", "#f4f1ea", "backdrop-filter:", "Inter", "Geist"):
        assert banned.lower() not in combined_css.lower(), f"Banned token {banned} in CSS block"


@pytest.mark.tier2
@pytest.mark.f2
def test_t2_f2_03_reference_materials_intact(repo_root: Path) -> None:
    """Verify pre-loaded GOST, Sistema, and ceramic_defect_1..4 reference assets are intact."""
    ref = repo_root / "data" / "reference_materials"
    assert (ref / "gost_28390_89_quality_table.md").stat().st_size > 100
    assert (ref / "sistema_2026_presentation_requirements.md").stat().st_size > 100
    for idx in range(1, 5):
        img = ref / "ceramic_defect_samples" / f"ceramic_defect_{idx}.jpg"
        assert img.is_file() and img.stat().st_size > 1000


@pytest.mark.tier2
@pytest.mark.f2
def test_t2_f2_04_no_forbidden_external_project_refs(repo_root: Path) -> None:
    """Verify foundation docs contain zero forbidden personal project references or team voice."""
    for filename in ("PROJECT.md", "ARCHITECTURE.md", "DESIGN_SPEC.md"):
        text = (repo_root / filename).read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_REPOS:
            assert forbidden not in text, f"Forbidden project {forbidden} found in {filename}"
        assert "мы с командой" not in text.lower()


@pytest.mark.tier2
@pytest.mark.f2
def test_t2_f2_05_author_and_legal_facts_consistent(repo_root: Path) -> None:
    """Verify author identity, GOST 28390-89, 289-FZ, and 300 ms SLA are consistent."""
    proj = (repo_root / "PROJECT.md").read_text(encoding="utf-8")
    arch = (repo_root / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "Pavlovets Alexander Ivanovich" in proj
    assert "Павловец Александр Иванович" in arch
    for doc_text in (proj, arch):
        assert "28390-89" in doc_text
        assert "289-FZ" in doc_text or "289-ФЗ" in doc_text
        assert "300 ms" in doc_text


@pytest.mark.tier2
@pytest.mark.f3
@pytest.mark.parametrize(
    "bad_img",
    [
        None,
        np.empty((0, 0, 3), dtype=np.uint8),
        np.zeros((64, 64), dtype=np.uint8),
        np.zeros((64, 64, 4), dtype=np.uint8),
    ],
)
def test_t2_f3_01_invalid_array_inputs_raise_value_error(bad_img: Any) -> None:
    """Verify preprocess_image rejects None, empty, 2D, and 4-channel arrays with ValueError."""
    from src.cv_core.preprocessing import preprocess_image

    with pytest.raises(ValueError):
        preprocess_image(bad_img, stage="glazed")


@pytest.mark.tier2
@pytest.mark.f3
def test_t2_f3_02_min_dimension_32px_boundary() -> None:
    """Verify <32px images raise ValueError while exact 32x32x3 succeeds."""
    from src.cv_core.preprocessing import preprocess_image

    for shape in ((31, 64, 3), (64, 31, 3), (31, 31, 3)):
        with pytest.raises(ValueError):
            preprocess_image(np.zeros(shape, dtype=np.uint8), stage="bisque")

    frame = preprocess_image(np.full((32, 32, 3), 180, dtype=np.uint8), stage="bisque")
    assert frame.roi_bgr.shape == (640, 640, 3)


@pytest.mark.tier2
@pytest.mark.f3
def test_t2_f3_03_oversized_image_original_bgr_capped(
    ceramic_image_factory: Any,
) -> None:
    """Verify large 2400x1800 inputs cap original_bgr max dimension to <=1600."""
    from src.cv_core.preprocessing import preprocess_image

    base = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", size=720)
    big = cv2.resize(base, (2400, 1800))
    frame = preprocess_image(big, stage="glazed")
    assert max(frame.original_bgr.shape[:2]) <= 1600
    assert frame.roi_bgr.shape == (640, 640, 3)


@pytest.mark.tier2
@pytest.mark.f3
def test_t2_f3_04_ceramic_defect_4_extreme_glare_handling(
    reference_defect_images: dict[str, Path],
) -> None:
    """Verify ceramic_defect_4.jpg macro glare & tilted borders isolate m_spec and m_interior."""
    from src.cv_core.preprocessing import preprocess_image

    img = cv2.imread(str(reference_defect_images["defect_4"]))
    assert img is not None
    frame = preprocess_image(img, stage="glazed")
    assert frame.geometry.scale_calibrated is False
    assert frame.glare_ratio > 0.35
    assert int(np.count_nonzero(frame.m_spec)) > 0
    assert int(np.count_nonzero(frame.m_interior[0, :])) == 0


@pytest.mark.tier2
@pytest.mark.f3
def test_t2_f3_05_ceramic_defect_3_nonsquare_150x200(
    reference_defect_images: dict[str, Path],
) -> None:
    """Verify non-square 150x200 ceramic_defect_3.jpg preprocesses cleanly to 640x640."""
    from src.cv_core.preprocessing import preprocess_image

    img = cv2.imread(str(reference_defect_images["defect_3"]))
    assert img is not None and img.shape[:2] == (150, 200)
    frame = preprocess_image(img, stage="glazed")
    assert frame.roi_bgr.shape == (640, 640, 3)
    assert frame.m_interior.dtype == np.uint8


@pytest.mark.tier2
@pytest.mark.f3
def test_t2_f3_06_masks_binary_and_disjoint_invariant(
    ceramic_image_factory: Any,
) -> None:
    """Verify all 5 segmentation masks are strictly {0,255} and rim/interior are disjoint."""
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True, glare=True)
    frame = preprocess_image(img, stage="glazed")
    for mask in (frame.m_item, frame.m_rim, frame.m_interior, frame.m_spec, frame.m_cobalt):
        assert set(np.unique(mask).tolist()).issubset({0, 255})
    assert int(np.count_nonzero(frame.m_rim & frame.m_interior)) == 0


@pytest.mark.tier2
@pytest.mark.f4
def test_t2_f4_01_invalid_weights_path_rejected(
    onnx_weights_factory: Callable[[str], Path],
) -> None:
    """Verify missing .onnx raises FileNotFoundError and non-.onnx raises ValueError."""
    from src.cv_core.defect_detector import DefectDetector

    with pytest.raises(FileNotFoundError):
        DefectDetector(weights_path=onnx_weights_factory("missing"))

    with pytest.raises(ValueError):
        DefectDetector(weights_path=onnx_weights_factory("non_onnx"))


@pytest.mark.tier2
@pytest.mark.f4
def test_t2_f4_02_invalid_onnx_rank_or_corrupt_rejected(
    ceramic_image_factory: Any,
    onnx_weights_factory: Callable[[str], Path],
) -> None:
    """Verify bad-rank ONNX tensor raises ValueError and corrupt ONNX fails loudly."""
    from src.cv_core.defect_detector import DefectDetector

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed")

    with pytest.raises(ValueError):
        DefectDetector(weights_path=onnx_weights_factory("bad_rank")).inspect(img)

    with pytest.raises((ValueError, RuntimeError, OSError)):
        DefectDetector(weights_path=onnx_weights_factory("corrupt")).inspect(img)


@pytest.mark.tier2
@pytest.mark.f4
def test_t2_f4_03_onnx_yolov5_layout_and_nms_dedup(
    ceramic_image_factory: Any,
    onnx_weights_factory: Callable[[str], Path],
) -> None:
    """Verify YOLOv5 (1,N,5+C) layout decoding and NMS suppression of overlapping boxes."""
    from src.cv_core.defect_detector import DefectDetector
    from src.cv_core.preprocessing import DefectClass

    weights = onnx_weights_factory("yolov5")
    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed")
    res = DefectDetector(weights_path=weights).inspect(img, stage="glazed")
    assert res.model_mode == "onnx"
    bare_boxes = [d for d in res.defects if d.defect_class == DefectClass.GLAZE_BARE]
    assert len(bare_boxes) == 1


@pytest.mark.tier2
@pytest.mark.f4
def test_t2_f4_04_shape_warp_over_1_5_pct_rejects_even_zero_surface_defects(
    ceramic_image_factory: Any,
    onnx_weights_factory: Callable[[str], Path],
) -> None:
    """Verify contour warp > 1.5% triggers SHAPE_WARP and REJECT in both CV and ONNX modes."""
    from src.cv_core.defect_detector import DefectDetector
    from src.cv_core.preprocessing import DefectClass

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", warp=True)
    for detector in (
        DefectDetector(),
        DefectDetector(weights_path=onnx_weights_factory("yolov8_clean")),
    ):
        res = detector.inspect(img, stage="bisque")
        assert res.contour_deviation_pct > 1.5
        assert res.verdict == "REJECT"
        assert any(d.defect_class == DefectClass.SHAPE_WARP for d in res.defects)


@pytest.mark.tier2
@pytest.mark.f4
def test_t2_f4_05_gost_pinhole_gt_8_and_total_gt_3_limits(
    ceramic_image_factory: Any,
) -> None:
    """Verify GOST 28390-89 >8 pinholes triggers REJECT and >3 defects never yields PASS."""
    from src.cv_core.defect_detector import DefectDetector

    det = DefectDetector()
    many_pinholes = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", pinhole_count=12)
    res_many = det.inspect(many_pinholes, stage="glazed")
    assert res_many.verdict == "REJECT"

    multi_minor = ceramic_image_factory.make_ceramic_plate_bgr(
        stage="glazed",
        pinhole_count=4,
        glaze_bare=True,
    )
    res_multi = det.inspect(multi_minor, stage="glazed")
    assert res_multi.verdict != "PASS"


@pytest.mark.tier2
@pytest.mark.f4
@pytest.mark.parametrize("defect_key", ["defect_1", "defect_2", "defect_3", "defect_4"])
def test_t2_f4_06_reference_defect_1_to_4_all_detected_under_sla(
    reference_defect_images: dict[str, Path],
    defect_key: str,
) -> None:
    """Verify all 4 real reference defect photos trigger non-PASS under 300 ms CPU SLA."""
    from src.cv_core.defect_detector import DefectDetector

    img = cv2.imread(str(reference_defect_images[defect_key]))
    assert img is not None
    stage = "bisque" if defect_key in ("defect_1", "defect_2") else "glazed"
    res = DefectDetector().inspect(img, stage=stage)  # type: ignore[arg-type]
    assert res.latency_ms < 300.0
    assert res.verdict in ("REWORK", "REJECT")
    assert res.defect_count >= 1


@pytest.mark.tier2
@pytest.mark.f4
def test_t2_f4_07_inspect_invalid_image_raises_value_error() -> None:
    """Verify DefectDetector.inspect propagates ValueError on sub-32px input."""
    from src.cv_core.defect_detector import DefectDetector

    with pytest.raises(ValueError):
        DefectDetector().inspect(np.zeros((20, 20, 3), dtype=np.uint8), stage="glazed")


@pytest.mark.tier2
@pytest.mark.f5
def test_t2_f5_01_cobalt_vs_blank_monochrome_guard(
    ceramic_image_factory: Any,
) -> None:
    """Verify cobalt reference vs blank ceramic probe forces color_score=0.0 and counterfeit."""
    from src.cv_core.fingerprint import extract_fingerprint, verify_fingerprint
    from src.cv_core.preprocessing import preprocess_image

    cobalt = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    blank = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=False)
    sig = extract_fingerprint(preprocess_image(cobalt, stage="glazed"))
    res = verify_fingerprint(sig, preprocess_image(blank, stage="glazed"))
    assert res.color_score == 0.0
    assert res.verdict == AUTH_FAKE_VERDICT


@pytest.mark.tier2
@pytest.mark.f5
def test_t2_f5_02_both_monochrome_guard_and_zero_orb(
    ceramic_image_factory: Any,
) -> None:
    """Verify two flat monochrome plates yield descriptors_b64='', color_score<=0.5, no crash."""
    from src.cv_core.fingerprint import extract_fingerprint, verify_fingerprint
    from src.cv_core.preprocessing import preprocess_image

    flat_a = np.full((640, 640, 3), 180, dtype=np.uint8)
    flat_b = np.full((640, 640, 3), 182, dtype=np.uint8)
    frame_a = preprocess_image(flat_a, stage="glazed")
    frame_b = preprocess_image(flat_b, stage="glazed")
    sig = extract_fingerprint(frame_a)
    assert sig.orb_count == 0
    assert sig.descriptors_b64 == ""

    res = verify_fingerprint(sig, frame_b)
    assert res.color_score <= 0.5
    assert res.verdict == AUTH_FAKE_VERDICT


@pytest.mark.tier2
@pytest.mark.f5
def test_t2_f5_03_dual_threshold_requires_min_matched_kp(
    ceramic_image_factory: Any,
) -> None:
    """Verify high match_percent with matched_kp < min_matched_kp still rejects."""
    from src.cv_core.fingerprint import (
        FingerprintConfig,
        extract_fingerprint,
        verify_fingerprint,
    )
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    frame = preprocess_image(img, stage="glazed")
    sig = extract_fingerprint(frame)
    cfg = FingerprintConfig(authentic_threshold_pct=10.0, min_matched_kp=999)
    res = verify_fingerprint(sig, frame, config=cfg)
    assert res.match_percent >= 10.0
    assert res.verdict == AUTH_FAKE_VERDICT


@pytest.mark.tier2
@pytest.mark.f5
def test_t2_f5_04_threshold_boundary_99_9_vs_0(ceramic_image_factory: Any) -> None:
    """Verify authentic_threshold_pct boundary rejects at 99.9 and accepts at 0.0."""
    from src.cv_core.fingerprint import (
        FingerprintConfig,
        extract_fingerprint,
        verify_fingerprint,
    )
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    probe = np.clip(img.astype(np.int16) + 8, 0, 255).astype(np.uint8)
    sig = extract_fingerprint(preprocess_image(img, stage="glazed"))
    probe_frame = preprocess_image(probe, stage="glazed")

    strict_res = verify_fingerprint(
        sig,
        probe_frame,
        config=FingerprintConfig(authentic_threshold_pct=99.9, min_matched_kp=1),
    )
    loose_res = verify_fingerprint(
        sig,
        probe_frame,
        config=FingerprintConfig(authentic_threshold_pct=0.0, min_matched_kp=1),
    )
    assert strict_res.verdict == AUTH_FAKE_VERDICT
    assert loose_res.verdict == AUTH_OK_VERDICT


@pytest.mark.tier2
@pytest.mark.f5
def test_t2_f5_05_heavy_glare_on_cobalt_bounded_scores(
    ceramic_image_factory: Any,
) -> None:
    """Verify heavy glare over cobalt artwork keeps all verification scores bounded."""
    from src.cv_core.fingerprint import extract_fingerprint, verify_fingerprint
    from src.cv_core.preprocessing import preprocess_image

    clean = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    glared = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True, glare=True)
    sig = extract_fingerprint(preprocess_image(clean, stage="glazed"))
    res = verify_fingerprint(sig, preprocess_image(glared, stage="glazed"))
    for val in (res.orb_score, res.phash_score, res.color_score):
        assert 0.0 <= val <= 1.0
    assert 0.0 <= res.match_percent <= 100.0


@pytest.mark.tier2
@pytest.mark.f5
def test_t2_f5_06_empty_reference_descriptors_safe(
    ceramic_image_factory: Any,
) -> None:
    """Verify zero-ORB reference signature does not crash BFMatcher and returns counterfeit."""
    from src.cv_core.fingerprint import (
        FingerprintSignature,
        extract_fingerprint,
        verify_fingerprint,
    )
    from src.cv_core.preprocessing import preprocess_image

    cobalt = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    frame = preprocess_image(cobalt, stage="glazed")
    real_sig = extract_fingerprint(frame)
    empty_sig = FingerprintSignature(
        phash_hex=real_sig.phash_hex,
        cobalt_hist=real_sig.cobalt_hist,
        cobalt_ratio=real_sig.cobalt_ratio,
        keypoints_xy=[],
        descriptors_b64="",
        orb_count=0,
    )
    res = verify_fingerprint(empty_sig, frame)
    assert res.orb_score == 0.0
    assert res.matched_kp == 0
    assert res.verdict == AUTH_FAKE_VERDICT


@pytest.mark.tier2
@pytest.mark.f6
@pytest.mark.parametrize(
    "bad_id",
    [
        "../evil",
        "a/b",
        "GZHEL; DROP TABLE passports;--",
        "ab",
        "X" * 65,
        "bad id!",
    ],
)
def test_t2_f6_01_item_id_path_traversal_and_sqli_rejected(
    isolated_api_server: tuple[str, dict[str, Path]],
    ceramic_image_factory: Any,
    bad_id: str,
) -> None:
    """Verify traversal, SQLi, and out-of-range item_ids are rejected with zero DB/file mutation."""
    base_url, paths = isolated_api_server
    png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        resp = client.post(
            "/api/passport/register",
            data={"item_id": bad_id},
            files={"file": ("plate.png", png, "image/png")},
        )
        assert resp.status_code in (400, 422), f"Unexpected {resp.status_code} for {bad_id}"

    assert list(paths["qr_dir"].glob("*")) == []
    assert list(paths["passport_dir"].glob("*")) == []
    with sqlite3.connect(paths["db_path"]) as conn:
        count = conn.execute("SELECT COUNT(*) FROM passports;").fetchone()[0]
        assert count == 0


@pytest.mark.tier2
@pytest.mark.f6
def test_t2_f6_02_duplicate_passport_id_rejected(
    isolated_workspace: dict[str, Path],
) -> None:
    """Verify SQLite primary key rejects duplicate passport item_id without overwriting."""
    from src.cv_core.storage import CeramStorage

    db_path = isolated_workspace["db_path"]
    CeramStorage(
        db_path=db_path,
        qr_dir=isolated_workspace["qr_dir"],
        passport_ref_dir=isolated_workspace["passport_dir"],
    )
    row = (
        "GZHEL-DUP-001",
        "SKU-1",
        "Мастер",
        "",
        "glazed",
        "CERAMSCAN:GZHEL-DUP-001",
        "qr.png",
        "ref.png",
        "{}",
        "2026-11-19T00:00:00Z",
    )
    sql = (
        "INSERT INTO passports (item_id, sku, master_name, batch_note, stage, "
        "qr_payload, qr_image_path, reference_image_path, signature_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(sql, row)
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(sql, row)


@pytest.mark.tier2
@pytest.mark.f6
def test_t2_f6_03_unknown_passport_lookup_returns_none(
    isolated_workspace: dict[str, Path],
) -> None:
    """Verify querying an unregistered passport ID returns None cleanly."""
    from src.cv_core.storage import CeramStorage

    db_path = isolated_workspace["db_path"]
    CeramStorage(
        db_path=db_path,
        qr_dir=isolated_workspace["qr_dir"],
        passport_ref_dir=isolated_workspace["passport_dir"],
    )
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM passports WHERE item_id = ?;",
            ("UNKNOWN-ITEM-999",),
        ).fetchone()
        assert row is None


@pytest.mark.tier2
@pytest.mark.f6
def test_t2_f6_04_sqlite_check_constraints_enforced(
    isolated_workspace: dict[str, Path],
) -> None:
    """Verify SQLite CHECK constraints reject invalid stage and verdict values."""
    from src.cv_core.storage import CeramStorage

    db_path = isolated_workspace["db_path"]
    CeramStorage(
        db_path=db_path,
        qr_dir=isolated_workspace["qr_dir"],
        passport_ref_dir=isolated_workspace["passport_dir"],
    )
    sql = (
        "INSERT INTO inspections (item_id, stage, shift, verdict, gost_grade, model_mode, "
        "defect_count, primary_defect, contour_deviation_pct, glare_ratio, latency_ms, "
        "defects_json, gost_rules_json, overlay_path, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
    )
    with sqlite3.connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                sql,
                ("ID-1", "raw", "S1", "PASS", "1 сорт", "deterministic", 0, None, 0.1, 0.0, 5.0, "[]", "[]", "o.png", "t"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                sql,
                ("ID-2", "glazed", "S1", "MAYBE", "1 сорт", "deterministic", 0, None, 0.1, 0.0, 5.0, "[]", "[]", "o.png", "t"),
            )


@pytest.mark.tier2
@pytest.mark.f6
def test_t2_f6_05_verification_unknown_fk_rejected(
    isolated_workspace: dict[str, Path],
) -> None:
    """Verify foreign_keys=ON rejects verification rows for nonexistent passports."""
    from src.cv_core.storage import CeramStorage

    db_path = isolated_workspace["db_path"]
    CeramStorage(
        db_path=db_path,
        qr_dir=isolated_workspace["qr_dir"],
        passport_ref_dir=isolated_workspace["passport_dir"],
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON;")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO passport_verifications (
                    item_id, match_percent, orb_score, phash_score,
                    color_score, matched_kp, verdict, latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                ("MISSING-999", 50.0, 0.5, 0.5, 0.5, 5, AUTH_FAKE_VERDICT, 10.0, "2026-11-19T00:00:00Z"),
            )


@pytest.mark.tier2
@pytest.mark.f6
def test_t2_f6_06_qr_dir_containment_preserved(
    isolated_api_server: tuple[str, dict[str, Path]],
    ceramic_image_factory: Any,
) -> None:
    """Verify boundary 5-char and 64-char valid item_ids write PNGs strictly inside target dirs."""
    base_url, paths = isolated_api_server
    png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        for edge_id in ("A_B-1", "Z" * 64):
            resp = client.post(
                "/api/passport/register",
                data={"item_id": edge_id},
                files={"file": ("plate.png", png, "image/png")},
            )
            assert resp.status_code == 200, resp.text
            qr_file = (paths["qr_dir"] / f"{edge_id}.png").resolve()
            ref_file = (paths["passport_dir"] / f"{edge_id}.png").resolve()
            assert qr_file.is_file()
            assert ref_file.is_file()
            assert qr_file.is_relative_to(paths["qr_dir"].resolve())
            assert ref_file.is_relative_to(paths["passport_dir"].resolve())


@pytest.mark.tier2
@pytest.mark.f7
def test_t2_f7_01_count_below_24_rejected(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify generate_synthetic_ceramic.py rejects --count < 24 with non-zero exit."""
    out_dir = tmp_path / "syn10"
    proc = _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(out_dir), "--count", "10"],
        cwd=repo_root,
    )
    assert proc.returncode != 0


@pytest.mark.tier2
@pytest.mark.f7
def test_t2_f7_02_exact_minimum_count_24_succeeds(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify exact minimum --count 24 succeeds with both bisque and glazed samples."""
    out_dir = tmp_path / "syn24"
    proc = _run_cli(
        venv_python,
        [
            "scripts/generate_synthetic_ceramic.py",
            "--output-dir",
            str(out_dir),
            "--count",
            "24",
            "--seed",
            "11",
        ],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["count"] == 24
    assert len(data["samples"]) == 24
    stages = {s["stage"] for s in data["samples"]}
    assert stages == {"bisque", "glazed"}


@pytest.mark.tier2
@pytest.mark.f7
def test_t2_f7_03_auth_pair_referential_integrity(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify every counterfeit sample references an existing genuine auth_pair_id."""
    out_dir = tmp_path / "syn_pairs"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(out_dir), "--count", "24", "--seed", "11"],
        cwd=repo_root,
    )
    samples = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["samples"]
    genuine_ids = {s["id"] for s in samples if not s["is_counterfeit"]}
    for s in samples:
        if s["is_counterfeit"]:
            assert s["auth_pair_id"] in genuine_ids


@pytest.mark.tier2
@pytest.mark.f7
def test_t2_f7_04_manifest_paths_no_traversal_and_nonzero_images(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify manifest image_paths contain no parent traversal and exceed 1 KB on disk."""
    out_dir = tmp_path / "syn_paths"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(out_dir), "--seed", "42"],
        cwd=repo_root,
    )
    samples = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))["samples"]
    for s in samples:
        raw = s["image_path"]
        assert ".." not in Path(raw).parts
        p = Path(raw)
        resolved = (p if p.is_absolute() else (out_dir / p if (out_dir / p).exists() else repo_root / p)).resolve()
        assert resolved.is_relative_to(out_dir.resolve())
        assert resolved.stat().st_size > 1024


@pytest.mark.tier2
@pytest.mark.f7
def test_t2_f7_05_existing_output_dir_overwrite_clean(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify re-running generator into existing output dir cleanly overwrites manifest count."""
    out_dir = tmp_path / "syn_overwrite"
    proc1 = _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(out_dir), "--count", "26"],
        cwd=repo_root,
    )
    assert proc1.returncode == 0
    proc2 = _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(out_dir), "--count", "24"],
        cwd=repo_root,
    )
    assert proc2.returncode == 0
    data = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["count"] == 24
    assert len(data["samples"]) == 24


@pytest.mark.tier2
@pytest.mark.f8
def test_t2_f8_01_missing_dataset_fails_cleanly(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify missing --dataset exits non-zero without writing output JSON."""
    out_json = tmp_path / "should_not_exist.json"
    proc = _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(tmp_path / "nope.json"),
            "--output-json",
            str(out_json),
        ],
        cwd=repo_root,
    )
    assert proc.returncode != 0
    assert not out_json.exists()


@pytest.mark.tier2
@pytest.mark.f8
def test_t2_f8_02_empty_manifest_samples_rejected(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify empty samples list exits non-zero cleanly without ZeroDivisionError."""
    bad_manifest = tmp_path / "empty_manifest.json"
    bad_manifest.write_text(json.dumps({"stand_label": "test", "samples": []}), encoding="utf-8")
    proc = _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(bad_manifest),
            "--output-json",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
    )
    assert proc.returncode != 0
    assert "ZeroDivisionError" not in proc.stderr


@pytest.mark.tier2
@pytest.mark.f8
def test_t2_f8_03_invalid_weights_flag_fails_fast(
    repo_root: Path,
    venv_python: Path,
    onnx_weights_factory: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """Verify run_benchmark fails fast on invalid or missing --weights."""
    syn_dir = tmp_path / "syn"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir)],
        cwd=repo_root,
    )
    for mode in ("non_onnx", "missing"):
        proc = _run_cli(
            venv_python,
            [
                "scripts/run_benchmark.py",
                "--dataset",
                str(syn_dir / "manifest.json"),
                "--weights",
                str(onnx_weights_factory(mode)),
                "--output-json",
                str(tmp_path / f"out_{mode}.json"),
            ],
            cwd=repo_root,
        )
        assert proc.returncode != 0


@pytest.mark.tier2
@pytest.mark.f8
def test_t2_f8_04_zero_auth_pairs_no_zero_division(
    repo_root: Path,
    venv_python: Path,
    ceramic_image_factory: Any,
    tmp_path: Path,
) -> None:
    """Verify benchmark with 0 counterfeit pairs avoids ZeroDivisionError and sets pair_count=0."""
    img_dir = tmp_path / "imgs"
    samples = []
    for idx in range(4):
        p = _copy_image_to(
            ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", crack=(idx % 2 == 1)),
            img_dir / f"s_{idx}.png",
        )
        samples.append(
            {
                "id": f"S-{idx}",
                "image_path": str(p),
                "stage": "bisque",
                "defect_class": "crack" if idx % 2 == 1 else "pass",
                "expected_verdict": "REJECT" if idx % 2 == 1 else "PASS",
                "auth_pair_id": None,
                "is_counterfeit": False,
            }
        )
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"stand_label": "zero-auth", "samples": samples}), encoding="utf-8")

    out_json = tmp_path / "bench.json"
    proc = _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(man),
            "--output-json",
            str(out_json),
            "--output-md",
            str(tmp_path / "bench.md"),
            "--overlays-dir",
            str(tmp_path / "overlays"),
        ],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    auth = json.loads(out_json.read_text(encoding="utf-8"))["auth_benchmark"]
    assert auth["pair_count"] == 0
    assert auth["auth_accuracy"] == 0.0


@pytest.mark.tier2
@pytest.mark.f8
def test_t2_f8_05_corrupt_image_in_manifest_rejected(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify 0-byte corrupt image referenced by manifest causes non-zero failure."""
    bad_img = tmp_path / "corrupt.png"
    bad_img.write_bytes(b"")
    man = tmp_path / "bad_manifest.json"
    man.write_text(
        json.dumps(
            {
                "stand_label": "bad",
                "samples": [
                    {
                        "id": "BAD-01",
                        "image_path": str(bad_img),
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
    proc = _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(man),
            "--output-json",
            str(tmp_path / "out.json"),
        ],
        cwd=repo_root,
    )
    assert proc.returncode != 0


@pytest.mark.tier2
@pytest.mark.f9
def test_t2_f9_01_zero_real_photos_fallback_clean(
    isolated_repo_workspace: Path,
    venv_python: Path,
    real_photos_dir_factory: Callable[[bool], Path],
) -> None:
    """Verify empty real_photos directory falls back cleanly to synthetic stand on --rebuild-all."""
    empty_dir = real_photos_dir_factory(False)
    proc = _run_cli(
        venv_python,
        ["scripts/ingest_real_photos.py", "--input", str(empty_dir), "--rebuild-all"],
        cwd=isolated_repo_workspace,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    combined_out = (proc.stdout + proc.stderr).lower()
    assert "fallback" in combined_out or "synthetic" in combined_out or "синтетич" in combined_out
    bench_json = isolated_repo_workspace / "reports" / "benchmark_results.json"
    assert bench_json.is_file()
    report = json.loads(bench_json.read_text(encoding="utf-8"))
    assert report["sample_count"] >= 24


@pytest.mark.tier2
@pytest.mark.f9
def test_t2_f9_02_non_image_files_ignored(
    isolated_repo_workspace: Path,
    venv_python: Path,
    real_photos_dir_factory: Callable[[bool], Path],
    ceramic_image_factory: Any,
) -> None:
    """Verify .gitkeep and .txt files inside real_photos are skipped cleanly without imread error."""
    photos_dir = real_photos_dir_factory(False)
    (photos_dir / "pass" / ".gitkeep").write_text("", encoding="utf-8")
    (photos_dir / "pass" / "notes.txt").write_text("ignore me", encoding="utf-8")
    _copy_image_to(
        ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True),
        photos_dir / "pass" / "real_01.jpg",
    )

    proc = _run_cli(
        venv_python,
        ["scripts/ingest_real_photos.py", "--input", str(photos_dir), "--rebuild-all"],
        cwd=isolated_repo_workspace,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(
        (isolated_repo_workspace / "reports" / "benchmark_results.json").read_text(encoding="utf-8")
    )
    assert report["sample_count"] == 1


@pytest.mark.tier2
@pytest.mark.f9
def test_t2_f9_03_missing_input_dir_fails_cleanly(
    isolated_repo_workspace: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify missing --input directory fails cleanly with non-zero exit."""
    proc = _run_cli(
        venv_python,
        ["scripts/ingest_real_photos.py", "--input", str(tmp_path / "does_not_exist")],
        cwd=isolated_repo_workspace,
    )
    assert proc.returncode != 0


@pytest.mark.tier2
@pytest.mark.f9
def test_t2_f9_04_invalid_weights_on_ingest_fails_fast(
    isolated_repo_workspace: Path,
    venv_python: Path,
    real_photos_dir_factory: Callable[[bool], Path],
    onnx_weights_factory: Callable[[str], Path],
) -> None:
    """Verify ingest_real_photos fails fast on invalid --weights without silent fallback."""
    photos_dir = real_photos_dir_factory(True)
    proc = _run_cli(
        venv_python,
        [
            "scripts/ingest_real_photos.py",
            "--input",
            str(photos_dir),
            "--weights",
            str(onnx_weights_factory("non_onnx")),
            "--rebuild-all",
        ],
        cwd=isolated_repo_workspace,
    )
    assert proc.returncode != 0


@pytest.mark.tier2
@pytest.mark.f9
def test_t2_f9_05_fake_without_pass_no_crash(
    isolated_repo_workspace: Path,
    venv_python: Path,
    real_photos_dir_factory: Callable[[bool], Path],
    ceramic_image_factory: Any,
) -> None:
    """Verify fake/ image without any pass/ image runs cleanly with pair_count == 0."""
    photos_dir = real_photos_dir_factory(False)
    _copy_image_to(
        ceramic_image_factory.make_counterfeit_plate_bgr(),
        photos_dir / "fake" / "fake_01.png",
    )
    proc = _run_cli(
        venv_python,
        ["scripts/ingest_real_photos.py", "--input", str(photos_dir), "--rebuild-all"],
        cwd=isolated_repo_workspace,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(
        (isolated_repo_workspace / "reports" / "benchmark_results.json").read_text(encoding="utf-8")
    )
    assert report["auth_benchmark"]["pair_count"] == 0


# ===========================================================================
# TIER 3: Cross-Feature Pairwise Combinations (12 Tests)
# ===========================================================================


@pytest.mark.tier3
@pytest.mark.f3
@pytest.mark.f4
def test_t3_01_f3_f4_preprocess_to_detector_consistency(
    ceramic_image_factory: Any,
) -> None:
    """Verify preprocess_image geometry and ratios match DefectDetector.inspect outputs."""
    from src.cv_core.defect_detector import DefectDetector
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True, glare=True)
    frame = preprocess_image(img, stage="glazed")
    res = DefectDetector().inspect(img, stage="glazed")

    assert res.contour_deviation_pct == pytest.approx(frame.geometry.contour_deviation_pct, abs=1e-3)
    assert res.glare_ratio == pytest.approx(frame.glare_ratio, abs=1e-3)
    assert res.cobalt_ratio == pytest.approx(frame.cobalt_ratio, abs=1e-3)
    assert res.scale_calibrated is frame.geometry.scale_calibrated


@pytest.mark.tier3
@pytest.mark.f3
@pytest.mark.f5
def test_t3_02_f3_f5_preprocess_to_fingerprint_cobalt_consistency(
    ceramic_image_factory: Any,
) -> None:
    """Verify cobalt_ratio flows consistently from preprocessing into fingerprint signature."""
    from src.cv_core.fingerprint import extract_fingerprint, verify_fingerprint
    from src.cv_core.preprocessing import preprocess_image

    img = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True)
    frame = preprocess_image(img, stage="glazed")
    sig = extract_fingerprint(frame)
    assert sig.cobalt_ratio == pytest.approx(frame.cobalt_ratio, abs=1e-4)
    res = verify_fingerprint(sig, frame)
    assert res.verdict == AUTH_OK_VERDICT


@pytest.mark.tier3
@pytest.mark.f4
@pytest.mark.f6
def test_t3_03_f4_f6_inspect_to_sqlite_wal_persistence(
    isolated_api_server: tuple[str, dict[str, Path]],
    ceramic_image_factory: Any,
) -> None:
    """Verify bisque crack and glazed pass inspections persist accurately to SQLite WAL."""
    base_url, paths = isolated_api_server
    crack_png = ceramic_image_factory.make_bisque_plate_png(with_crack=True)
    pass_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        r1 = client.post(
            "/api/inspect",
            data={"stage": "bisque", "shift": "S1"},
            files={"file": ("crack.png", crack_png, "image/png")},
        )
        r2 = client.post(
            "/api/inspect",
            data={"stage": "glazed", "shift": "S2"},
            files={"file": ("pass.png", pass_png, "image/png")},
        )
        assert r1.status_code == 200 and r2.status_code == 200

    with sqlite3.connect(paths["db_path"]) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM inspections ORDER BY id ASC;").fetchall()
        assert len(rows) == 2
        assert rows[0]["verdict"] == "REJECT"
        assert rows[0]["primary_defect"] == "crack"
        assert rows[1]["verdict"] == "PASS"


@pytest.mark.tier3
@pytest.mark.f5
@pytest.mark.f6
def test_t3_04_f5_f6_fingerprint_to_passport_qr_roundtrip(
    isolated_api_server: tuple[str, dict[str, Path]],
    ceramic_image_factory: Any,
) -> None:
    """Verify register -> QR PNG decode -> DB signature -> verify roundtrip for genuine & fake."""
    base_url, paths = isolated_api_server
    genuine_png = ceramic_image_factory.make_cobalt_plate_png(variant=0)
    fake_png = ceramic_image_factory.make_cobalt_plate_png(variant=1)

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        reg = client.post(
            "/api/passport/register",
            data={"item_id": "GZHEL-QR-RT-01"},
            files={"file": ("gen.png", genuine_png, "image/png")},
        )
        assert reg.status_code == 200

    qr_img = cv2.imread(str(paths["qr_dir"] / "GZHEL-QR-RT-01.png"))
    decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(qr_img)
    assert decoded.startswith("CERAMSCAN:")
    decoded_id = decoded.split(":", 1)[1]

    with sqlite3.connect(paths["db_path"]) as conn:
        row = conn.execute(
            "SELECT signature_json FROM passports WHERE item_id = ?;",
            (decoded_id,),
        ).fetchone()
        assert row is not None and "phash_hex" in json.loads(row[0])

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        ok_resp = client.post(
            f"/api/passport/verify/{decoded_id}",
            files={"file": ("gen.png", genuine_png, "image/png")},
        )
        bad_resp = client.post(
            f"/api/passport/verify/{decoded_id}",
            files={"file": ("fake.png", fake_png, "image/png")},
        )
        assert ok_resp.status_code == 200 and bad_resp.status_code == 200
        assert ok_resp.json()["verdict"] == AUTH_OK_VERDICT
        assert bad_resp.json()["verdict"] == AUTH_FAKE_VERDICT


@pytest.mark.tier3
@pytest.mark.f7
@pytest.mark.f8
def test_t3_05_f7_f8_synthetic_stand_to_benchmark(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify full F7 synthetic generation -> F8 benchmark execution pipeline."""
    syn_dir = tmp_path / "syn"
    overlays_dir = tmp_path / "overlays"
    out_json = tmp_path / "bench.json"
    assert _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--count", "26", "--seed", "42"],
        cwd=repo_root,
    ).returncode == 0
    assert _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(syn_dir / "manifest.json"),
            "--output-json",
            str(out_json),
            "--output-md",
            str(tmp_path / "bench.md"),
            "--overlays-dir",
            str(overlays_dir),
        ],
        cwd=repo_root,
    ).returncode == 0

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["sample_count"] == 26
    assert report["avg_latency_ms"] < 300.0
    assert len(report["overlay_files"]) == 26
    assert report["auth_benchmark"]["pair_count"] >= 4


@pytest.mark.tier3
@pytest.mark.f4
@pytest.mark.f8
def test_t3_06_f4_f8_benchmark_with_onnx_weights(
    repo_root: Path,
    venv_python: Path,
    onnx_weights_factory: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """Verify run_benchmark.py --weights switches model_mode to onnx end-to-end."""
    syn_dir = tmp_path / "syn"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--seed", "42"],
        cwd=repo_root,
    )
    out_json = tmp_path / "bench_onnx.json"
    proc = _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(syn_dir / "manifest.json"),
            "--weights",
            str(onnx_weights_factory("yolov8")),
            "--output-json",
            str(out_json),
            "--output-md",
            str(tmp_path / "bench_onnx.md"),
            "--overlays-dir",
            str(tmp_path / "overlays_onnx"),
        ],
        cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["model_mode"] == "onnx"
    assert report["sample_count"] == 26
    assert len(report["overlay_files"]) == 26


@pytest.mark.tier3
@pytest.mark.f7
@pytest.mark.f3
@pytest.mark.f4
def test_t3_07_f7_f3_f4_synthetic_samples_through_cv_core(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify canonical synthetic archetypes match DefectDetector expected verdicts."""
    from src.cv_core.defect_detector import DefectDetector

    syn_dir = tmp_path / "syn"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--seed", "42"],
        cwd=repo_root,
    )
    samples = json.loads((syn_dir / "manifest.json").read_text(encoding="utf-8"))["samples"]
    detector = DefectDetector()

    picked: dict[tuple[str, str], dict[str, Any]] = {}
    for s in samples:
        if s["is_counterfeit"]:
            continue
        key = (s["stage"], s["defect_class"])
        if key in {("bisque", "pass"), ("bisque", "crack"), ("glazed", "pass"), ("glazed", "crack")}:
            picked.setdefault(key, s)

    assert len(picked) == 4
    for (stage, _), sample in picked.items():
        p = Path(sample["image_path"])
        resolved = p if p.is_absolute() else (syn_dir / p if (syn_dir / p).exists() else repo_root / p)
        img = cv2.imread(str(resolved))
        assert img is not None
        res = detector.inspect(img, stage=stage)  # type: ignore[arg-type]
        assert res.verdict == sample["expected_verdict"]


@pytest.mark.tier3
@pytest.mark.f7
@pytest.mark.f5
def test_t3_08_f7_f5_synthetic_counterfeit_pairs_discriminated(
    repo_root: Path,
    venv_python: Path,
    tmp_path: Path,
) -> None:
    """Verify genuine vs counterfeit synthetic pairs separate cleanly via fingerprint."""
    from src.cv_core.fingerprint import extract_fingerprint, verify_fingerprint
    from src.cv_core.preprocessing import preprocess_image

    syn_dir = tmp_path / "syn"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--seed", "42"],
        cwd=repo_root,
    )
    samples = json.loads((syn_dir / "manifest.json").read_text(encoding="utf-8"))["samples"]
    by_id = {s["id"]: s for s in samples}
    pairs = [s for s in samples if s["is_counterfeit"] and s["auth_pair_id"]]
    assert len(pairs) >= 4

    for fake_sample in pairs:
        gen_sample = by_id[fake_sample["auth_pair_id"]]
        gen_path = Path(gen_sample["image_path"])
        fake_path = Path(fake_sample["image_path"])
        gen_res_path = gen_path if gen_path.is_absolute() else (syn_dir / gen_path if (syn_dir / gen_path).exists() else repo_root / gen_path)
        fake_res_path = fake_path if fake_path.is_absolute() else (syn_dir / fake_path if (syn_dir / fake_path).exists() else repo_root / fake_path)

        gen_frame = preprocess_image(cv2.imread(str(gen_res_path)), stage="glazed")
        fake_frame = preprocess_image(cv2.imread(str(fake_res_path)), stage="glazed")
        sig = extract_fingerprint(gen_frame)
        self_res = verify_fingerprint(sig, gen_frame)
        fake_res = verify_fingerprint(sig, fake_frame)
        assert self_res.match_percent > fake_res.match_percent


@pytest.mark.tier3
@pytest.mark.f9
@pytest.mark.f8
def test_t3_09_f9_f8_ingest_real_photos_updates_benchmark(
    isolated_repo_workspace: Path,
    venv_python: Path,
    ceramic_image_factory: Any,
    tmp_path: Path,
) -> None:
    """Verify ingesting 6 real photos updates benchmark sample_count and overlay count to 6."""
    photos = tmp_path / "six_photos"
    _copy_image_to(ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True), photos / "pass" / "g1.jpg")
    _copy_image_to(ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque"), photos / "pass" / "bisque_01.jpg")
    _copy_image_to(ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", chip=True), photos / "chip" / "c1.jpg")
    _copy_image_to(ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", crack=True), photos / "crack" / "k1.jpg")
    _copy_image_to(ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", glaze_bare=True), photos / "glaze" / "gl1.jpg")
    _copy_image_to(ceramic_image_factory.make_counterfeit_plate_bgr(), photos / "fake" / "f1.jpg")

    proc = _run_cli(
        venv_python,
        ["scripts/ingest_real_photos.py", "--input", str(photos), "--rebuild-all"],
        cwd=isolated_repo_workspace,
        timeout=90,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(
        (isolated_repo_workspace / "reports" / "benchmark_results.json").read_text(encoding="utf-8")
    )
    assert report["sample_count"] == 6
    assert len(report["overlay_files"]) == 6


@pytest.mark.tier3
@pytest.mark.f9
@pytest.mark.f4
def test_t3_10_f9_f4_ingest_yolo_labels_valid_for_detector_classes(
    isolated_repo_workspace: Path,
    venv_python: Path,
    real_photos_dir_factory: Callable[[bool], Path],
    tmp_path: Path,
) -> None:
    """Verify YOLO data.yaml class names align with DefectClass enum values."""
    from src.cv_core.preprocessing import DefectClass

    photos = real_photos_dir_factory(True)
    yolo_dir = tmp_path / "yolo_align"
    proc = _run_cli(
        venv_python,
        ["scripts/ingest_real_photos.py", "--input", str(photos), "--export-yolo", str(yolo_dir)],
        cwd=isolated_repo_workspace,
    )
    assert proc.returncode == 0, proc.stderr
    yaml_text = (yolo_dir / "data.yaml").read_text(encoding="utf-8")
    for cls in DefectClass:
        assert cls.value in yaml_text


@pytest.mark.tier3
@pytest.mark.f1
@pytest.mark.f9
def test_t3_11_f1_f9_gitignore_protects_real_photos_during_ingest(
    repo_root: Path,
    isolated_repo_workspace: Path,
    venv_python: Path,
    ceramic_image_factory: Any,
) -> None:
    """Verify gitignore keeps dropped real photos untracked during ingest runs."""
    shutil.copy2(repo_root / ".gitignore", isolated_repo_workspace / ".gitignore")
    subprocess.run(["git", "init", "-q"], cwd=str(isolated_repo_workspace), check=True)
    _copy_image_to(
        ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True),
        isolated_repo_workspace / "data" / "real_photos" / "pass" / "factory_01.jpg",
    )
    proc = _run_cli(
        venv_python,
        ["scripts/ingest_real_photos.py", "--input", "data/real_photos"],
        cwd=isolated_repo_workspace,
    )
    assert proc.returncode == 0, proc.stderr
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(isolated_repo_workspace),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "factory_01.jpg" not in status.stdout


@pytest.mark.tier3
@pytest.mark.f4
@pytest.mark.f8
def test_t3_12_f4_f8_deterministic_vs_onnx_parity_on_clean_item(
    repo_root: Path,
    venv_python: Path,
    onnx_weights_factory: Callable[[str], Path],
    tmp_path: Path,
) -> None:
    """Verify clean ONNX weights retain SHAPE_WARP detection while surface defects drop."""
    syn_dir = tmp_path / "syn"
    _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--seed", "42"],
        cwd=repo_root,
    )
    det_json = tmp_path / "det.json"
    onnx_json = tmp_path / "onnx_clean.json"
    assert _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(syn_dir / "manifest.json"),
            "--output-json",
            str(det_json),
            "--output-md",
            str(tmp_path / "det.md"),
            "--overlays-dir",
            str(tmp_path / "ov_det"),
        ],
        cwd=repo_root,
    ).returncode == 0
    assert _run_cli(
        venv_python,
        [
            "scripts/run_benchmark.py",
            "--dataset",
            str(syn_dir / "manifest.json"),
            "--weights",
            str(onnx_weights_factory("yolov8_clean")),
            "--output-json",
            str(onnx_json),
            "--output-md",
            str(tmp_path / "onnx.md"),
            "--overlays-dir",
            str(tmp_path / "ov_onnx"),
        ],
        cwd=repo_root,
    ).returncode == 0

    det_report = json.loads(det_json.read_text(encoding="utf-8"))
    onnx_report = json.loads(onnx_json.read_text(encoding="utf-8"))
    assert det_report["verdict_accuracy"] > onnx_report["verdict_accuracy"]
    shape_warp_metrics = onnx_report["per_class"].get("shape_warp", {})
    assert shape_warp_metrics.get("recall", 0.0) > 0.0


# ===========================================================================
# TIER 4: Real-World Factory & 289-FZ Workflows (5 Tests)
# ===========================================================================


@pytest.mark.tier4
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f6
def test_t4_01_full_factory_shift_qc_batch(
    isolated_api_server: tuple[str, dict[str, Path]],
    ceramic_image_factory: Any,
) -> None:
    """Simulate 10-item S1 bisque + S2 glazed shift batch verifying <300ms SLA and DB analytics."""
    base_url, paths = isolated_api_server
    batch = [
        ("S1", "bisque", ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", seed=1)),
        ("S1", "bisque", ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", seed=2)),
        ("S1", "bisque", ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", crack=True, seed=3)),
        ("S1", "bisque", ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", chip=True, seed=4)),
        ("S1", "bisque", ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", warp=True, seed=5)),
        ("S2", "glazed", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True, seed=6)),
        ("S2", "glazed", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", crack=True, seed=7)),
        ("S2", "glazed", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True, glare=True, seed=8)),
        ("S2", "glazed", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", glaze_bare=True, seed=9)),
        ("S2", "glazed", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", pinhole_count=12, seed=10)),
    ]

    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        for idx, (shift, stage, bgr) in enumerate(batch):
            png = ceramic_image_factory.encode_png(bgr)
            resp = client.post(
                "/api/inspect",
                data={"stage": stage, "shift": shift, "item_id": f"SHIFT-{idx:02d}"},
                files={"file": (f"item_{idx}.png", png, "image/png")},
            )
            assert resp.status_code == 200, resp.text
            payload = resp.json()
            assert payload["metrics"]["latency_ms"] < 300.0
            assert base64.b64decode(payload["overlay_base64"]).startswith(b"\x89PNG\r\n\x1a\n")

        analytics = client.get("/api/analytics?limit=25").json()
        assert analytics["summary"]["total_inspections"] == 10
        assert analytics["summary"]["reject_count"] >= 4

    with sqlite3.connect(paths["db_path"]) as conn:
        s1_rejects = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE shift = 'S1' AND verdict = 'REJECT';"
        ).fetchone()[0]
        assert s1_rejects >= 2


@pytest.mark.tier4
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f6
def test_t4_02_reference_ceramic_defect_1_to_4_triage_batch(
    isolated_api_server: tuple[str, dict[str, Path]],
    reference_defect_images: dict[str, Path],
) -> None:
    """Inspect all 4 reference ceramic defect photos end-to-end and persist to SQLite."""
    base_url, paths = isolated_api_server
    cases = (
        ("defect_1", "bisque"),
        ("defect_2", "bisque"),
        ("defect_3", "glazed"),
        ("defect_4", "glazed"),
    )
    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        for key, stage in cases:
            img_bytes = reference_defect_images[key].read_bytes()
            resp = client.post(
                "/api/inspect",
                data={"stage": stage, "item_id": f"REF-{key}"},
                files={"file": (f"{key}.jpg", img_bytes, "image/jpeg")},
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            assert body["metrics"]["latency_ms"] < 300.0
            assert body["verdict"] in ("REWORK", "REJECT")
            assert base64.b64decode(body["overlay_base64"]).startswith(b"\x89PNG\r\n\x1a\n")

    with sqlite3.connect(paths["db_path"]) as conn:
        non_pass = conn.execute(
            "SELECT COUNT(*) FROM inspections WHERE verdict != 'PASS';"
        ).fetchone()[0]
        assert non_pass == 4


@pytest.mark.tier4
@pytest.mark.f3
@pytest.mark.f4
@pytest.mark.f5
@pytest.mark.f6
@pytest.mark.f7
@pytest.mark.f8
def test_t4_03_synthetic_stand_benchmark_and_auth_e2e(
    repo_root: Path,
    venv_python: Path,
    isolated_api_server: tuple[str, dict[str, Path]],
    tmp_path: Path,
) -> None:
    """Run synthetic stand -> benchmark -> live passport QR registration & verification."""
    syn_dir = tmp_path / "syn"
    overlays_dir = tmp_path / "overlays"
    out_json = tmp_path / "bench.json"
    out_md = tmp_path / "bench.md"
    assert _run_cli(
        venv_python,
        ["scripts/generate_synthetic_ceramic.py", "--output-dir", str(syn_dir), "--count", "26", "--seed", "42"],
        cwd=repo_root,
    ).returncode == 0
    assert _run_cli(
        venv_python,
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
        cwd=repo_root,
    ).returncode == 0

    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["avg_latency_ms"] < 300.0
    assert report["auth_benchmark"]["genuine_mean_match"] > report["auth_benchmark"]["fake_mean_match"]

    samples = json.loads((syn_dir / "manifest.json").read_text(encoding="utf-8"))["samples"]
    fake_sample = next(s for s in samples if s["is_counterfeit"] and s["auth_pair_id"])
    gen_sample = next(s for s in samples if s["id"] == fake_sample["auth_pair_id"])

    def _resolve(raw: str) -> Path:
        p = Path(raw)
        return p if p.is_absolute() else (syn_dir / p if (syn_dir / p).exists() else repo_root / p)

    gen_bytes = _resolve(gen_sample["image_path"]).read_bytes()
    fake_bytes = _resolve(fake_sample["image_path"]).read_bytes()

    base_url, paths = isolated_api_server
    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        reg = client.post(
            "/api/passport/register",
            data={"item_id": "SYN-PASS-01"},
            files={"file": ("gen.png", gen_bytes, "image/png")},
        )
        assert reg.status_code == 200
        qr_img = cv2.imread(str(paths["qr_dir"] / "SYN-PASS-01.png"))
        decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(qr_img)
        assert decoded == "CERAMSCAN:SYN-PASS-01"

        v_ok = client.post(
            "/api/passport/verify/SYN-PASS-01",
            files={"file": ("gen.png", gen_bytes, "image/png")},
        )
        v_fake = client.post(
            "/api/passport/verify/SYN-PASS-01",
            files={"file": ("fake.png", fake_bytes, "image/png")},
        )
        assert v_ok.status_code == 200 and v_fake.status_code == 200
        assert v_ok.json()["verdict"] == AUTH_OK_VERDICT
        assert v_fake.json()["verdict"] == AUTH_FAKE_VERDICT


@pytest.mark.tier4
@pytest.mark.f7
@pytest.mark.f8
@pytest.mark.f9
def test_t4_04_factory_dropin_ingest_rebuild_all_isolated(
    isolated_repo_workspace: Path,
    venv_python: Path,
    ceramic_image_factory: Any,
    tmp_path: Path,
) -> None:
    """Drop 12 factory photos in isolated repo, export YOLO, and rebuild reports (+ PPTX if built)."""
    drop_dir = isolated_repo_workspace / "data" / "real_photos"
    yolo_dir = tmp_path / "yolo_12"
    photos = [
        (drop_dir / "pass" / "bisque_01.jpg", ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", seed=201)),
        (drop_dir / "pass" / "util_02.jpg", ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", seed=202)),
        (drop_dir / "pass" / "glazed_03.jpg", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True, seed=203)),
        (drop_dir / "pass" / "glazed_04.jpg", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True, seed=204)),
        (drop_dir / "chip" / "chip_01.jpg", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", chip=True, seed=205)),
        (drop_dir / "chip" / "chip_02.jpg", ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", chip=True, seed=206)),
        (drop_dir / "crack" / "crack_01.jpg", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", crack=True, seed=207)),
        (drop_dir / "crack" / "crack_02.jpg", ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", crack=True, seed=208)),
        (drop_dir / "glaze" / "glaze_01.jpg", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", glaze_bare=True, seed=209)),
        (drop_dir / "glaze" / "glaze_02.jpg", ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", glaze_bare=True, seed=210)),
        (drop_dir / "fake" / "fake_01.jpg", ceramic_image_factory.make_counterfeit_plate_bgr(seed=211)),
        (drop_dir / "fake" / "fake_02.jpg", ceramic_image_factory.make_counterfeit_plate_bgr(seed=212)),
    ]
    for path, bgr in photos:
        _copy_image_to(bgr, path)

    proc = _run_cli(
        venv_python,
        [
            "scripts/ingest_real_photos.py",
            "--input",
            str(drop_dir),
            "--export-yolo",
            str(yolo_dir),
            "--rebuild-all",
        ],
        cwd=isolated_repo_workspace,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert (yolo_dir / "data.yaml").is_file()
    images = list((yolo_dir / "images").glob("*"))
    labels = list((yolo_dir / "labels").glob("*.txt"))
    assert len(images) >= 10
    assert len(labels) >= 10
    assert len(labels) == len(images)

    report = json.loads(
        (isolated_repo_workspace / "reports" / "benchmark_results.json").read_text(encoding="utf-8")
    )
    assert report["sample_count"] == 12
    assert report["avg_latency_ms"] < 300.0
    assert len(report["overlay_files"]) == 12

    pptx_builder = isolated_repo_workspace / "grant_case" / "build_presentation.py"
    pptx_file = isolated_repo_workspace / "grant_case" / "CeramScan_Sistema_2026.pptx"
    if pptx_builder.is_file() and pptx_file.is_file():
        from pptx import Presentation

        prs = Presentation(str(pptx_file))
        assert len(prs.slides) == 10


@pytest.mark.tier4
@pytest.mark.f3
@pytest.mark.f5
@pytest.mark.f6
def test_t4_05_289fz_marketplace_passport_lifecycle(
    isolated_api_server: tuple[str, dict[str, Path]],
    ceramic_image_factory: Any,
) -> None:
    """Verify full 289-FZ passport lifecycle across 3 SKUs, QR decode, and verification audit."""
    base_url, paths = isolated_api_server
    genuine_bgr = ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True, seed=42)
    genuine_png = ceramic_image_factory.encode_png(genuine_bgr)
    fake_png = ceramic_image_factory.encode_png(ceramic_image_factory.make_counterfeit_plate_bgr(seed=77))
    blank_png = ceramic_image_factory.encode_png(
        ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=False, seed=42)
    )

    h, w = genuine_bgr.shape[:2]
    rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 4.0, 1.0)
    shifted = np.clip(
        cv2.warpAffine(genuine_bgr, rot, (w, h), borderValue=(28, 28, 30)).astype(np.int16) + 8,
        0,
        255,
    ).astype(np.uint8)
    shifted_png = ceramic_image_factory.encode_png(shifted)

    sku_ids = ("GZHEL-SKU-A1", "GZHEL-SKU-B2", "GZHEL-SKU-C3")
    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        for item_id in sku_ids:
            reg = client.post(
                "/api/passport/register",
                data={"item_id": item_id, "sku": f"ART-{item_id}", "master_name": "Мастер"},
                files={"file": (f"{item_id}.png", genuine_png, "image/png")},
            )
            assert reg.status_code == 200, reg.text
            qr_img = cv2.imread(str(paths["qr_dir"] / f"{item_id}.png"))
            decoded, _, _ = cv2.QRCodeDetector().detectAndDecode(qr_img)
            assert decoded == f"CERAMSCAN:{item_id}"

        res_gen = client.post(
            "/api/passport/verify/GZHEL-SKU-A1",
            files={"file": ("probe_gen.png", shifted_png, "image/png")},
        )
        res_fake = client.post(
            "/api/passport/verify/GZHEL-SKU-B2",
            files={"file": ("probe_fake.png", fake_png, "image/png")},
        )
        res_blank = client.post(
            "/api/passport/verify/GZHEL-SKU-C3",
            files={"file": ("probe_blank.png", blank_png, "image/png")},
        )
        assert res_gen.status_code == 200
        assert res_fake.status_code == 200
        assert res_blank.status_code == 200
        assert res_gen.json()["verdict"] == AUTH_OK_VERDICT
        assert res_fake.json()["verdict"] == AUTH_FAKE_VERDICT
        assert res_blank.json()["color_score"] == 0.0
        assert res_blank.json()["verdict"] == AUTH_FAKE_VERDICT

    with sqlite3.connect(paths["db_path"]) as conn:
        ver_count = conn.execute("SELECT COUNT(*) FROM passport_verifications;").fetchone()[0]
        assert ver_count == 3
