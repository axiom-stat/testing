"""Integration tests for CeramScan FastAPI endpoints, static mounts, and validation guards."""

from __future__ import annotations

import base64
import io
import re
import struct
import zlib
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, UnidentifiedImageError

from src.api import _DECODE_ERRORS, MAX_UPLOAD_BYTES, PROJECT_ROOT, create_app
from src.cv_core import VERDICT_AUTHENTIC, VERDICT_COUNTERFEIT

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def _make_plate_bgr(
    *,
    stage: str = "glazed",
    cobalt: bool = False,
    crack: bool = False,
    counterfeit: bool = False,
    size: int = 640,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), (28, 28, 30), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    radius = int(size * 0.36)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), radius, 255, -1)

    base = (176, 178, 182) if stage == "bisque" else (218, 220, 224)
    plate = np.full((size, size, 3), base, dtype=np.uint8)
    noise = rng.integers(-2, 3, size=(size, size, 3))
    plate = np.clip(plate.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if cobalt and not counterfeit:
        dark = (182, 52, 20)
        mid = (205, 86, 44)
        cv2.ellipse(plate, (cx, cy), (int(radius * 0.72), int(radius * 0.72)), 0, 0, 360, mid, 3, cv2.LINE_AA)
        for idx in range(12):
            theta = idx * (2.0 * np.pi / 12.0)
            px = int(cx + radius * 0.42 * np.cos(theta))
            py = int(cy + radius * 0.42 * np.sin(theta))
            deg = int(np.degrees(theta))
            cv2.ellipse(plate, (px, py), (36, 15), deg, 0, 360, dark, -1, cv2.LINE_AA)
            dx = int(cx + radius * 0.60 * np.cos(theta + 0.14))
            dy = int(cy + radius * 0.60 * np.sin(theta + 0.14))
            cv2.circle(plate, (dx, dy), 7, mid, -1, cv2.LINE_AA)
        cv2.circle(plate, (cx, cy), int(radius * 0.14), dark, 4, cv2.LINE_AA)

    if counterfeit:
        stamp = (168, 155, 42)
        step = max(36, radius // 5)
        for offset in range(-radius, radius + 1, step):
            cv2.line(plate, (cx + offset, cy - radius), (cx + offset, cy + radius), stamp, 2, cv2.LINE_AA)
            cv2.line(plate, (cx - radius, cy + offset), (cx + radius, cy + offset), stamp, 2, cv2.LINE_AA)
        cv2.rectangle(
            plate,
            (cx - int(radius * 0.42), cy - int(radius * 0.42)),
            (cx + int(radius * 0.42), cy + int(radius * 0.42)),
            (130, 125, 35),
            5,
        )

    if crack:
        pts = np.array(
            [
                [cx - int(radius * 0.40), cy - int(radius * 0.10)],
                [cx - int(radius * 0.15), cy - int(radius * 0.04)],
                [cx + int(radius * 0.08), cy + int(radius * 0.06)],
                [cx + int(radius * 0.34), cy + int(radius * 0.02)],
            ],
            dtype=np.int32,
        )
        cv2.polylines(plate, [pts], False, (16, 16, 18), 4, cv2.LINE_AA)

    img[mask > 0] = plate[mask > 0]
    return img


def _encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    assert ok
    return bytes(buf)


def _build_isolated_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TestClient, dict[str, Path]]:
    db_path = tmp_path / "ceramscan_test.db"
    overlay_dir = tmp_path / "overlays" / "runtime"
    qr_dir = tmp_path / "qr"
    passport_dir = tmp_path / "passports"

    monkeypatch.setenv("CERAMSCAN_DB_PATH", str(db_path))
    monkeypatch.setenv("CERAMSCAN_OVERLAY_DIR", str(overlay_dir))
    monkeypatch.setenv("CERAMSCAN_QR_DIR", str(qr_dir))
    monkeypatch.setenv("CERAMSCAN_PASSPORT_DIR", str(passport_dir))
    monkeypatch.delenv("CERAMSCAN_WEIGHTS_PATH", raising=False)

    app = create_app()
    paths = {
        "db_path": db_path,
        "overlay_dir": overlay_dir,
        "qr_dir": qr_dir,
        "passport_dir": passport_dir,
    }
    return TestClient(app), paths


def test_health_and_favicon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /health returns ok status and /favicon.ico avoids 404."""
    client, _paths = _build_isolated_client(tmp_path, monkeypatch)

    health_resp = client.get("/health")
    assert health_resp.status_code == 200
    assert health_resp.json() == {
        "status": "ok",
        "version": "0.1.0",
        "model_mode": "deterministic",
        "db_ok": True,
    }

    fav_resp = client.get("/favicon.ico")
    assert fav_resp.status_code == 204


def test_health_returns_500_on_db_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /health returns HTTP 500 when the configured DB path cannot be opened."""
    bad_db_dir = tmp_path / "db_is_a_directory"
    bad_db_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("CERAMSCAN_DB_PATH", str(bad_db_dir))
    monkeypatch.setenv("CERAMSCAN_OVERLAY_DIR", str(tmp_path / "overlays" / "runtime"))
    monkeypatch.setenv("CERAMSCAN_QR_DIR", str(tmp_path / "qr"))
    monkeypatch.setenv("CERAMSCAN_PASSPORT_DIR", str(tmp_path / "passports"))

    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 500
    assert resp.json().get("detail")


def test_demo_samples_and_empty_db_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify /api/demo-samples exposes bisque/glazed samples, unique passports, and leaves DB empty."""
    client, _paths = _build_isolated_client(tmp_path, monkeypatch)

    resp = client.get("/api/demo-samples")
    assert resp.status_code == 200
    payload = resp.json()

    samples = payload["samples"]
    assert len(samples) >= 4
    assert samples[0]["stage"] == "bisque"
    stages = {s["stage"] for s in samples}
    assert {"bisque", "glazed"}.issubset(stages)
    for sample in samples:
        assert {"id", "label", "stage", "image_url"}.issubset(sample.keys())
        asset_resp = client.get(sample["image_url"])
        assert asset_resp.status_code == 200
        assert asset_resp.content[:8] == PNG_MAGIC or asset_resp.content[:3] == JPEG_MAGIC

    passports = payload["passports"]
    assert len(passports) >= 2
    item_ids = [p["item_id"] for p in passports]
    assert len(item_ids) == len(set(item_ids))
    assert any(not p["is_counterfeit"] for p in passports)
    assert any(p["is_counterfeit"] for p in passports)
    for p in passports:
        probe_resp = client.get(p["probe_image_url"])
        assert probe_resp.status_code == 200
        assert probe_resp.content[:8] == PNG_MAGIC or probe_resp.content[:3] == JPEG_MAGIC

    summary = client.get("/api/analytics").json()["summary"]
    assert summary["total_inspections"] == 0
    assert summary["total_passports"] == 0
    assert summary["total_verifications"] == 0


def test_inspect_bisque_glazed_and_save_to_db_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify POST /api/inspect handles bisque/glazed, save_to_db=false, and overlay serving."""
    client, paths = _build_isolated_client(tmp_path, monkeypatch)
    bisque_crack_png = _encode_png(_make_plate_bgr(stage="bisque", crack=True, seed=11))
    glazed_clean_png = _encode_png(_make_plate_bgr(stage="glazed", cobalt=True, seed=12))

    res_bisque = client.post(
        "/api/inspect",
        files={"file": ("bisque.png", bisque_crack_png, "image/png")},
        data={"stage": "bisque", "item_id": "BISQUE-001", "shift": "S2", "save_to_db": "true"},
    )
    assert res_bisque.status_code == 200
    b_body = res_bisque.json()
    assert isinstance(b_body["inspection_id"], int)
    assert b_body["item_id"] == "BISQUE-001"
    assert b_body["stage"] == "bisque"
    assert b_body["verdict"] == "REJECT"
    assert b_body["defect_count"] >= 1
    assert any(d["defect_class"] == "crack" and isinstance(d["bbox_xywh"], list) for d in b_body["defects"])
    assert b_body["metrics"]["latency_ms"] < 300.0
    assert b_body["overlay_url"].startswith("/overlays/runtime/")

    decoded_b64 = base64.b64decode(b_body["overlay_base64"])
    assert decoded_b64[:8] == PNG_MAGIC

    overlay_resp = client.get(b_body["overlay_url"])
    assert overlay_resp.status_code == 200
    assert overlay_resp.content[:8] == PNG_MAGIC
    assert (paths["overlay_dir"] / Path(b_body["overlay_url"]).name).is_file()

    res_nosave = client.post(
        "/api/inspect",
        files={"file": ("glazed.png", glazed_clean_png, "image/png")},
        data={"stage": "glazed", "save_to_db": "false"},
    )
    assert res_nosave.status_code == 200
    ns_body = res_nosave.json()
    assert ns_body["inspection_id"] is None
    assert ns_body["stage"] == "glazed"
    assert ns_body["verdict"] in ("PASS", "REWORK", "REJECT")
    assert client.get(ns_body["overlay_url"]).status_code == 200

    analytics = client.get("/api/analytics").json()
    assert analytics["summary"]["total_inspections"] == 1
    assert analytics["by_shift"][0]["shift"] == "S2"


def test_inspect_validation_errors_400_413_422(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify /api/inspect rejects empty, corrupt, tiny, oversized, invalid item_id, and bad form fields."""
    client, _paths = _build_isolated_client(tmp_path, monkeypatch)
    valid_png = _encode_png(_make_plate_bgr(stage="glazed", cobalt=True, seed=21))
    tiny_png = _encode_png(np.full((16, 16, 3), 180, dtype=np.uint8))
    oversized = PNG_MAGIC + b"\x00" * (MAX_UPLOAD_BYTES + 1)

    assert client.post(
        "/api/inspect",
        files={"file": ("empty.png", b"", "image/png")},
        data={"stage": "glazed"},
    ).status_code == 400

    corrupt_resp = client.post(
        "/api/inspect",
        files={"file": ("corrupt.jpg", b"not-an-image", "image/jpeg")},
        data={"stage": "glazed"},
    )
    assert corrupt_resp.status_code == 400
    corrupt_detail = corrupt_resp.json().get("detail", "")
    assert corrupt_detail
    assert "BytesIO" not in corrupt_detail
    assert "0x" not in corrupt_detail

    assert client.post(
        "/api/inspect",
        files={"file": ("tiny.png", tiny_png, "image/png")},
        data={"stage": "glazed"},
    ).status_code == 400

    assert client.post(
        "/api/inspect",
        files={"file": ("bad_id.png", valid_png, "image/png")},
        data={"stage": "glazed", "item_id": "../../etc/passwd", "save_to_db": "false"},
    ).status_code == 400

    assert client.post(
        "/api/inspect",
        files={"file": ("huge.png", oversized, "image/png")},
        data={"stage": "glazed"},
    ).status_code == 413

    assert client.post(
        "/api/inspect",
        files={"file": ("bad_stage.png", valid_png, "image/png")},
        data={"stage": "kiln"},
    ).status_code == 422

    for bad_diam in ("0", "-1", "inf", "+inf", "-inf", "Infinity", "nan", "NaN"):
        diam_resp = client.post(
            "/api/inspect",
            files={"file": ("bad_diam.png", valid_png, "image/png")},
            data={"stage": "glazed", "reference_diameter_mm": bad_diam},
        )
        assert diam_resp.status_code == 422, f"Expected 422 for {bad_diam!r}"

    ok_diam_resp = client.post(
        "/api/inspect",
        files={"file": ("ok_diam.png", valid_png, "image/png")},
        data={"stage": "glazed", "reference_diameter_mm": "200.0", "save_to_db": "false"},
    )
    assert ok_diam_resp.status_code == 200


def test_passport_register_auto_custom_duplicate_and_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify passport registration auto/custom IDs, QR decode, 409 duplicate, and 400 traversal guard."""
    client, paths = _build_isolated_client(tmp_path, monkeypatch)
    cobalt_png = _encode_png(_make_plate_bgr(stage="glazed", cobalt=True, seed=31))

    for bad_id in ("../evil", "../../etc/passwd", "a/b", "x'; DROP TABLE passports;--", "ab", "X" * 65):
        bad_resp = client.post(
            "/api/passport/register",
            files={"file": ("bad.png", cobalt_png, "image/png")},
            data={"item_id": bad_id, "stage": "glazed"},
        )
        assert bad_resp.status_code == 400

    assert list(paths["qr_dir"].iterdir()) == []
    assert list(paths["passport_dir"].iterdir()) == []
    assert client.get("/api/analytics").json()["summary"]["total_passports"] == 0

    auto_resp = client.post(
        "/api/passport/register",
        files={"file": ("auto.png", cobalt_png, "image/png")},
        data={"stage": "glazed"},
    )
    assert auto_resp.status_code == 200
    auto_body = auto_resp.json()
    assert re.match(r"^GZHEL-\d{8}-[0-9a-f]{6}$", auto_body["item_id"])
    assert auto_body["qr_payload"] == f"CERAMSCAN:{auto_body['item_id']}"

    custom_resp = client.post(
        "/api/passport/register",
        files={"file": ("custom.png", cobalt_png, "image/png")},
        data={
            "item_id": "GZHEL-TEST-001",
            "sku": "SKU-99",
            "master_name": "Иванов И.",
            "batch_note": "Печь-1",
            "stage": "glazed",
        },
    )
    assert custom_resp.status_code == 200
    c_body = custom_resp.json()
    assert c_body["qr_image_url"] == "/qr/GZHEL-TEST-001.png"
    assert c_body["reference_image_url"] == "/passports/GZHEL-TEST-001.png"

    qr_resp = client.get(c_body["qr_image_url"])
    assert qr_resp.status_code == 200
    assert qr_resp.content[:8] == PNG_MAGIC
    qr_arr = np.frombuffer(qr_resp.content, dtype=np.uint8)
    qr_img = cv2.imdecode(qr_arr, cv2.IMREAD_COLOR)
    assert qr_img is not None
    decoded_text, _pts, _straight = cv2.QRCodeDetector().detectAndDecode(qr_img)
    assert decoded_text == "CERAMSCAN:GZHEL-TEST-001"

    ref_resp = client.get(c_body["reference_image_url"])
    assert ref_resp.status_code == 200
    assert ref_resp.content[:8] == PNG_MAGIC

    dup_resp = client.post(
        "/api/passport/register",
        files={"file": ("dup.png", cobalt_png, "image/png")},
        data={"item_id": "GZHEL-TEST-001", "stage": "glazed"},
    )
    assert dup_resp.status_code == 409


def test_passport_verify_genuine_counterfeit_lazy_demo_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify passport matching, counterfeit detection, lazy demo seeding, and 404/400/413 guards."""
    client, _paths = _build_isolated_client(tmp_path, monkeypatch)
    genuine_png = _encode_png(_make_plate_bgr(stage="glazed", cobalt=True, seed=42))
    fake_png = _encode_png(_make_plate_bgr(stage="glazed", counterfeit=True, seed=99))
    oversized = PNG_MAGIC + b"\x00" * (MAX_UPLOAD_BYTES + 1)

    assert client.post(
        "/api/passport/register",
        files={"file": ("ref.png", genuine_png, "image/png")},
        data={"item_id": "GZHEL-VERIFY-01", "sku": "SKU-V1", "master_name": "Мастер А.", "batch_note": "B1"},
    ).status_code == 200

    ok_resp = client.post(
        "/api/passport/verify/GZHEL-VERIFY-01",
        files={"file": ("probe.png", genuine_png, "image/png")},
    )
    assert ok_resp.status_code == 200
    ok_body = ok_resp.json()
    assert ok_body["verdict"] == VERDICT_AUTHENTIC
    assert ok_body["match_percent"] >= 62.0
    assert ok_body["registered_sku"] == "SKU-V1"
    assert ok_body["reference_image_url"] == "/passports/GZHEL-VERIFY-01.png"
    assert client.get(ok_body["reference_image_url"]).status_code == 200

    fake_resp = client.post(
        "/api/passport/verify/GZHEL-VERIFY-01",
        files={"file": ("fake.png", fake_png, "image/png")},
    )
    assert fake_resp.status_code == 200
    fake_body = fake_resp.json()
    assert fake_body["verdict"] == VERDICT_COUNTERFEIT
    assert fake_body["match_percent"] < ok_body["match_percent"]

    # Lazy demo passport verification from /api/demo-samples without prior /register
    demo_passports = client.get("/api/demo-samples").json()["passports"]
    genuine_demo = next(p for p in demo_passports if not p["is_counterfeit"])
    fake_demo = next(p for p in demo_passports if p["is_counterfeit"])

    g_probe_bytes = client.get(genuine_demo["probe_image_url"]).content
    g_demo_ver = client.post(
        f"/api/passport/verify/{genuine_demo['item_id']}",
        files={"file": ("demo_g.png", g_probe_bytes, "image/png")},
    )
    assert g_demo_ver.status_code == 200
    assert g_demo_ver.json()["verdict"] == VERDICT_AUTHENTIC
    assert client.get(g_demo_ver.json()["reference_image_url"]).status_code == 200

    f_probe_bytes = client.get(fake_demo["probe_image_url"]).content
    f_demo_ver = client.post(
        f"/api/passport/verify/{fake_demo['item_id']}",
        files={"file": ("demo_f.png", f_probe_bytes, "image/png")},
    )
    assert f_demo_ver.status_code == 200
    assert f_demo_ver.json()["verdict"] == VERDICT_COUNTERFEIT
    assert client.get(f_demo_ver.json()["reference_image_url"]).status_code == 200

    assert client.post(
        "/api/passport/verify/GZHEL-NONEXISTENT-999",
        files={"file": ("probe.png", genuine_png, "image/png")},
    ).status_code == 404

    assert client.post(
        "/api/passport/verify/ab",
        files={"file": ("probe.png", genuine_png, "image/png")},
    ).status_code == 400

    assert client.post(
        f"/api/passport/verify/{quote('bad id!', safe='')}",
        files={"file": ("probe.png", genuine_png, "image/png")},
    ).status_code == 400

    assert client.post(
        f"/api/passport/verify/{quote('../evil', safe='')}",
        files={"file": ("probe.png", genuine_png, "image/png")},
    ).status_code in (400, 404)

    assert client.post(
        "/api/passport/verify/GZHEL-VERIFY-01",
        files={"file": ("bad.png", b"corrupt", "image/png")},
    ).status_code == 400

    assert client.post(
        "/api/passport/verify/GZHEL-VERIFY-01",
        files={"file": ("huge.png", oversized, "image/png")},
    ).status_code == 413


def test_analytics_global_summary_empty_query_normalization_and_422(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify /api/analytics preserves global summary across filters, normalizes '' params, and returns 422."""
    client, _paths = _build_isolated_client(tmp_path, monkeypatch)
    bisque_crack_png = _encode_png(_make_plate_bgr(stage="bisque", crack=True, seed=51))
    glazed_clean_png = _encode_png(_make_plate_bgr(stage="glazed", cobalt=True, seed=52))

    assert client.post(
        "/api/inspect",
        files={"file": ("b.png", bisque_crack_png, "image/png")},
        data={"stage": "bisque"},
    ).status_code == 200
    assert client.post(
        "/api/inspect",
        files={"file": ("g.png", glazed_clean_png, "image/png")},
        data={"stage": "glazed"},
    ).status_code == 200

    empty_filter_resp = client.get("/api/analytics", params={"limit": 25, "stage": "", "verdict": ""})
    assert empty_filter_resp.status_code == 200
    full_body = empty_filter_resp.json()
    assert full_body["summary"]["total_inspections"] == 2
    assert len(full_body["recent_inspections"]) == 2

    bisque_only = client.get("/api/analytics", params={"limit": 25, "stage": "bisque", "verdict": ""})
    assert bisque_only.status_code == 200
    b_data = bisque_only.json()
    assert b_data["summary"]["total_inspections"] == 2
    assert len(b_data["recent_inspections"]) == 1
    assert b_data["recent_inspections"][0]["stage"] == "bisque"

    pass_only = client.get("/api/analytics", params={"limit": 25, "stage": "", "verdict": "PASS"})
    assert pass_only.status_code == 200
    p_data = pass_only.json()
    assert p_data["summary"]["total_inspections"] == 2
    assert all(row["verdict"] == "PASS" for row in p_data["recent_inspections"])

    for bad_params in ({"limit": 0}, {"limit": 201}, {"stage": "kiln"}, {"verdict": "UNKNOWN"}):
        assert client.get("/api/analytics", params=bad_params).status_code == 422


def test_benchmark_overlays_and_invalid_weights_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify /overlays serves benchmark PNGs alongside /overlays/runtime and invalid weights fail fast."""
    client, _paths = _build_isolated_client(tmp_path, monkeypatch)
    overlays_dir = PROJECT_ROOT / "reports" / "overlays"
    existing_pngs = sorted(overlays_dir.glob("*.png"))
    if existing_pngs:
        resp = client.get(f"/overlays/{existing_pngs[0].name}")
        assert resp.status_code == 200
        assert resp.content[:8] == PNG_MAGIC

    monkeypatch.setenv("CERAMSCAN_WEIGHTS_PATH", str(tmp_path / "missing.onnx"))
    with pytest.raises(FileNotFoundError):
        create_app()


def test_decode_robustness_truncated_decompression_bomb_and_no_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify malformed payloads, decompression bombs, and decode errors return clean 400 without leaks."""
    assert {
        UnidentifiedImageError,
        Image.DecompressionBombError,
        EOFError,
        OSError,
        ValueError,
        SyntaxError,
        struct.error,
        zlib.error,
        cv2.error,
    }.issubset(set(_DECODE_ERRORS))

    client, _paths = _build_isolated_client(tmp_path, monkeypatch)
    valid_png = _encode_png(_make_plate_bgr(stage="glazed", cobalt=True, seed=61))

    def _png_chunk(tag: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(tag + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", crc)

    gif_buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 140, 180)).save(gif_buf, format="GIF")
    gif_bytes = gif_buf.getvalue()

    ihdr_64 = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 64, 8, 2, 0, 0, 0))
    ihdr_bomb = _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 20000, 20000, 8, 2, 0, 0, 0))
    iend = _png_chunk(b"IEND", b"")

    bomb_png = PNG_MAGIC + ihdr_bomb + _png_chunk(b"IDAT", zlib.compress(b"\x00" * 16)) + iend
    assert len(bomb_png) < 200

    malformed_cases: list[tuple[str, bytes, str]] = [
        ("truncated.png", valid_png[: len(valid_png) // 2], "image/png"),
        ("truncated.gif", gif_bytes[: len(gif_bytes) // 2], "image/gif"),
        ("header_only.gif", b"GIF89a\x40\x00\x40\x00\x00\x00\x00;", "image/gif"),
        ("bomb.png", bomb_png, "image/png"),
        ("no_idat.png", PNG_MAGIC + ihdr_64 + iend, "image/png"),
        ("corrupt_zlib.png", PNG_MAGIC + ihdr_64 + _png_chunk(b"IDAT", b"not_zlib") + iend, "image/png"),
    ]

    for filename, payload, content_type in malformed_cases:
        resp = client.post(
            "/api/inspect",
            files={"file": (filename, payload, content_type)},
            data={"stage": "glazed", "save_to_db": "false"},
        )
        assert resp.status_code == 400, f"Expected 400 for {filename}, got {resp.status_code}"
        detail = resp.json().get("detail", "")
        assert detail, f"Empty detail for {filename}"
        assert "BytesIO" not in detail, f"BytesIO leaked for {filename}: {detail}"
        assert "0x" not in detail, f"Memory address leaked for {filename}: {detail}"

    reg_ok = client.post(
        "/api/passport/register",
        files={"file": ("ref.png", valid_png, "image/png")},
        data={"item_id": "GZHEL-ROBUST-01", "stage": "glazed"},
    )
    assert reg_ok.status_code == 200

    for filename, payload, content_type in (
        ("truncated.png", valid_png[: len(valid_png) // 2], "image/png"),
        ("bomb.png", bomb_png, "image/png"),
        ("corrupt.bin", b"NOT_AN_IMAGE_PAYLOAD", "application/octet-stream"),
    ):
        reg_bad = client.post(
            "/api/passport/register",
            files={"file": (filename, payload, content_type)},
            data={"stage": "glazed"},
        )
        assert reg_bad.status_code == 400, f"Expected 400 on register for {filename}"
        reg_detail = reg_bad.json().get("detail", "")
        assert reg_detail
        assert "BytesIO" not in reg_detail
        assert "0x" not in reg_detail

        ver_bad = client.post(
            "/api/passport/verify/GZHEL-ROBUST-01",
            files={"file": (filename, payload, content_type)},
        )
        assert ver_bad.status_code == 400, f"Expected 400 on verify for {filename}"
        ver_detail = ver_bad.json().get("detail", "")
        assert ver_detail
        assert "BytesIO" not in ver_detail
        assert "0x" not in ver_detail
