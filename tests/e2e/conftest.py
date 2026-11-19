"""Shared fixtures and test infrastructure for CeramScan E2E suite."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ANALYTICS_PYTHON = Path("/Users/alex/analytics_env/bin/python")
SLOP_SCAN_SCRIPT = Path(
    "/Users/alex/.gemini/config/skills/anti-ai-slop-architect/scripts/slop_scan.py"
)


def pytest_configure(config: pytest.Config) -> None:
    """Register tier1..tier4 and f1..f15 markers programmatically."""
    tier_docs = (
        ("tier1", "Tier 1: Feature Coverage happy-path tests"),
        ("tier2", "Tier 2: Boundary & Corner Case tests"),
        ("tier3", "Tier 3: Cross-Feature Pairwise Combination tests"),
        ("tier4", "Tier 4: Real-World Application Scenario tests"),
    )
    for name, desc in tier_docs:
        config.addinivalue_line("markers", f"{name}: {desc}")

    for idx in range(1, 16):
        config.addinivalue_line("markers", f"f{idx}: Feature F{idx} from PROJECT.md")


# ---------------------------------------------------------------------------
# Pure-Python ONNX ModelProto wire encoder (zero external `onnx` dependency)
# ---------------------------------------------------------------------------


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
    # AttributeProto.AttributeType.TENSOR == 4
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
        + _pb_field_bytes(2, b"ceramscan-e2e")
        + _pb_field_bytes(7, graph_proto)
        + _pb_field_bytes(8, opset_proto)
    )


# ---------------------------------------------------------------------------
# Deterministic Ceramic Plate & Counterfeit Image Factory
# ---------------------------------------------------------------------------


class CeramicImageFactory:
    """Deterministic generator for synthetic ceramic plates and test payloads."""

    def make_ceramic_plate_bgr(
        self,
        stage: str = "glazed",
        cobalt: bool = False,
        crack: bool = False,
        chip: bool = False,
        warp: bool = False,
        glare: bool = False,
        glaze_bare: bool = False,
        pinhole_count: int = 0,
        drip: bool = False,
        quad_frame: bool = False,
        size: int = 720,
        seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)
        img = np.full((size, size, 3), (28, 28, 30), dtype=np.uint8)
        cx, cy = size // 2, size // 2
        radius = int(size * 0.36)

        if quad_frame:
            card_pts = np.array(
                [
                    [int(size * 0.14), int(size * 0.12)],
                    [int(size * 0.86), int(size * 0.15)],
                    [int(size * 0.84), int(size * 0.88)],
                    [int(size * 0.12), int(size * 0.85)],
                ],
                dtype=np.int32,
            )
            cv2.fillConvexPoly(img, card_pts, (54, 58, 64))
            cv2.polylines(img, [card_pts], True, (240, 240, 240), 4, cv2.LINE_AA)

        base_color = (176, 178, 182) if stage == "bisque" else (218, 220, 224)
        ax_x = radius
        ax_y = int(radius * 0.86) if warp else radius

        plate_mask = np.zeros((size, size), dtype=np.uint8)
        if warp:
            angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
            r_wave = 1.0 + 0.045 * np.sin(3 * angles) + 0.03 * np.cos(2 * angles)
            pts = np.stack(
                [
                    cx + ax_x * r_wave * np.cos(angles),
                    cy + ax_y * r_wave * np.sin(angles),
                ],
                axis=1,
            ).astype(np.int32)
            cv2.fillPoly(plate_mask, [pts], 255)
        else:
            cv2.ellipse(plate_mask, (cx, cy), (ax_x, ax_y), 0, 0, 360, 255, -1)

        if chip:
            chip_cx = cx + ax_x - int(size * 0.015)
            chip_cy = cy
            cv2.circle(plate_mask, (chip_cx, chip_cy), max(24, int(size * 0.042)), 0, -1)

        plate_layer = np.full((size, size, 3), base_color, dtype=np.uint8)
        noise_sigma = 3 if stage == "bisque" else 1
        noise = rng.integers(-noise_sigma, noise_sigma + 1, size=(size, size, 3))
        plate_layer = np.clip(plate_layer.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        rim_tone = (166, 168, 172) if stage == "bisque" else (206, 210, 214)
        cv2.ellipse(plate_layer, (cx, cy), (int(ax_x * 0.96), int(ax_y * 0.96)), 0, 0, 360, rim_tone, 2)

        if cobalt:
            cobalt_dark = (182, 52, 20)
            cobalt_mid = (205, 86, 44)
            cv2.ellipse(plate_layer, (cx, cy), (int(radius * 0.72), int(radius * 0.72)), 0, 0, 360, cobalt_mid, 3, cv2.LINE_AA)
            for idx in range(12):
                theta = idx * (2 * np.pi / 12.0)
                px = int(cx + radius * 0.42 * np.cos(theta))
                py = int(cy + radius * 0.42 * np.sin(theta))
                deg = int(np.degrees(theta))
                cv2.ellipse(plate_layer, (px, py), (38, 16), deg, 0, 360, cobalt_dark, -1, cv2.LINE_AA)
                dx = int(cx + radius * 0.62 * np.cos(theta + 0.14))
                dy = int(cy + radius * 0.62 * np.sin(theta + 0.14))
                cv2.circle(plate_layer, (dx, dy), 7, cobalt_mid, -1, cv2.LINE_AA)
                leaf_x = int(cx + radius * 0.25 * np.cos(theta + 0.26))
                leaf_y = int(cy + radius * 0.25 * np.sin(theta + 0.26))
                cv2.ellipse(plate_layer, (leaf_x, leaf_y), (18, 7), deg + 35, 0, 360, cobalt_dark, -1, cv2.LINE_AA)
            cv2.circle(plate_layer, (cx, cy), int(radius * 0.14), cobalt_dark, 4, cv2.LINE_AA)
            for k in range(8):
                ang = k * (np.pi / 4.0)
                sx = int(cx + radius * 0.08 * np.cos(ang))
                sy = int(cy + radius * 0.08 * np.sin(ang))
                cv2.circle(plate_layer, (sx, sy), 4, cobalt_mid, -1, cv2.LINE_AA)

        if glaze_bare:
            bare_center = (cx - int(radius * 0.28), cy + int(radius * 0.22))
            cv2.ellipse(plate_layer, bare_center, (34, 24), 22, 0, 360, (148, 142, 134), -1, cv2.LINE_AA)
            cv2.ellipse(plate_layer, bare_center, (34, 24), 22, 0, 360, (112, 108, 102), 2, cv2.LINE_AA)

        if drip:
            drip_pts = np.array(
                [
                    [cx + int(radius * 0.15), cy - int(radius * 0.35)],
                    [cx + int(radius * 0.18), cy - int(radius * 0.12)],
                    [cx + int(radius * 0.16), cy + int(radius * 0.04)],
                ],
                dtype=np.int32,
            )
            cv2.polylines(plate_layer, [drip_pts], False, (192, 200, 212), 14, cv2.LINE_AA)
            cv2.polylines(plate_layer, [drip_pts], False, (132, 140, 152), 3, cv2.LINE_AA)

        if pinhole_count > 0:
            cols = max(4, int(np.ceil(np.sqrt(pinhole_count))))
            span = int(radius * 0.95)
            step = max(24, span // cols)
            placed = 0
            for r_idx in range(cols):
                for c_idx in range(cols):
                    if placed >= pinhole_count:
                        break
                    px = cx - span // 2 + c_idx * step + 8
                    py = cy - span // 2 + r_idx * step + 8
                    if (px - cx) ** 2 + (py - cy) ** 2 < (radius * 0.72) ** 2:
                        cv2.circle(plate_layer, (px, py), 2, (20, 20, 22), -1)
                        placed += 1

        if crack:
            crack_pts = np.array(
                [
                    [cx - int(radius * 0.42), cy - int(radius * 0.12)],
                    [cx - int(radius * 0.18), cy - int(radius * 0.05)],
                    [cx + int(radius * 0.04), cy + int(radius * 0.07)],
                    [cx + int(radius * 0.34), cy + int(radius * 0.02)],
                ],
                dtype=np.int32,
            )
            cv2.polylines(plate_layer, [crack_pts], False, (18, 18, 20), 3, cv2.LINE_AA)

        if glare and stage == "glazed":
            glare_mask = np.zeros((size, size), dtype=np.float32)
            cv2.ellipse(
                glare_mask,
                (cx - int(radius * 0.15), cy - int(radius * 0.20)),
                (int(radius * 0.42), int(radius * 0.25)),
                -25,
                0,
                360,
                1.0,
                -1,
            )
            glare_mask = cv2.GaussianBlur(glare_mask, (15, 15), 5)
            white = np.full_like(plate_layer, 252, dtype=np.float32)
            alpha = glare_mask[:, :, None]
            plate_layer = np.clip(plate_layer.astype(np.float32) * (1.0 - alpha) + white * alpha, 0, 255).astype(np.uint8)

        img[plate_mask > 0] = plate_layer[plate_mask > 0]

        if chip:
            chip_cx = cx + ax_x - int(size * 0.015)
            cv2.circle(img, (chip_cx, cy), max(24, int(size * 0.042)), (28, 28, 30), -1)
            cv2.circle(img, (chip_cx - 4, cy), max(24, int(size * 0.042)), (120, 114, 108), 2)

        return img

    def make_counterfeit_plate_bgr(self, size: int = 720, seed: int = 99) -> np.ndarray:
        rng = np.random.default_rng(seed)
        img = np.full((size, size, 3), (28, 28, 30), dtype=np.uint8)
        cx, cy = size // 2, size // 2
        radius = int(size * 0.36)
        plate_mask = np.zeros((size, size), dtype=np.uint8)
        cv2.circle(plate_mask, (cx, cy), radius, 255, -1)

        plate_layer = np.full((size, size, 3), (218, 220, 224), dtype=np.uint8)
        noise = rng.integers(-1, 2, size=(size, size, 3))
        plate_layer = np.clip(plate_layer.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Distinct stamped geometric lattice with shifted teal-cyan tint
        stamp_color = (168, 155, 42)
        step = max(36, radius // 5)
        for offset in range(-radius, radius + 1, step):
            cv2.line(plate_layer, (cx + offset, cy - radius), (cx + offset, cy + radius), stamp_color, 2, cv2.LINE_AA)
            cv2.line(plate_layer, (cx - radius, cy + offset), (cx + radius, cy + offset), stamp_color, 2, cv2.LINE_AA)
        cv2.rectangle(
            plate_layer,
            (cx - int(radius * 0.45), cy - int(radius * 0.45)),
            (cx + int(radius * 0.45), cy + int(radius * 0.45)),
            (130, 125, 35),
            5,
        )
        img[plate_mask > 0] = plate_layer[plate_mask > 0]
        return img

    def encode_png(self, bgr: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            raise ValueError("Failed to encode image as PNG")
        return bytes(buf)

    def make_bisque_plate_png(self, with_crack: bool = False) -> bytes:
        bgr = self.make_ceramic_plate_bgr(stage="bisque", crack=with_crack, size=640, seed=42)
        return self.encode_png(bgr)

    def make_cobalt_plate_png(self, variant: int = 0) -> bytes:
        if variant == 0:
            bgr = self.make_ceramic_plate_bgr(stage="glazed", cobalt=True, size=640, seed=42)
        else:
            bgr = self.make_counterfeit_plate_bgr(size=640, seed=99 + variant)
        return self.encode_png(bgr)

    def make_tiny_png(self, size: int = 16) -> bytes:
        tiny = np.full((size, size, 3), 180, dtype=np.uint8)
        return self.encode_png(tiny)


# ---------------------------------------------------------------------------
# Session Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return repository root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def venv_python() -> Path:
    """Return path to `.venv/bin/python` and assert it exists."""
    py = PROJECT_ROOT / ".venv" / "bin" / "python"
    assert py.is_file(), f"Missing .venv Python at {py}"
    return py


@pytest.fixture(scope="session")
def slop_scanner_cmd() -> list[str]:
    """Return command prefix for running `slop_scan.py`."""
    assert ANALYTICS_PYTHON.is_file(), f"Missing analytics Python at {ANALYTICS_PYTHON}"
    assert SLOP_SCAN_SCRIPT.is_file(), f"Missing slop_scan.py at {SLOP_SCAN_SCRIPT}"
    return [str(ANALYTICS_PYTHON), str(SLOP_SCAN_SCRIPT)]


@pytest.fixture(scope="session")
def reference_defect_images(repo_root: Path) -> dict[str, Path]:
    """Return paths to pre-loaded `ceramic_defect_1..4.jpg` samples."""
    samples_dir = repo_root / "data" / "reference_materials" / "ceramic_defect_samples"
    mapping = {f"defect_{idx}": samples_dir / f"ceramic_defect_{idx}.jpg" for idx in range(1, 5)}
    for key, path in mapping.items():
        assert path.is_file(), f"Missing reference defect sample {key}: {path}"
    return mapping


@pytest.fixture(scope="session")
def ceramic_image_factory() -> CeramicImageFactory:
    """Provide deterministic synthetic ceramic plate and PNG byte factory."""
    return CeramicImageFactory()


# ---------------------------------------------------------------------------
# Function / Module Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def invalid_image_payloads(ceramic_image_factory: CeramicImageFactory) -> dict[str, bytes]:
    """Return invalid binary image payloads for negative upload & CV boundary tests."""
    return {
        "empty": b"",
        "corrupt": b"not-an-image-payload-000",
        "too_small_png": ceramic_image_factory.make_tiny_png(16),
        "oversized_over_10mb": b"\x89PNG\r\n\x1a\n" + b"\x00" * (10 * 1024 * 1024 + 512),
    }


@pytest.fixture()
def onnx_weights_factory(tmp_path: Path) -> Callable[[str], Path]:
    """Generate real ONNX protobuf files or invalid paths for `--weights` testing."""

    def _create(mode: str) -> Path:
        if mode == "missing":
            return tmp_path / "missing_model.onnx"
        if mode == "non_onnx":
            bad_path = tmp_path / "weights.txt"
            bad_path.write_text("not-onnx", encoding="utf-8")
            return bad_path
        if mode == "corrupt":
            corrupt_path = tmp_path / "corrupt.onnx"
            corrupt_path.write_bytes(b"corrupt-onnx-protobuf-bytes")
            return corrupt_path
        if mode == "bad_rank":
            arr = np.zeros((1, 10), dtype=np.float32)
            out_path = tmp_path / "bad_rank.onnx"
            out_path.write_bytes(_build_constant_onnx_bytes(arr))
            return out_path
        if mode == "yolov8":
            # Shape (1, 10, 4): 4 bbox coords + 6 classes, 4 candidates
            arr = np.zeros((1, 10, 4), dtype=np.float32)
            arr[0, 0, 0] = 320.0
            arr[0, 1, 0] = 320.0
            arr[0, 2, 0] = 140.0
            arr[0, 3, 0] = 20.0
            arr[0, 4, 0] = 0.92  # class 0: CRACK
            for cand in range(1, 4):
                arr[0, 0:4, cand] = [100.0 * cand, 100.0, 20.0, 20.0]
                arr[0, 4, cand] = 0.01
            out_path = tmp_path / "yolov8_crack.onnx"
            out_path.write_bytes(_build_constant_onnx_bytes(arr))
            return out_path
        if mode == "yolov8_clean":
            arr = np.zeros((1, 10, 4), dtype=np.float32)
            for cand in range(4):
                arr[0, 0:4, cand] = [200.0, 200.0, 20.0, 20.0]
                arr[0, 4, cand] = 0.01
            out_path = tmp_path / "yolov8_clean.onnx"
            out_path.write_bytes(_build_constant_onnx_bytes(arr))
            return out_path
        if mode == "yolov5":
            # Shape (1, 4, 11): 4 bbox + 1 obj + 6 classes; two overlapping GLAZE_BARE boxes
            arr = np.zeros((1, 4, 11), dtype=np.float32)
            arr[0, 0, :5] = [300.0, 300.0, 90.0, 90.0, 0.95]
            arr[0, 0, 5 + 2] = 0.95  # class 2: GLAZE_BARE
            arr[0, 1, :5] = [304.0, 302.0, 90.0, 90.0, 0.90]
            arr[0, 1, 5 + 2] = 0.90
            out_path = tmp_path / "yolov5_glaze.onnx"
            out_path.write_bytes(_build_constant_onnx_bytes(arr))
            return out_path
        raise ValueError(f"Unsupported onnx_weights_factory mode: {mode}")

    return _create


@pytest.fixture()
def isolated_workspace(tmp_path: Path) -> dict[str, Path]:
    """Create isolated temp workspace directories for DB, reports, overlays, and photos."""
    reports_dir = tmp_path / "reports"
    overlays_dir = reports_dir / "overlays"
    runtime_overlay_dir = overlays_dir / "runtime"
    qr_dir = reports_dir / "qr"
    passport_dir = reports_dir / "passports"
    synthetic_dir = tmp_path / "data" / "synthetic"
    real_photos_dir = tmp_path / "data" / "real_photos"
    yolo_export_dir = tmp_path / "yolo_export"

    for directory in (
        reports_dir,
        overlays_dir,
        runtime_overlay_dir,
        qr_dir,
        passport_dir,
        synthetic_dir,
        yolo_export_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    for sub in ("pass", "chip", "crack", "glaze", "fake"):
        (real_photos_dir / sub).mkdir(parents=True, exist_ok=True)

    return {
        "db_path": tmp_path / "ceramscan.db",
        "reports_dir": reports_dir,
        "overlays_dir": overlays_dir,
        "runtime_overlay_dir": runtime_overlay_dir,
        "qr_dir": qr_dir,
        "passport_dir": passport_dir,
        "synthetic_dir": synthetic_dir,
        "real_photos_dir": real_photos_dir,
        "yolo_export_dir": yolo_export_dir,
    }


@pytest.fixture()
def isolated_repo_workspace(tmp_path: Path, repo_root: Path) -> Path:
    """Create an isolated repo copy in `tmp_path/ws` for safe `--rebuild-all` runs."""
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)

    for rel_dir in ("src", "scripts", "grant_case"):
        src_path = repo_root / rel_dir
        if src_path.is_dir():
            shutil.copytree(src_path, ws / rel_dir)

    ref_src = repo_root / "data" / "reference_materials"
    if ref_src.is_dir():
        shutil.copytree(ref_src, ws / "data" / "reference_materials")

    for sub in ("pass", "chip", "crack", "glaze", "fake"):
        (ws / "data" / "real_photos" / sub).mkdir(parents=True, exist_ok=True)

    for rep_sub in ("overlays", "qr", "passports"):
        (ws / "reports" / rep_sub).mkdir(parents=True, exist_ok=True)

    return ws


@pytest.fixture()
def real_photos_dir_factory(
    tmp_path: Path,
    ceramic_image_factory: CeramicImageFactory,
) -> Callable[[bool], Path]:
    """Create empty or populated `real_photos` folder trees (`pass/chip/crack/glaze/fake`)."""
    counter = {"idx": 0}

    def _build(populate: bool = True) -> Path:
        counter["idx"] += 1
        base = tmp_path / f"real_photos_{counter['idx']}"
        for sub in ("pass", "chip", "crack", "glaze", "fake"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        if populate:
            cv2.imwrite(
                str(base / "pass" / "bisque_plate_01.jpg"),
                ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", seed=101),
            )
            cv2.imwrite(
                str(base / "pass" / "util_cup_02.jpg"),
                ceramic_image_factory.make_ceramic_plate_bgr(stage="bisque", seed=102),
            )
            cv2.imwrite(
                str(base / "pass" / "glazed_plate_03.jpg"),
                ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", cobalt=True, seed=103),
            )
            cv2.imwrite(
                str(base / "chip" / "chip_01.jpg"),
                ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", chip=True, seed=104),
            )
            cv2.imwrite(
                str(base / "crack" / "crack_01.jpg"),
                ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", crack=True, seed=105),
            )
            cv2.imwrite(
                str(base / "glaze" / "glaze_01.jpg"),
                ceramic_image_factory.make_ceramic_plate_bgr(stage="glazed", glaze_bare=True, seed=106),
            )
            cv2.imwrite(
                str(base / "fake" / "fake_01.jpg"),
                ceramic_image_factory.make_counterfeit_plate_bgr(seed=107),
            )
        return base

    return _build


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_uvicorn_process(proc: subprocess.Popen[str], log_file: Any) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    log_file.close()


@pytest.fixture(scope="session")
def api_server_factory(
    tmp_path_factory: pytest.TempPathFactory,
    repo_root: Path,
    venv_python: Path,
) -> Iterator[Callable[[str], tuple[str, dict[str, Path]]]]:
    """Spawn isolated uvicorn FastAPI servers with `CERAMSCAN_*` env paths."""
    running: dict[str, tuple[subprocess.Popen[str], Any]] = {}

    def _start(name: str = "srv") -> tuple[str, dict[str, Path]]:
        root_dir = tmp_path_factory.mktemp(name)
        db_path = root_dir / "ceramscan.db"
        overlay_dir = root_dir / "overlays" / "runtime"
        qr_dir = root_dir / "qr"
        passport_dir = root_dir / "passports"
        for d in (overlay_dir, qr_dir, passport_dir):
            d.mkdir(parents=True, exist_ok=True)

        port = _find_free_port()
        env = os.environ.copy()
        env["CERAMSCAN_DB_PATH"] = str(db_path)
        env["CERAMSCAN_OVERLAY_DIR"] = str(overlay_dir)
        env["CERAMSCAN_QR_DIR"] = str(qr_dir)
        env["CERAMSCAN_PASSPORT_DIR"] = str(passport_dir)
        env["PYTHONPATH"] = str(repo_root)

        log_path = root_dir / "uvicorn.log"
        log_file = log_path.open("w+", encoding="utf-8")
        proc = subprocess.Popen(
            [
                str(venv_python),
                "-m",
                "uvicorn",
                "src.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=str(repo_root),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

        base_url = f"http://127.0.0.1:{port}"
        running[base_url] = (proc, log_file)
        deadline = time.monotonic() + 15.0
        with httpx.Client(timeout=2.0) as client:
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    log_file.flush()
                    log_output = log_path.read_text(encoding="utf-8", errors="replace")
                    running.pop(base_url, None)
                    log_file.close()
                    raise RuntimeError(
                        f"uvicorn exited early with code {proc.returncode}:\n{log_output}"
                    )
                try:
                    resp = client.get(f"{base_url}/health")
                    if resp.status_code == 200:
                        paths = {
                            "root_dir": root_dir,
                            "db_path": db_path,
                            "overlay_dir": overlay_dir,
                            "qr_dir": qr_dir,
                            "passport_dir": passport_dir,
                        }
                        return base_url, paths
                except httpx.HTTPError:
                    time.sleep(0.2)

        log_file.flush()
        log_output = log_path.read_text(encoding="utf-8", errors="replace")
        running.pop(base_url, None)
        _stop_uvicorn_process(proc, log_file)
        raise TimeoutError(f"uvicorn did not become healthy within 15s:\n{log_output}")

    def _stop(base_url: str) -> None:
        entry = running.pop(base_url, None)
        if entry is not None:
            proc, log_file = entry
            _stop_uvicorn_process(proc, log_file)

    setattr(_start, "stop", _stop)
    try:
        yield _start
    finally:
        for base_url in list(running.keys()):
            _stop(base_url)


@pytest.fixture(scope="module")
def live_api_url(
    api_server_factory: Callable[[str], tuple[str, dict[str, Path]]],
) -> Iterator[str]:
    """Yield base URL for a module-scoped isolated FastAPI server."""
    base_url, _ = api_server_factory("live_api_mod")
    try:
        yield base_url
    finally:
        stop_fn = getattr(api_server_factory, "stop", None)
        if callable(stop_fn):
            stop_fn(base_url)


@pytest.fixture()
def isolated_api_server(
    api_server_factory: Callable[[str], tuple[str, dict[str, Path]]],
) -> Iterator[tuple[str, dict[str, Path]]]:
    """Yield `(base_url, paths_dict)` for a fresh isolated FastAPI server."""
    base_url, paths = api_server_factory("iso_api_fn")
    try:
        yield base_url, paths
    finally:
        stop_fn = getattr(api_server_factory, "stop", None)
        if callable(stop_fn):
            stop_fn(base_url)


@pytest.fixture()
def api_client(
    isolated_api_server: tuple[str, dict[str, Path]],
) -> Iterator[httpx.Client]:
    """Yield an `httpx.Client` bound to `isolated_api_server`."""
    base_url, _ = isolated_api_server
    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        yield client


@pytest.fixture(scope="module")
def playwright_browser() -> Iterator[Any]:
    """Launch headless Google Chrome via Playwright `sync_api`."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        try:
            yield browser
        finally:
            browser.close()


@pytest.fixture()
def ui_page(
    playwright_browser: Any,
) -> Iterator[tuple[Any, dict[str, list[object]]]]:
    """Create a fresh Playwright context/page with error & HTTP failure listeners."""
    context = playwright_browser.new_context()
    page = context.new_page()
    telemetry: dict[str, list[object]] = {
        "console_errors": [],
        "page_errors": [],
        "failed_requests": [],
        "http_errors": [],
    }

    page.on(
        "console",
        lambda msg: telemetry["console_errors"].append(msg.text)
        if msg.type == "error"
        else None,
    )
    page.on("pageerror", lambda err: telemetry["page_errors"].append(str(err)))
    page.on(
        "requestfailed",
        lambda req: telemetry["failed_requests"].append(
            f"{req.method} {req.url}: {req.failure}"
        ),
    )
    page.on(
        "response",
        lambda resp: telemetry["http_errors"].append(f"{resp.status} {resp.url}")
        if resp.status >= 400
        else None,
    )

    try:
        yield page, telemetry
    finally:
        context.close()
