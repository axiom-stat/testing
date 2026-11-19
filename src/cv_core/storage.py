"""SQLite WAL persistence for OTK inspections, 289-FZ digital passports, and QR PNGs."""

from __future__ import annotations

import contextlib
import json
import re
import sqlite3
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import qrcode

from .defect_detector import InspectionResult
from .fingerprint import FingerprintSignature, VerificationResult
from .preprocessing import StageType, VerdictType

ITEM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{3,64}$")
VALID_STAGES: frozenset[str] = frozenset({"bisque", "glazed"})
VALID_VERDICTS: frozenset[str] = frozenset({"PASS", "REWORK", "REJECT"})

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS inspections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT,
    stage TEXT NOT NULL CHECK (stage IN ('bisque', 'glazed')),
    shift TEXT NOT NULL DEFAULT 'S1',
    verdict TEXT NOT NULL CHECK (verdict IN ('PASS', 'REWORK', 'REJECT')),
    gost_grade TEXT NOT NULL,
    model_mode TEXT NOT NULL CHECK (model_mode IN ('deterministic', 'onnx')),
    defect_count INTEGER NOT NULL,
    primary_defect TEXT,
    contour_deviation_pct REAL NOT NULL,
    glare_ratio REAL NOT NULL,
    latency_ms REAL NOT NULL,
    defects_json TEXT NOT NULL,
    gost_rules_json TEXT NOT NULL,
    overlay_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS passports (
    item_id TEXT PRIMARY KEY,
    sku TEXT NOT NULL,
    master_name TEXT NOT NULL,
    batch_note TEXT NOT NULL DEFAULT '',
    stage TEXT NOT NULL CHECK (stage IN ('bisque', 'glazed')),
    qr_payload TEXT NOT NULL,
    qr_image_path TEXT NOT NULL,
    reference_image_path TEXT NOT NULL DEFAULT '',
    signature_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS passport_verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES passports(item_id) ON DELETE CASCADE,
    match_percent REAL NOT NULL,
    orb_score REAL NOT NULL,
    phash_score REAL NOT NULL,
    color_score REAL NOT NULL,
    matched_kp INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_inspections_created_at ON inspections(created_at);
CREATE INDEX IF NOT EXISTS idx_inspections_stage_verdict ON inspections(stage, verdict);
CREATE INDEX IF NOT EXISTS idx_verifications_item_id ON passport_verifications(item_id);
"""


class InvalidItemIdError(ValueError):
    """Raised when item_id fails regex or path containment validation."""


class DuplicatePassportError(ValueError):
    """Raised when registering an already existing passport item_id."""


class PassportNotFoundError(KeyError):
    """Raised when looking up or verifying an unregistered passport item_id."""


def _validate_item_id(item_id: str) -> str:
    if not isinstance(item_id, str) or not ITEM_ID_RE.fullmatch(item_id):
        raise InvalidItemIdError(
            "item_id must match ^[A-Za-z0-9_-]{3,64}$ without whitespace or path separators."
        )
    return item_id


def _safe_child_path(base_dir: Path, filename: str) -> Path:
    base_resolved = base_dir.resolve()
    target = (base_resolved / filename).resolve()
    if not target.is_relative_to(base_resolved):
        raise InvalidItemIdError(f"Path traversal rejected for filename: {filename}")
    return target


class CeramStorage:
    """Thread-safe SQLite WAL repository for CeramScan inspections and passports."""

    def __init__(
        self,
        db_path: Path,
        qr_dir: Path,
        passport_ref_dir: Path | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.qr_dir = Path(qr_dir)
        self.passport_ref_dir = (
            Path(passport_ref_dir)
            if passport_ref_dir is not None
            else self.qr_dir.parent / "passports"
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.qr_dir.mkdir(parents=True, exist_ok=True)
        self.passport_ref_dir.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_schema(self) -> None:
        with contextlib.closing(self._connect()) as conn:
            conn.executescript(_SCHEMA_SQL)
            conn.commit()

    def health_check(self) -> bool:
        """Verify database connection responds to SELECT 1."""
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute("SELECT 1;").fetchone()
            return bool(row is not None and row[0] == 1)

    def save_inspection(
        self,
        result: InspectionResult,
        overlay_path: str,
        item_id: str | None = None,
        shift: str = "S1",
    ) -> int:
        """Persist an OTK inspection result and return its row ID."""
        norm_item_id: str | None = None
        if item_id is not None:
            if not isinstance(item_id, str):
                raise InvalidItemIdError(
                    "item_id must match ^[A-Za-z0-9_-]{3,64}$ without whitespace or path separators."
                )
            if item_id.strip() != "":
                norm_item_id = _validate_item_id(item_id)

        severity_rank = {"critical": 0, "rework": 1, "minor": 2}
        primary_defect = (
            min(
                result.defects,
                key=lambda d: (severity_rank.get(d.severity, 3), -d.area_px, -d.confidence),
            ).defect_class.value
            if result.defects
            else None
        )
        serialized_defects = [
            {
                "defect_class": d.defect_class.value,
                "gost_name_ru": d.gost_name_ru,
                "severity": d.severity,
                "bbox_xywh": list(d.bbox_xywh),
                "area_px": d.area_px,
                "area_mm2": d.area_mm2,
                "length_mm": d.length_mm,
                "confidence": d.confidence,
                "zone": d.zone,
                "source": d.source,
            }
            for d in result.defects
        ]
        created_at = datetime.now(timezone.utc).isoformat()

        with contextlib.closing(self._connect()) as conn:
            cur = conn.execute(
                """
                INSERT INTO inspections (
                    item_id, stage, shift, verdict, gost_grade, model_mode,
                    defect_count, primary_defect, contour_deviation_pct, glare_ratio,
                    latency_ms, defects_json, gost_rules_json, overlay_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    norm_item_id,
                    result.stage,
                    shift.strip() or "S1",
                    result.verdict,
                    result.gost_grade,
                    result.model_mode,
                    result.defect_count,
                    primary_defect,
                    float(result.contour_deviation_pct),
                    float(result.glare_ratio),
                    float(result.latency_ms),
                    json.dumps(serialized_defects, ensure_ascii=False),
                    json.dumps(result.gost_rules_triggered, ensure_ascii=False),
                    overlay_path,
                    created_at,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_inspections(
        self,
        limit: int = 25,
        stage: StageType | None = None,
        verdict: VerdictType | None = None,
    ) -> list[dict[str, Any]]:
        """Query recent OTK inspections with optional stage/verdict filtering."""
        if not (1 <= limit <= 200):
            raise ValueError("limit must be between 1 and 200.")

        predicates: list[str] = []
        params: list[Any] = []
        if stage is not None and stage != "":
            if stage not in VALID_STAGES:
                raise ValueError(f"Unsupported stage filter: {stage}")
            predicates.append("stage = ?")
            params.append(stage)
        if verdict is not None and verdict != "":
            if verdict not in VALID_VERDICTS:
                raise ValueError(f"Unsupported verdict filter: {verdict}")
            predicates.append("verdict = ?")
            params.append(verdict)

        where_clause = f"WHERE {' AND '.join(predicates)}" if predicates else ""
        sql = f"SELECT * FROM inspections {where_clause} ORDER BY id DESC LIMIT ?;"
        params.append(limit)

        with contextlib.closing(self._connect()) as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()

        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "id": int(row["id"]),
                    "item_id": row["item_id"],
                    "stage": row["stage"],
                    "shift": row["shift"],
                    "verdict": row["verdict"],
                    "gost_grade": row["gost_grade"],
                    "model_mode": row["model_mode"],
                    "defect_count": int(row["defect_count"]),
                    "primary_defect": row["primary_defect"],
                    "contour_deviation_pct": float(row["contour_deviation_pct"]),
                    "glare_ratio": float(row["glare_ratio"]),
                    "latency_ms": float(row["latency_ms"]),
                    "defects": json.loads(row["defects_json"]),
                    "gost_rules": json.loads(row["gost_rules_json"]),
                    "overlay_path": row["overlay_path"],
                    "created_at": row["created_at"],
                }
            )
        return out

    def get_analytics(
        self,
        limit: int = 25,
        stage: StageType | None = None,
        verdict: VerdictType | None = None,
    ) -> dict[str, Any]:
        """Return global OTK/289-FZ aggregates and filtered recent inspections."""
        recent = self.list_inspections(limit=limit, stage=stage, verdict=verdict)

        with contextlib.closing(self._connect()) as conn:
            insp_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_inspections,
                    COALESCE(SUM(CASE WHEN verdict = 'PASS' THEN 1 ELSE 0 END), 0) AS pass_count,
                    COALESCE(SUM(CASE WHEN verdict = 'REWORK' THEN 1 ELSE 0 END), 0) AS rework_count,
                    COALESCE(SUM(CASE WHEN verdict = 'REJECT' THEN 1 ELSE 0 END), 0) AS reject_count,
                    COALESCE(AVG(latency_ms), 0.0) AS avg_latency_ms
                FROM inspections;
                """
            ).fetchone()

            pass_row = conn.execute(
                "SELECT COUNT(*) AS total_passports FROM passports;"
            ).fetchone()
            ver_row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_verifications,
                    COALESCE(
                        SUM(CASE WHEN verdict != 'Подлинник' THEN 1 ELSE 0 END),
                        0
                    ) AS counterfeit_flags
                FROM passport_verifications;
                """
            ).fetchone()

            defect_rows = conn.execute(
                """
                SELECT
                    json_extract(j.value, '$.defect_class') AS defect_class,
                    COUNT(*) AS cnt
                FROM inspections AS i, json_each(i.defects_json) AS j
                GROUP BY defect_class
                ORDER BY cnt DESC, defect_class ASC;
                """
            ).fetchall()

            stage_rows = conn.execute(
                """
                SELECT
                    stage,
                    COUNT(*) AS total,
                    COALESCE(SUM(CASE WHEN verdict = 'PASS' THEN 1 ELSE 0 END), 0) AS pass_count,
                    COALESCE(SUM(CASE WHEN verdict = 'REWORK' THEN 1 ELSE 0 END), 0) AS rework_count,
                    COALESCE(SUM(CASE WHEN verdict = 'REJECT' THEN 1 ELSE 0 END), 0) AS reject_count
                FROM inspections
                GROUP BY stage
                ORDER BY stage ASC;
                """
            ).fetchall()

            shift_rows = conn.execute(
                """
                SELECT
                    shift,
                    COUNT(*) AS total,
                    COALESCE(SUM(CASE WHEN verdict = 'PASS' THEN 1 ELSE 0 END), 0) AS pass_count,
                    COALESCE(SUM(CASE WHEN verdict = 'REWORK' THEN 1 ELSE 0 END), 0) AS rework_count,
                    COALESCE(SUM(CASE WHEN verdict = 'REJECT' THEN 1 ELSE 0 END), 0) AS reject_count
                FROM inspections
                GROUP BY shift
                ORDER BY shift ASC;
                """
            ).fetchall()

        total_insp = int(insp_row["total_inspections"])
        pass_count = int(insp_row["pass_count"])
        pass_rate = round((pass_count / total_insp) * 100.0, 2) if total_insp > 0 else 0.0

        return {
            "summary": {
                "total_inspections": total_insp,
                "pass_count": pass_count,
                "rework_count": int(insp_row["rework_count"]),
                "reject_count": int(insp_row["reject_count"]),
                "pass_rate_pct": pass_rate,
                "avg_latency_ms": round(float(insp_row["avg_latency_ms"]), 2),
                "total_passports": int(pass_row["total_passports"]),
                "total_verifications": int(ver_row["total_verifications"]),
                "counterfeit_flags": int(ver_row["counterfeit_flags"]),
            },
            "by_defect_class": [
                {"defect_class": row["defect_class"], "count": int(row["cnt"])}
                for row in defect_rows
                if row["defect_class"]
            ],
            "by_stage": [
                {
                    "stage": row["stage"],
                    "total": int(row["total"]),
                    "pass_count": int(row["pass_count"]),
                    "rework_count": int(row["rework_count"]),
                    "reject_count": int(row["reject_count"]),
                }
                for row in stage_rows
            ],
            "by_shift": [
                {
                    "shift": row["shift"],
                    "total": int(row["total"]),
                    "pass_count": int(row["pass_count"]),
                    "rework_count": int(row["rework_count"]),
                    "reject_count": int(row["reject_count"]),
                }
                for row in shift_rows
            ],
            "recent_inspections": recent,
        }

    def register_passport(
        self,
        signature: FingerprintSignature,
        item_id: str | None = None,
        sku: str = "GZHEL-SKU-01",
        master_name: str = "Мастер Гжель",
        batch_note: str = "",
        stage: StageType = "glazed",
        reference_image_bgr: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Register a 289-FZ digital passport, write QR + ref PNG atomically, and save signature."""
        if stage not in VALID_STAGES:
            raise ValueError(f"Unsupported stage: {stage}")

        if item_id is None:
            date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
            resolved_id = f"GZHEL-{date_tag}-{uuid.uuid4().hex[:6]}"
        elif not isinstance(item_id, str):
            raise InvalidItemIdError(
                "item_id must match ^[A-Za-z0-9_-]{3,64}$ without whitespace or path separators."
            )
        elif item_id.strip() == "":
            date_tag = datetime.now(timezone.utc).strftime("%Y%m%d")
            resolved_id = f"GZHEL-{date_tag}-{uuid.uuid4().hex[:6]}"
        else:
            resolved_id = _validate_item_id(item_id)

        ref_bytes: bytes | None = None
        if reference_image_bgr is not None:
            if (
                not isinstance(reference_image_bgr, np.ndarray)
                or reference_image_bgr.size == 0
                or reference_image_bgr.ndim not in (2, 3)
            ):
                raise ValueError("reference_image_bgr must be a non-empty numpy ndarray.")
            try:
                ok, encoded = cv2.imencode(".png", reference_image_bgr)
            except cv2.error as exc:
                raise ValueError("Invalid reference_image_bgr for PNG encoding.") from exc
            if not ok:
                raise RuntimeError("Failed to encode passport reference PNG.")
            ref_bytes = bytes(encoded)

        qr_file = _safe_child_path(self.qr_dir, f"{resolved_id}.png")
        ref_file = _safe_child_path(self.passport_ref_dir, f"{resolved_id}.png")
        qr_payload = f"CERAMSCAN:{resolved_id}"
        created_at = datetime.now(timezone.utc).isoformat()
        sig_json = json.dumps(asdict(signature), ensure_ascii=False)

        written_files: list[Path] = []
        with contextlib.closing(self._connect()) as conn:
            conn.execute("BEGIN IMMEDIATE;")
            existing = conn.execute(
                "SELECT 1 FROM passports WHERE item_id = ?;",
                (resolved_id,),
            ).fetchone()
            if existing is not None:
                conn.rollback()
                raise DuplicatePassportError(f"Passport '{resolved_id}' already exists.")

            try:
                qr = qrcode.QRCode(box_size=8, border=2)
                qr.add_data(qr_payload)
                qr.make(fit=True)
                qr_img = qr.make_image(fill_color="black", back_color="white")
                qr_img.save(qr_file)
                written_files.append(qr_file)

                ref_path_str = ""
                if ref_bytes is not None:
                    ref_file.write_bytes(ref_bytes)
                    written_files.append(ref_file)
                    ref_path_str = str(ref_file)

                conn.execute(
                    """
                    INSERT INTO passports (
                        item_id, sku, master_name, batch_note, stage,
                        qr_payload, qr_image_path, reference_image_path,
                        signature_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        resolved_id,
                        sku.strip() or "GZHEL-SKU-01",
                        master_name.strip() or "Мастер Гжель",
                        batch_note.strip(),
                        stage,
                        qr_payload,
                        str(qr_file),
                        ref_path_str,
                        sig_json,
                        created_at,
                    ),
                )
                conn.commit()
            except (sqlite3.Error, OSError, ValueError, RuntimeError):
                with contextlib.suppress(sqlite3.Error):
                    conn.rollback()
                for orphan_path in written_files:
                    with contextlib.suppress(OSError):
                        orphan_path.unlink(missing_ok=True)
                raise

        return {
            "item_id": resolved_id,
            "sku": sku.strip() or "GZHEL-SKU-01",
            "master_name": master_name.strip() or "Мастер Гжель",
            "batch_note": batch_note.strip(),
            "stage": stage,
            "qr_payload": qr_payload,
            "qr_image_path": str(qr_file),
            "reference_image_path": ref_path_str,
            "orb_count": signature.orb_count,
            "cobalt_ratio": round(signature.cobalt_ratio, 4),
            "created_at": created_at,
        }

    def get_passport(self, item_id: str) -> dict[str, Any]:
        """Fetch registered passport metadata and reconstructed FingerprintSignature."""
        valid_id = _validate_item_id(item_id)
        with contextlib.closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM passports WHERE item_id = ?;",
                (valid_id,),
            ).fetchone()
        if row is None:
            raise PassportNotFoundError(f"Passport '{valid_id}' not found.")

        sig_dict = json.loads(row["signature_json"])
        signature = FingerprintSignature(**sig_dict)
        return {
            "item_id": row["item_id"],
            "sku": row["sku"],
            "master_name": row["master_name"],
            "batch_note": row["batch_note"],
            "stage": row["stage"],
            "qr_payload": row["qr_payload"],
            "qr_image_path": row["qr_image_path"],
            "reference_image_path": row["reference_image_path"],
            "created_at": row["created_at"],
            "signature": signature,
        }

    def record_verification(self, item_id: str, result: VerificationResult) -> int:
        """Persist a 289-FZ verification event linked via foreign key to passports."""
        valid_id = _validate_item_id(item_id)
        created_at = datetime.now(timezone.utc).isoformat()
        with contextlib.closing(self._connect()) as conn:
            try:
                cur = conn.execute(
                    """
                    INSERT INTO passport_verifications (
                        item_id, match_percent, orb_score, phash_score,
                        color_score, matched_kp, verdict, latency_ms, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        valid_id,
                        float(result.match_percent),
                        float(result.orb_score),
                        float(result.phash_score),
                        float(result.color_score),
                        int(result.matched_kp),
                        result.verdict,
                        float(result.latency_ms),
                        created_at,
                    ),
                )
                conn.commit()
                return int(cur.lastrowid)
            except sqlite3.IntegrityError as exc:
                conn.rollback()
                raise PassportNotFoundError(f"Passport '{valid_id}' not found.") from exc
