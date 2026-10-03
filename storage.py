"""storage.py - SQLite storage and export utilities for 3E QC PoC.

Manages inspection history, user feedback, metrics calculation,
CSV export, and YOLO active-learning dataset export.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

import config

DB_PATH = config.DATA_DIR / "qc.db"
IMAGES_DIR = config.DATA_DIR / "images"
MASKS_DIR = config.DATA_DIR / "masks"
BORESIGHT_DIR = config.DATA_DIR / "boresight"
MTF_DIR = config.DATA_DIR / "mtf"
TORQUE_DIR = config.DATA_DIR / "torque"
EXPORTS_DIR = config.DATA_DIR / "exports"


def get_db() -> sqlite3.Connection:
    """Returns a SQLite connection configured with Row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initializes SQLite database tables and storage directories."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    BORESIGHT_DIR.mkdir(parents=True, exist_ok=True)
    MTF_DIR.mkdir(parents=True, exist_ok=True)
    TORQUE_DIR.mkdir(parents=True, exist_ok=True)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                inspector TEXT,
                product_group TEXT,
                serial_no TEXT,
                image_path TEXT,
                model_versions TEXT,
                engine_outputs TEXT,
                ai_decision TEXT,
                ai_defect_type TEXT,
                confidence REAL,
                defect_score REAL,
                human_decision TEXT,
                human_defect_type TEXT,
                note TEXT,
                agreement_flag INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inspection_id INTEGER NOT NULL,
                label TEXT NOT NULL,
                x1 REAL NOT NULL,
                y1 REAL NOT NULL,
                x2 REAL NOT NULL,
                y2 REAL NOT NULL,
                mask_path TEXT,
                source TEXT NOT NULL,
                FOREIGN KEY (inspection_id) REFERENCES inspections (id)
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS boresight_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                inspector TEXT,
                serial_no TEXT,
                stage TEXT,
                channel TEXT,
                dx_px REAL,
                dy_px REAL,
                err_mrad REAL,
                tolerance_mrad REAL,
                result TEXT,
                image_path TEXT,
                inter_channel_mrad REAL,
                drift_mrad REAL,
                params TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mtf_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                inspector TEXT,
                serial_no TEXT,
                channel TEXT,
                mtf50_cyc_px REAL,
                mtf50_lpmm REAL,
                mtf_nyquist REAL,
                edge_angle_deg REAL,
                spec_mtf50 REAL,
                result TEXT,
                image_path TEXT,
                params TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS torque_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                inspector TEXT,
                serial_no TEXT,
                stage TEXT,
                paint_color TEXT,
                angle_deg REAL,
                offset_px REAL,
                tolerance_deg REAL,
                result TEXT,
                image_path TEXT,
                params TEXT
            );
            """
        )
        conn.commit()


def compute_agreement(
    ai_decision: str | None,
    human_decision: str | None,
    ai_defect_type: str | None,
    human_defect_type: str | None,
) -> int | None:
    """Calculates AI-Human agreement flag according to specifications:

    - AI REVIEW rows -> NULL (None)
    - ai ACCEPT and human ACCEPT -> 1
    - ai REJECT and human REJECT and types equal -> 1
    - all other cases -> 0
    """
    ai_dec = (ai_decision or "").strip().upper()
    hum_dec = (human_decision or "").strip().upper()

    if ai_dec == "REVIEW":
        return None

    if ai_dec == "ACCEPT" and hum_dec == "ACCEPT":
        return 1

    if ai_dec == "REJECT" and hum_dec == "REJECT":
        ai_t = (ai_defect_type or "").strip().lower()
        hum_t = (human_defect_type or "").strip().lower()
        return 1 if ai_t == hum_t else 0

    return 0


def _clean_for_json(obj: Any) -> Any:
    """Recursively converts objects to JSON-serializable types,

    stripping PIL Images and bytes.
    """
    if isinstance(obj, (Image.Image, bytes)):
        return None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (list, tuple)):
        return [_clean_for_json(x) for x in obj if not isinstance(x, (Image.Image, bytes))]
    if isinstance(obj, dict):
        return {
            str(k): _clean_for_json(v)
            for k, v in obj.items()
            if not isinstance(v, (Image.Image, bytes))
        }
    if isinstance(obj, Path):
        return str(obj)
    return str(obj) if obj is not None else None


def save_inspection(
    inspector: str,
    product_group: str,
    serial_no: str,
    image: Image.Image | str | Path | None,
    pipeline_result: dict[str, Any] | None = None,
    human_decision: str = "ACCEPT",
    human_defect_type: str = "Yok",
    note: str = "",
    ai_decision: str | None = None,
    ai_defect_type: str | None = None,
    confidence: float | None = None,
    defect_score: float | None = None,
    model_versions: dict[str, Any] | None = None,
    engine_outputs: dict[str, Any] | None = None,
    operator_labels: list[dict[str, Any]] | None = None,
) -> int:
    """Saves inspection data to SQLite and image to DATA_DIR/images/<id>.png.

    Persists any operator labels and masks to operator_labels table and DATA_DIR/masks/.
    Returns the inserted record ID.
    """
    init_db()

    # Extract or fallback from pipeline_result
    if pipeline_result:
        if ai_decision is None:
            ai_decision = pipeline_result.get("decision", "ACCEPT")
        if ai_defect_type is None:
            ai_defect_type = pipeline_result.get("defect_type", "Yok")
        if confidence is None:
            confidence = float(pipeline_result.get("confidence", 0.0))
        if defect_score is None:
            defect_score = float(pipeline_result.get("defect_score", 0.0))
        if model_versions is None:
            model_versions = pipeline_result.get("model_versions", {})
        if engine_outputs is None:
            engine_outputs = pipeline_result.get("engines", {})

    ai_decision = (ai_decision or "ACCEPT").strip().upper()
    human_decision = (human_decision or "ACCEPT").strip().upper()
    ai_defect_type = (ai_defect_type or "Yok").strip()
    human_defect_type = (human_defect_type or "Yok").strip()
    confidence = float(confidence or 0.0)
    defect_score = float(defect_score or 0.0)

    agreement_flag = compute_agreement(
        ai_decision=ai_decision,
        human_decision=human_decision,
        ai_defect_type=ai_defect_type,
        human_defect_type=human_defect_type,
    )

    clean_model_versions = _clean_for_json(model_versions or {})
    clean_engine_outputs = _clean_for_json(engine_outputs or {})

    ts = datetime.now(timezone.utc).isoformat()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO inspections (
                ts, inspector, product_group, serial_no, image_path,
                model_versions, engine_outputs, ai_decision, ai_defect_type,
                confidence, defect_score, human_decision, human_defect_type,
                note, agreement_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                inspector or "Muayene Uzmanı",
                product_group or "Optik lens grubu",
                serial_no or f"SN-{int(datetime.now().timestamp())}",
                "",  # updated right below once ID is obtained
                json.dumps(clean_model_versions, ensure_ascii=False),
                json.dumps(clean_engine_outputs, ensure_ascii=False),
                ai_decision,
                ai_defect_type,
                confidence,
                defect_score,
                human_decision,
                human_defect_type,
                note,
                agreement_flag,
            ),
        )
        record_id = cursor.lastrowid
        assert record_id is not None

        # Save image as PNG
        img_dest = IMAGES_DIR / f"{record_id}.png"
        if isinstance(image, Image.Image):
            rgb_img = image.convert("RGB")
            rgb_img.save(img_dest, format="PNG")
        elif isinstance(image, (str, Path)) and Path(image).exists():
            loaded = Image.open(image).convert("RGB")
            loaded.save(img_dest, format="PNG")
        else:
            # Fallback placeholder image
            placeholder = Image.new("RGB", (512, 512), color=(40, 44, 52))
            draw = ImageDraw.Draw(placeholder)
            draw.text((30, 240), f"Inspection #{record_id}", fill=(220, 220, 220))
            placeholder.save(img_dest, format="PNG")

        cursor.execute(
            "UPDATE inspections SET image_path = ? WHERE id = ?",
            (str(img_dest), record_id),
        )

        # Persist operator labels if provided
        if operator_labels:
            for idx, op in enumerate(operator_labels):
                lab = op.get("label") or human_defect_type or "Diğer"
                box = op.get("box", [])
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                else:
                    x1 = float(op.get("x1", op.get("xmin", 0)))
                    y1 = float(op.get("y1", op.get("ymin", 0)))
                    x2 = float(op.get("x2", op.get("xmax", 0)))
                    y2 = float(op.get("y2", op.get("ymax", 0)))

                source = op.get("source", "operator")
                mask_val = op.get("mask")
                mask_path_str: str | None = None

                if mask_val is not None:
                    if isinstance(mask_val, (str, Path)) and Path(mask_val).exists():
                        mask_path_str = str(mask_val)
                    elif isinstance(mask_val, np.ndarray):
                        mask_path = MASKS_DIR / f"{record_id}_{idx}.png"
                        arr = (
                            (mask_val > 0).astype(np.uint8) * 255
                            if np.max(mask_val) <= 1
                            else mask_val.astype(np.uint8)
                        )
                        Image.fromarray(arr).save(mask_path, format="PNG")
                        mask_path_str = str(mask_path)
                    elif isinstance(mask_val, Image.Image):
                        mask_path = MASKS_DIR / f"{record_id}_{idx}.png"
                        mask_val.save(mask_path, format="PNG")
                        mask_path_str = str(mask_path)

                cursor.execute(
                    """
                    INSERT INTO operator_labels (
                        inspection_id, label, x1, y1, x2, y2, mask_path, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (record_id, lab, x1, y1, x2, y2, mask_path_str, source),
                )

        conn.commit()

    return record_id


def save_operator_labels(
    inspection_id: int,
    labels: list[dict[str, Any]],
) -> list[int]:
    """Persists operator labels and masks for an existing inspection record."""
    init_db()
    ids: list[int] = []
    with get_db() as conn:
        cursor = conn.cursor()
        for idx, op in enumerate(labels):
            lab = op.get("label", "Diğer")
            box = op.get("box", [])
            if isinstance(box, (list, tuple)) and len(box) >= 4:
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            else:
                x1 = float(op.get("x1", op.get("xmin", 0)))
                y1 = float(op.get("y1", op.get("ymin", 0)))
                x2 = float(op.get("x2", op.get("xmax", 0)))
                y2 = float(op.get("y2", op.get("ymax", 0)))

            source = op.get("source", "operator")
            mask_val = op.get("mask")
            mask_path_str: str | None = None

            if mask_val is not None:
                if isinstance(mask_val, (str, Path)) and Path(mask_val).exists():
                    mask_path_str = str(mask_val)
                elif isinstance(mask_val, np.ndarray):
                    mask_path = MASKS_DIR / f"{inspection_id}_{idx}.png"
                    arr = (
                        (mask_val > 0).astype(np.uint8) * 255
                        if np.max(mask_val) <= 1
                        else mask_val.astype(np.uint8)
                    )
                    Image.fromarray(arr).save(mask_path, format="PNG")
                    mask_path_str = str(mask_path)
                elif isinstance(mask_val, Image.Image):
                    mask_path = MASKS_DIR / f"{inspection_id}_{idx}.png"
                    mask_val.save(mask_path, format="PNG")
                    mask_path_str = str(mask_path)

            cursor.execute(
                """
                INSERT INTO operator_labels (
                    inspection_id, label, x1, y1, x2, y2, mask_path, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (inspection_id, lab, x1, y1, x2, y2, mask_path_str, source),
            )
            assert cursor.lastrowid is not None
            ids.append(cursor.lastrowid)
        conn.commit()
    return ids


def get_operator_labels(inspection_id: int) -> list[dict[str, Any]]:
    """Fetches operator labels for an inspection."""
    init_db()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM operator_labels WHERE inspection_id = ? ORDER BY id ASC",
            (inspection_id,),
        )
        return [dict(r) for r in cursor.fetchall()]


def list_inspections(limit: int = 100) -> list[dict[str, Any]]:
    """Returns the most recent inspections as a list of dictionaries (pandas-free)."""
    init_db()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM inspections
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cursor.fetchall()

    results: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        # Parse JSON fields safely
        try:
            d["model_versions"] = json.loads(d["model_versions"]) if d["model_versions"] else {}
        except Exception:
            pass
        try:
            d["engine_outputs"] = json.loads(d["engine_outputs"]) if d["engine_outputs"] else {}
        except Exception:
            pass
        results.append(d)

    return results


def save_boresight(
    inspector: str,
    serial_no: str,
    stage: str,
    channel: str,
    dx_px: float,
    dy_px: float,
    err_mrad: float,
    tolerance_mrad: float,
    result: str,
    image: Image.Image | str | Path | None = None,
    inter_channel_mrad: float | None = None,
    drift_mrad: float | None = None,
    params: dict[str, Any] | None = None,
) -> int:
    """Saves a post-assembly boresight optical axis test record to SQLite and image to DATA_DIR/boresight/<id>.png.

    Returns the inserted record ID.
    """
    init_db()
    ts = datetime.now(timezone.utc).isoformat()
    clean_params = _clean_for_json(params or {})

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO boresight_tests (
                ts, inspector, serial_no, stage, channel,
                dx_px, dy_px, err_mrad, tolerance_mrad, result,
                image_path, inter_channel_mrad, drift_mrad, params
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                inspector or "Muayene Uzmanı",
                serial_no or f"SN-{int(datetime.now().timestamp())}",
                stage or "son test",
                channel or "Görünür",
                float(dx_px),
                float(dy_px),
                float(err_mrad),
                float(tolerance_mrad),
                result or ("PASS" if float(err_mrad) <= float(tolerance_mrad) else "FAIL"),
                "",
                float(inter_channel_mrad) if inter_channel_mrad is not None else None,
                float(drift_mrad) if drift_mrad is not None else None,
                json.dumps(clean_params, ensure_ascii=False),
            ),
        )
        record_id = cursor.lastrowid
        assert record_id is not None

        if image is not None:
            img_dest = BORESIGHT_DIR / f"{record_id}.png"
            if isinstance(image, Image.Image):
                image.convert("RGB").save(img_dest, format="PNG")
            elif isinstance(image, (str, Path)) and Path(image).exists():
                Image.open(image).convert("RGB").save(img_dest, format="PNG")
            cursor.execute(
                "UPDATE boresight_tests SET image_path = ? WHERE id = ?",
                (str(img_dest), record_id),
            )

        conn.commit()

    return record_id


def list_boresight(
    serial_no: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Returns the most recent boresight test records, optionally filtered by serial number."""
    init_db()
    with get_db() as conn:
        cursor = conn.cursor()
        if serial_no:
            cursor.execute(
                """
                SELECT * FROM boresight_tests
                WHERE serial_no = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (serial_no, limit),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM boresight_tests
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
        rows = cursor.fetchall()

    results: list[dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        try:
            d["params"] = json.loads(d["params"]) if d.get("params") else {}
        except Exception:
            pass
        results.append(d)
    return results


def _f(v: Any) -> float | None:
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def _save_image(image: Any, dest: Path) -> bool:
    try:
        if isinstance(image, Image.Image):
            image.convert("RGB").save(dest, format="PNG")
            return True
        if isinstance(image, (str, Path)) and Path(image).exists():
            Image.open(image).convert("RGB").save(dest, format="PNG")
            return True
    except Exception:
        pass
    return False


def _normalize_result(result: str | None) -> str:
    r = (result or "").strip().upper()
    if r in ("PASS", "GEÇTİ", "GECTI", "GEÇTI"):
        return "PASS"
    if r in ("FAIL", "KALDI", "RET", "REJECT"):
        return "FAIL"
    return "UNCERTAIN"


def save_mtf(
    inspector: str,
    serial_no: str,
    channel: str,
    mtf50_cyc_px: float | None,
    mtf50_lpmm: float | None,
    mtf_nyquist: float | None,
    edge_angle_deg: float | None,
    spec_mtf50: float | None,
    result: str,
    image: Any = None,
    params: dict[str, Any] | None = None,
) -> int:
    """Saves an MTF (slanted-edge) test record. Returns the record ID."""
    init_db()
    ts = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO mtf_tests (ts, inspector, serial_no, channel, mtf50_cyc_px, mtf50_lpmm,
                mtf_nyquist, edge_angle_deg, spec_mtf50, result, image_path, params)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                inspector or "Muayene Uzmanı",
                serial_no or f"SN-{int(datetime.now().timestamp())}",
                channel or "Görünür",
                _f(mtf50_cyc_px), _f(mtf50_lpmm), _f(mtf_nyquist), _f(edge_angle_deg), _f(spec_mtf50),
                _normalize_result(result),
                "",
                json.dumps(_clean_for_json(params or {}), ensure_ascii=False),
            ),
        )
        rid = cur.lastrowid
        assert rid is not None
        if image is not None:
            dest = MTF_DIR / f"{rid}.png"
            if _save_image(image, dest):
                cur.execute("UPDATE mtf_tests SET image_path = ? WHERE id = ?", (str(dest), rid))
        conn.commit()
    return rid


def save_torque(
    inspector: str,
    serial_no: str,
    stage: str,
    paint_color: str,
    angle_deg: float | None,
    offset_px: float | None,
    tolerance_deg: float | None,
    result: str,
    image: Any = None,
    params: dict[str, Any] | None = None,
) -> int:
    """Saves a torque-mark test record (GEÇTİ/KALDI/BELİRSİZ -> PASS/FAIL/UNCERTAIN). Returns the record ID."""
    init_db()
    ts = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO torque_tests (ts, inspector, serial_no, stage, paint_color, angle_deg,
                offset_px, tolerance_deg, result, image_path, params)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                inspector or "Muayene Uzmanı",
                serial_no or f"SN-{int(datetime.now().timestamp())}",
                stage or "son test",
                paint_color or "auto",
                _f(angle_deg), _f(offset_px), _f(tolerance_deg),
                _normalize_result(result),
                "",
                json.dumps(_clean_for_json(params or {}), ensure_ascii=False),
            ),
        )
        rid = cur.lastrowid
        assert rid is not None
        if image is not None:
            dest = TORQUE_DIR / f"{rid}.png"
            if _save_image(image, dest):
                cur.execute("UPDATE torque_tests SET image_path = ? WHERE id = ?", (str(dest), rid))
        conn.commit()
    return rid


def _list_table(table: str, serial_no: str | None, limit: int) -> list[dict[str, Any]]:
    init_db()
    with get_db() as conn:
        cur = conn.cursor()
        if serial_no:
            cur.execute(f"SELECT * FROM {table} WHERE serial_no = ? ORDER BY id DESC LIMIT ?", (serial_no, limit))
        else:
            cur.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
        rows = [dict(r) for r in cur.fetchall()]
    for d in rows:
        try:
            d["params"] = json.loads(d["params"]) if d.get("params") else {}
        except Exception:
            pass
    return rows


def list_mtf(serial_no: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Returns recent MTF test records, optionally filtered by serial number."""
    return _list_table("mtf_tests", serial_no, limit)


def list_torque(serial_no: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Returns recent torque-mark test records, optionally filtered by serial number."""
    return _list_table("torque_tests", serial_no, limit)


def _export_table_csv(table: str, target: Path) -> Path:
    init_db()
    target.parent.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table} ORDER BY id ASC")
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    with open(target, mode="w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(list(r))
    return target


def export_mtf_csv(dest_path: Path | str | None = None) -> Path:
    return _export_table_csv("mtf_tests", Path(dest_path) if dest_path else EXPORTS_DIR / "mtf_tests.csv")


def export_torque_csv(dest_path: Path | str | None = None) -> Path:
    return _export_table_csv("torque_tests", Path(dest_path) if dest_path else EXPORTS_DIR / "torque_tests.csv")


def export_post_production_zip(dest_path: Path | str | None = None) -> Path:
    """Bundles boresight, MTF and torque-mark CSVs into one ZIP."""
    target = Path(dest_path) if dest_path else EXPORTS_DIR / "post_production_tests.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    files = {
        "boresight_tests.csv": export_boresight_csv(),
        "mtf_tests.csv": export_mtf_csv(),
        "torque_tests.csv": export_torque_csv(),
    }
    with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        for arc, p in files.items():
            z.write(p, arcname=arc)
    return target


def stats() -> dict[str, Any]:
    """Computes summary statistics for the dashboard:

    - total inspections
    - accept / reject / review counts
    - human review rate (%)
    - ai-human agreement rate (%)
    - defect type distribution
    """
    init_db()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM inspections")
        total = cursor.fetchone()[0]

        if total == 0:
            return {
                "total": 0,
                "ai_accept": 0,
                "ai_reject": 0,
                "ai_review": 0,
                "human_accept": 0,
                "human_reject": 0,
                "human_review_rate": 0.0,
                "agreement_pct": 0.0,
                "defect_distribution": {d: 0 for d in config.DEFECT_TYPES if d != "Yok"},
            }

        cursor.execute("SELECT ai_decision, COUNT(*) FROM inspections GROUP BY ai_decision")
        ai_counts = dict(cursor.fetchall())

        cursor.execute("SELECT human_decision, COUNT(*) FROM inspections GROUP BY human_decision")
        human_counts = dict(cursor.fetchall())

        cursor.execute("SELECT COUNT(*) FROM inspections WHERE agreement_flag IS NOT NULL")
        evaluated_agreement = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM inspections WHERE agreement_flag = 1")
        agreed_count = cursor.fetchone()[0]

        agreement_pct = (
            round((agreed_count / evaluated_agreement * 100.0), 1)
            if evaluated_agreement > 0
            else 0.0
        )

        ai_review_count = ai_counts.get("REVIEW", 0)
        human_review_rate = round((ai_review_count / total * 100.0), 1)

        # Defect distribution (based on human recorded defects when rejected or defect noted)
        cursor.execute(
            """
            SELECT human_defect_type, COUNT(*)
            FROM inspections
            WHERE human_defect_type IS NOT NULL AND human_defect_type != 'Yok'
            GROUP BY human_defect_type
            """
        )
        defects_raw = dict(cursor.fetchall())

        defect_distribution = {}
        for d in config.DEFECT_TYPES:
            if d != "Yok":
                defect_distribution[d] = defects_raw.get(d, 0)

        for k, v in defects_raw.items():
            if k not in defect_distribution:
                defect_distribution[k] = v

        # Sistemin kaçırıp operatörün işaretlediği kusur:
        # Inspections where AI decision was ACCEPT (or detector had no boxes) but operator added >=1 label.
        cursor.execute(
            """
            SELECT DISTINCT i.id, i.ai_decision, i.engine_outputs
            FROM inspections i
            JOIN operator_labels ol ON i.id = ol.inspection_id
            WHERE ol.source = 'operator'
            """
        )
        missed_rows = cursor.fetchall()
        missed_by_ai = 0
        for mr in missed_rows:
            ai_dec = (mr["ai_decision"] or "").strip().upper()
            if ai_dec == "ACCEPT":
                missed_by_ai += 1
            else:
                try:
                    eng = json.loads(mr["engine_outputs"]) if mr["engine_outputs"] else {}
                except Exception:
                    eng = {}
                dets = eng.get("detector", {}).get("detections", [])
                if not dets:
                    missed_by_ai += 1

        return {
            "total": total,
            "ai_accept": ai_counts.get("ACCEPT", 0),
            "ai_reject": ai_counts.get("REJECT", 0),
            "ai_review": ai_review_count,
            "human_accept": human_counts.get("ACCEPT", 0),
            "human_reject": human_counts.get("REJECT", 0),
            "human_review_rate": human_review_rate,
            "agreement_pct": agreement_pct,
            "defect_distribution": defect_distribution,
            "missed_by_ai": missed_by_ai,
        }


def export_boresight_csv(dest_path: Path | str | None = None) -> Path:
    """Exports all boresight test records to a CSV file.

    Returns the Path to the generated CSV.
    """
    init_db()
    if dest_path is None:
        target = EXPORTS_DIR / "boresight_tests.csv"
    else:
        target = Path(dest_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM boresight_tests ORDER BY id ASC")
        rows = cursor.fetchall()
        column_names = [d[0] for d in cursor.description]

    with open(target, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        for row in rows:
            writer.writerow(list(row))

    return target


def export_csv(dest_path: Path | str | None = None) -> Path:
    """Exports all inspections to a CSV file.

    Returns the Path to the generated CSV.
    Also ensures boresight_tests.csv is updated in EXPORTS_DIR.
    """
    init_db()
    if dest_path is None:
        target = EXPORTS_DIR / "inspections.csv"
    else:
        target = Path(dest_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inspections ORDER BY id ASC")
        rows = cursor.fetchall()
        column_names = [d[0] for d in cursor.description]

    with open(target, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        for row in rows:
            writer.writerow(list(row))

    # Also generate boresight_tests.csv alongside inspections.csv
    try:
        export_boresight_csv(EXPORTS_DIR / "boresight_tests.csv")
        export_mtf_csv()
        export_torque_csv()
    except Exception:
        pass

    return target


def export_csv_bundle(dest_path: Path | str | None = None) -> Path:
    """Bundles inspections.csv and boresight_tests.csv into a single ZIP archive."""
    init_db()
    if dest_path is None:
        target = EXPORTS_DIR / "qc_audit_logs.zip"
    else:
        target = Path(dest_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    insp_csv = export_csv()
    bore_csv = export_boresight_csv()
    mtf_csv = export_mtf_csv()
    torque_csv = export_torque_csv()

    with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_f:
        if insp_csv.exists():
            zip_f.write(insp_csv, arcname="inspections.csv")
        if bore_csv.exists():
            zip_f.write(bore_csv, arcname="boresight_tests.csv")
        zip_f.write(mtf_csv, arcname="mtf_tests.csv")
        zip_f.write(torque_csv, arcname="torque_tests.csv")

    return target


def export_yolo_zip(dest_path: Path | str | None = None) -> Path:
    """Generates an active-learning dataset ZIP archive containing:

    - images/ (inspections with detector bounding boxes or operator labels)
    - labels_detect/ (YOLO bbox txt: <class_id> <cx> <cy> <w> <h>)
    - labels_segment/ (YOLO-seg polygons: <class_id> <x1> <y1> <x2> <y2> ...)
    - labels/ (copy of labels_detect for backwards compatibility)
    - classes.txt (DEFECT_TYPES taxonomy)
    - README.md (instructions for retraining)

    Prefers operator labels over model detections when present.
    Returns the Path to the generated ZIP file.
    """
    init_db()
    if dest_path is None:
        target = EXPORTS_DIR / "yolo_dataset.zip"
    else:
        target = Path(dest_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, image_path, engine_outputs, human_decision, ai_decision,
                   human_defect_type, ai_defect_type
            FROM inspections
            WHERE human_decision = 'REJECT' OR ai_decision = 'REJECT'
               OR id IN (SELECT DISTINCT inspection_id FROM operator_labels)
            ORDER BY id ASC
            """
        )
        rows = cursor.fetchall()

        with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_f:
            # 1. classes.txt
            classes_content = "\n".join(config.DEFECT_TYPES) + "\n"
            zip_f.writestr("classes.txt", classes_content)

            # 2. README.md explaining the active learning dataset
            readme_content = (
                "# 3E Kalite Kontrol — Aktif Öğrenme (Active Learning) Veri Paketi\n\n"
                "Bu paket, görsel kalite kontrol denetiminde reddedilen numunelerin\n"
                "ve operatör tarafından etiketlenen/düzeltilen kusurların aktif öğrenme verilerini içerir.\n\n"
                "Öncelik Kuralı:\n"
                "- Operatör etiketleri mevcut olduğunda model tahminlerine önceliklidir.\n\n"
                "Dizin Yapısı ve Format:\n"
                "- classes.txt: Kusur sınıfları fihristi (satır numarası = sınıf ID)\n"
                "- images/<id>.png: Muayene parça görseli\n"
                "- labels_detect/<id>.txt: YOLO nesne tespiti formatı: <class_id> <x_center> <y_center> <width> <height>\n"
                "- labels_segment/<id>.txt: YOLO örnek segmentasyonu formatı: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>\n"
                "  (MobileSAM maske konturlarından veya kutu geometrisinden türetilmiş normalize poligonlar)\n"
                "- labels/<id>.txt: labels_detect ile aynı (geriye dönük uyumluluk)\n\n"
                "Kullanım:\n"
                "Bu veri paketi YOLO11 (yolo11n-seg / yolo11s-seg) modellerinin ince ayarında (fine-tuning)\n"
                "kullanılarak sistemin kaçırdığı kusurlarda tespit başarımını artırır.\n"
            )
            zip_f.writestr("README.md", readme_content)

            # 3. Export images and label files
            exported_count = 0
            for r in rows:
                rec_id = r["id"]
                img_path_str = r["image_path"]
                raw_eng = r["engine_outputs"]
                human_t = r["human_defect_type"]
                ai_t = r["ai_defect_type"]

                if not img_path_str or not Path(img_path_str).exists():
                    continue

                try:
                    with Image.open(img_path_str) as img:
                        img_w, img_h = img.size
                    with open(img_path_str, "rb") as img_file:
                        img_bytes = img_file.read()
                except Exception:
                    continue

                # Check for operator labels first
                cursor.execute(
                    "SELECT * FROM operator_labels WHERE inspection_id = ? ORDER BY id ASC",
                    (rec_id,),
                )
                op_rows = cursor.fetchall()

                detect_lines: list[str] = []
                segment_lines: list[str] = []

                if op_rows:
                    for op in op_rows:
                        lab = op["label"]
                        if lab in config.DEFECT_TYPES:
                            cls_id = config.DEFECT_TYPES.index(lab)
                        else:
                            mapped = config.map_class(lab)
                            cls_id = (
                                config.DEFECT_TYPES.index(mapped)
                                if mapped in config.DEFECT_TYPES
                                else config.DEFECT_TYPES.index("Diğer")
                            )

                        x1, y1 = float(op["x1"]), float(op["y1"])
                        x2, y2 = float(op["x2"]), float(op["y2"])
                        xmin, xmax = min(x1, x2), max(x1, x2)
                        ymin, ymax = min(y1, y2), max(y1, y2)

                        # BBox normalized
                        cx = max(0.0, min(1.0, ((xmin + xmax) / 2.0) / img_w))
                        cy = max(0.0, min(1.0, ((ymin + ymax) / 2.0) / img_h))
                        bw = max(0.0, min(1.0, (xmax - xmin) / img_w))
                        bh = max(0.0, min(1.0, (ymax - ymin) / img_h))
                        detect_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                        # Segment polygon
                        mask_path_val = op["mask_path"]
                        poly_str = ""
                        if mask_path_val and Path(mask_path_val).exists():
                            try:
                                mask_img = cv2.imread(str(mask_path_val), cv2.IMREAD_GRAYSCALE)
                                if mask_img is not None:
                                    contours, _ = cv2.findContours(
                                        (mask_img > 0).astype(np.uint8),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                    )
                                    if contours:
                                        largest = max(contours, key=cv2.contourArea)
                                        pts = []
                                        for pt in largest:
                                            nx = max(0.0, min(1.0, float(pt[0][0]) / img_w))
                                            ny = max(0.0, min(1.0, float(pt[0][1]) / img_h))
                                            pts.append(f"{nx:.6f} {ny:.6f}")
                                        if len(pts) >= 3:
                                            poly_str = " ".join(pts)
                            except Exception:
                                poly_str = ""

                        if not poly_str:
                            # 4 corners fallback
                            p1 = (max(0.0, min(1.0, xmin / img_w)), max(0.0, min(1.0, ymin / img_h)))
                            p2 = (max(0.0, min(1.0, xmax / img_w)), max(0.0, min(1.0, ymin / img_h)))
                            p3 = (max(0.0, min(1.0, xmax / img_w)), max(0.0, min(1.0, ymax / img_h)))
                            p4 = (max(0.0, min(1.0, xmin / img_w)), max(0.0, min(1.0, ymax / img_h)))
                            poly_str = f"{p1[0]:.6f} {p1[1]:.6f} {p2[0]:.6f} {p2[1]:.6f} {p3[0]:.6f} {p3[1]:.6f} {p4[0]:.6f} {p4[1]:.6f}"

                        segment_lines.append(f"{cls_id} {poly_str}")

                else:
                    # Model detections fallback
                    try:
                        eng_data = json.loads(raw_eng) if raw_eng else {}
                    except Exception:
                        eng_data = {}

                    detector_out = eng_data.get("detector", {})
                    detections = detector_out.get("detections", [])

                    for det in detections:
                        box = det.get("box")
                        if not box or len(box) != 4:
                            continue

                        label_tr = det.get("label_tr") or det.get("label") or human_t or ai_t
                        if label_tr in config.DEFECT_TYPES:
                            cls_id = config.DEFECT_TYPES.index(label_tr)
                        else:
                            mapped = config.map_class(label_tr)
                            cls_id = (
                                config.DEFECT_TYPES.index(mapped)
                                if mapped in config.DEFECT_TYPES
                                else config.DEFECT_TYPES.index("Diğer")
                            )

                        x1, y1 = float(box[0]), float(box[1])
                        x2, y2 = float(box[2]), float(box[3])
                        xmin, xmax = min(x1, x2), max(x1, x2)
                        ymin, ymax = min(y1, y2), max(y1, y2)

                        cx = max(0.0, min(1.0, ((xmin + xmax) / 2.0) / img_w))
                        cy = max(0.0, min(1.0, ((ymin + ymax) / 2.0) / img_h))
                        bw = max(0.0, min(1.0, (xmax - xmin) / img_w))
                        bh = max(0.0, min(1.0, (ymax - ymin) / img_h))

                        detect_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

                        p1 = (max(0.0, min(1.0, xmin / img_w)), max(0.0, min(1.0, ymin / img_h)))
                        p2 = (max(0.0, min(1.0, xmax / img_w)), max(0.0, min(1.0, ymin / img_h)))
                        p3 = (max(0.0, min(1.0, xmax / img_w)), max(0.0, min(1.0, ymax / img_h)))
                        p4 = (max(0.0, min(1.0, xmin / img_w)), max(0.0, min(1.0, ymax / img_h)))
                        poly_str = f"{p1[0]:.6f} {p1[1]:.6f} {p2[0]:.6f} {p2[1]:.6f} {p3[0]:.6f} {p3[1]:.6f} {p4[0]:.6f} {p4[1]:.6f}"
                        segment_lines.append(f"{cls_id} {poly_str}")

                if detect_lines or segment_lines:
                    zip_f.writestr(f"images/{rec_id}.png", img_bytes)
                    if detect_lines:
                        zip_f.writestr(f"labels_detect/{rec_id}.txt", "\n".join(detect_lines) + "\n")
                        zip_f.writestr(f"labels/{rec_id}.txt", "\n".join(detect_lines) + "\n")
                    if segment_lines:
                        zip_f.writestr(f"labels_segment/{rec_id}.txt", "\n".join(segment_lines) + "\n")
                    exported_count += 1

    return target


def seed_demo() -> None:
    """Inserts ~8 realistic historical demo rows if the database is empty.

    Each row is clearly marked with 'demo kaydı'.
    """
    init_db()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM inspections")
        count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM boresight_tests")
        b_count = cursor.fetchone()[0]

    if count > 0 or b_count > 0:
        _seed_post_production()
        return

    demo_records = [
        # Optik lens grubu (metal_nut)
        {
            "inspector": "Ahmet Yılmaz",
            "product_group": "Optik lens grubu",
            "serial_no": "LN-20261001-014",
            "ai_decision": "ACCEPT",
            "ai_defect_type": "Yok",
            "confidence": 0.96,
            "defect_score": 0.05,
            "human_decision": "ACCEPT",
            "human_defect_type": "Yok",
            "note": "demo kaydı: Optik element pürüzsüz, kaplama homojen.",
            "detections": [],
            "anomaly_p": 0.88,
            "color": (28, 40, 51),
        },
        {
            "inspector": "Merve Kaya",
            "product_group": "Optik lens grubu",
            "serial_no": "LN-20261001-029",
            "ai_decision": "REJECT",
            "ai_defect_type": "Yüzey çiziği",
            "confidence": 0.93,
            "defect_score": 0.89,
            "human_decision": "REJECT",
            "human_defect_type": "Yüzey çiziği",
            "note": "demo kaydı: Kenar periferinde 1.8 mm derin yüzey çiziği.",
            "detections": [
                {
                    "label": "scratch",
                    "label_tr": "Yüzey çiziği",
                    "conf": 0.88,
                    "box": [110, 160, 220, 240],
                }
            ],
            "anomaly_p": 0.01,
            "color": (70, 30, 30),
        },
        {
            "inspector": "Ahmet Yılmaz",
            "product_group": "Optik lens grubu",
            "serial_no": "LN-20261002-005",
            "ai_decision": "REVIEW",
            "ai_defect_type": "Kaplama kusuru",
            "confidence": 0.54,
            "defect_score": 0.47,
            "human_decision": "REJECT",
            "human_defect_type": "Kaplama kusuru",
            "note": "demo kaydı: AR kaplama yansımasında bölgesel dalgalanma.",
            "detections": [
                {
                    "label": "color",
                    "label_tr": "Kaplama kusuru",
                    "conf": 0.52,
                    "box": [180, 200, 280, 300],
                }
            ],
            "anomaly_p": 0.08,
            "color": (50, 45, 30),
        },
        # Termal kamera modülü (transistor)
        {
            "inspector": "Caner Demir",
            "product_group": "Termal kamera modülü",
            "serial_no": "TC-20261002-018",
            "ai_decision": "ACCEPT",
            "ai_defect_type": "Yok",
            "confidence": 0.94,
            "defect_score": 0.07,
            "human_decision": "ACCEPT",
            "human_defect_type": "Yok",
            "note": "demo kaydı: Sensör bacak dizilimi ve lehimler hatasız.",
            "detections": [],
            "anomaly_p": 0.85,
            "color": (30, 50, 40),
        },
        {
            "inspector": "Caner Demir",
            "product_group": "Termal kamera modülü",
            "serial_no": "TC-20261002-031",
            "ai_decision": "REJECT",
            "ai_defect_type": "Konektör/montaj kusuru",
            "confidence": 0.91,
            "defect_score": 0.86,
            "human_decision": "REJECT",
            "human_defect_type": "Konektör/montaj kusuru",
            "note": "demo kaydı: Konektör pininde mekanik eğrilik ve temas riski.",
            "detections": [
                {
                    "label": "bent",
                    "label_tr": "Konektör/montaj kusuru",
                    "conf": 0.86,
                    "box": [220, 290, 310, 390],
                }
            ],
            "anomaly_p": 0.02,
            "color": (60, 25, 35),
        },
        {
            "inspector": "Merve Kaya",
            "product_group": "Termal kamera modülü",
            "serial_no": "TC-20261003-002",
            "ai_decision": "REVIEW",
            "ai_defect_type": "Optik eksen şüphesi",
            "confidence": 0.51,
            "defect_score": 0.44,
            "human_decision": "ACCEPT",
            "human_defect_type": "Yok",
            "note": "demo kaydı: Eksen şüphesi uyarısı alındı, kolimatör ölçümü tolerans içinde.",
            "detections": [],
            "anomaly_p": 0.09,
            "color": (45, 45, 55),
        },
        # Gözetleme ünitesi (cable)
        {
            "inspector": "Ahmet Yılmaz",
            "product_group": "Gözetleme ünitesi",
            "serial_no": "GU-20261003-011",
            "ai_decision": "ACCEPT",
            "ai_defect_type": "Yok",
            "confidence": 0.97,
            "defect_score": 0.04,
            "human_decision": "ACCEPT",
            "human_defect_type": "Yok",
            "note": "demo kaydı: Dış gövde ve montaj yivleri standartlara tam uyumlu.",
            "detections": [],
            "anomaly_p": 0.92,
            "color": (25, 40, 45),
        },
        {
            "inspector": "Caner Demir",
            "product_group": "Gözetleme ünitesi",
            "serial_no": "GU-20261003-024",
            "ai_decision": "REJECT",
            "ai_defect_type": "Kaplama kusuru",
            "confidence": 0.84,
            "defect_score": 0.79,
            "human_decision": "REJECT",
            "human_defect_type": "Yüzey çiziği",  # Differing defect type -> agreement = 0
            "note": "demo kaydı: AI kaplama lekesi olarak sınıflandırdı, operatör derin çizik olarak düzeltti.",
            "detections": [
                {
                    "label": "color",
                    "label_tr": "Kaplama kusuru",
                    "conf": 0.77,
                    "box": [150, 160, 240, 230],
                }
            ],
            "anomaly_p": 0.03,
            "color": (65, 35, 25),
        },
    ]

    if count == 0:
        for rec in demo_records:
            # Create a visually distinct dummy image with text
            demo_img = Image.new("RGB", (512, 512), color=rec["color"])
            draw = ImageDraw.Draw(demo_img)
            draw.rectangle([20, 20, 492, 492], outline=(180, 180, 180), width=2)
            draw.text((40, 40), f"3E Elektro Optik - {rec['product_group']}", fill=(240, 240, 240))
            draw.text((40, 70), f"SN: {rec['serial_no']}", fill=(200, 200, 200))
            draw.text((40, 100), f"Durum: {rec['ai_decision']}", fill=(255, 200, 100))

            for det in rec["detections"]:
                b = det["box"]
                draw.rectangle(b, outline=(255, 50, 50), width=3)
                draw.text((b[0] + 4, b[1] + 4), f"{det['label_tr']} ({det['conf']:.2f})", fill=(255, 100, 100))

            engine_outputs = {
                "anomaly": {
                    "available": True,
                    "backend": "efficientad-onnx",
                    "score": 1.0 - rec["anomaly_p"],
                    "p_value": rec["anomaly_p"],
                    "prob": round(1.0 - rec["anomaly_p"], 2),
                    "latency_ms": 38,
                    "error": None,
                },
                "detector": {
                    "available": True,
                    "detections": rec["detections"],
                    "prob": round(max([d["conf"] for d in rec["detections"]], default=0.04), 2),
                    "latency_ms": 32,
                    "error": None,
                },
                "vlm": {
                    "available": True,
                    "called": (rec["ai_decision"] != "ACCEPT"),
                    "model": config.GEMINI_MODEL,
                    "results": [],
                    "majority_type": rec["ai_defect_type"] if rec["ai_decision"] != "ACCEPT" else None,
                    "consistency": 1.0,
                    "prob": rec["defect_score"],
                    "reasoning": f"Görsel incelemesinde tespit: {rec['note']}",
                    "location": "Merkez bölge",
                    "latency_ms": 420 if rec["ai_decision"] != "ACCEPT" else 0,
                    "error": None,
                },
            }

            model_versions = {
                "anomaly": "efficientad-v1.0",
                "detector": "yolo11s-seg-v1.0",
                "vlm": config.GEMINI_MODEL,
                "app": config.APP_VERSION,
            }

            save_inspection(
                inspector=rec["inspector"],
                product_group=rec["product_group"],
                serial_no=rec["serial_no"],
                image=demo_img,
                ai_decision=rec["ai_decision"],
                ai_defect_type=rec["ai_defect_type"],
                confidence=rec["confidence"],
                defect_score=rec["defect_score"],
                human_decision=rec["human_decision"],
                human_defect_type=rec["human_defect_type"],
                note=rec["note"],
                model_versions=model_versions,
                engine_outputs=engine_outputs,
            )

    # Seed demo boresight records
    demo_bore_records = [
        {
            "inspector": "Ahmet Yılmaz",
            "serial_no": "LN-20261001-014",
            "stage": "son test",
            "channel": "Görünür",
            "dx_px": 1.15,
            "dy_px": -0.82,
            "err_mrad": 0.098,
            "tolerance_mrad": 0.5,
            "result": "PASS",
            "inter_channel_mrad": None,
            "drift_mrad": None,
            "params": {"pixel_pitch_um": 3.45, "focal_length_mm": 50.0, "quality_score": 0.94, "note": "demo kaydı: Boresight eksen testi başarılı, tolerans içinde."},
        },
        {
            "inspector": "Caner Demir",
            "serial_no": "TC-20261003-002",
            "stage": "titreşim öncesi",
            "channel": "Termal",
            "dx_px": 0.42,
            "dy_px": -0.31,
            "err_mrad": 0.251,
            "tolerance_mrad": 0.5,
            "result": "PASS",
            "inter_channel_mrad": None,
            "drift_mrad": None,
            "params": {"pixel_pitch_um": 12.0, "focal_length_mm": 25.0, "quality_score": 0.91, "az_mrad": 0.201, "el_mrad": -0.150, "note": "demo kaydı: Titreşim öncesi referans eksen ölçümü."},
        },
        {
            "inspector": "Caner Demir",
            "serial_no": "TC-20261003-002",
            "stage": "titreşim sonrası",
            "channel": "Termal",
            "dx_px": 0.61,
            "dy_px": -0.45,
            "err_mrad": 0.364,
            "tolerance_mrad": 0.5,
            "result": "PASS",
            "inter_channel_mrad": None,
            "drift_mrad": 0.113,
            "params": {"pixel_pitch_um": 12.0, "focal_length_mm": 25.0, "quality_score": 0.92, "az_mrad": 0.292, "el_mrad": -0.216, "note": "demo kaydı: Titreşim sonrası eksen ölçümü, kayma 0.113 mrad (tolerans <0.50 mrad)."},
        },
        {
            "inspector": "Caner Demir",
            "serial_no": "TC-20261002-031",
            "stage": "son test",
            "channel": "Termal",
            "dx_px": 2.45,
            "dy_px": -1.88,
            "err_mrad": 1.482,
            "tolerance_mrad": 0.5,
            "result": "FAIL",
            "inter_channel_mrad": None,
            "drift_mrad": None,
            "params": {"pixel_pitch_um": 12.0, "focal_length_mm": 25.0, "quality_score": 0.88, "note": "demo kaydı: Optik eksen kaçıklığı toleransı aştı (1.482 mrad > 0.500 mrad)."},
        },
    ]
    for b_rec in demo_bore_records:
        save_boresight(
            inspector=b_rec["inspector"],
            serial_no=b_rec["serial_no"],
            stage=b_rec["stage"],
            channel=b_rec["channel"],
            dx_px=b_rec["dx_px"],
            dy_px=b_rec["dy_px"],
            err_mrad=b_rec["err_mrad"],
            tolerance_mrad=b_rec["tolerance_mrad"],
            result=b_rec["result"],
            inter_channel_mrad=b_rec["inter_channel_mrad"],
            drift_mrad=b_rec["drift_mrad"],
            params=b_rec["params"],
        )
    _seed_post_production()


def _seed_post_production() -> None:
    """Seeds demo boresight/MTF/torque rows, each table only when it is empty."""
    init_db()
    with get_db() as conn:
        counts = {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("boresight_tests", "mtf_tests", "torque_tests")
        }
    note = {"note": "demo kaydı"}
    if counts["boresight_tests"] == 0:
        save_boresight("Ahmet Yılmaz", "LN-DEMO-001", "son test", "Görünür", 0.9, -0.5, 0.12, 0.5, "PASS", params=dict(note))
        save_boresight("Caner Demir", "TC-DEMO-002", "son test", "Termal", 2.1, -1.7, 1.2, 0.5, "FAIL", params=dict(note))
    if counts["mtf_tests"] == 0:
        save_mtf("Ahmet Yılmaz", "LN-DEMO-001", "Görünür", 0.31, 45.0, 0.08, 5.0, 0.25, "PASS", params=dict(note))
        save_mtf("Caner Demir", "TC-DEMO-002", "Termal", 0.17, 20.0, 0.03, 5.0, 0.25, "FAIL", params=dict(note))
        save_mtf("Ahmet Yılmaz", "LN-DEMO-003", "Görünür", 0.26, 40.0, 0.06, 12.0, 0.25, "UNCERTAIN", params=dict(note))
    if counts["torque_tests"] == 0:
        save_torque("Ahmet Yılmaz", "LN-DEMO-001", "son test", "kırmızı", 1.8, 0.6, 5.0, "PASS", params=dict(note))
        save_torque("Caner Demir", "TC-DEMO-002", "son test", "sarı", 30.0, 9.5, 5.0, "FAIL", params=dict(note))
        save_torque("Caner Demir", "TC-DEMO-004", "titreşim sonrası", "kırmızı", None, None, 5.0, "UNCERTAIN", params=dict(note))


_PP_LABEL = {"PASS": "🟢 GEÇTİ", "FAIL": "🔴 KALDI", "UNCERTAIN": "🟡 BELİRSİZ"}


def post_production_stats() -> dict[str, dict[str, Any]]:
    """Per post-production test: total / pass / fail / uncertain counts and pass rate (%)."""
    init_db()
    tables = {"Optik Eksen": "boresight_tests", "MTF": "mtf_tests", "Tork İşareti": "torque_tests"}
    out: dict[str, dict[str, Any]] = {}
    with get_db() as conn:
        for name, table in tables.items():
            c = {"PASS": 0, "FAIL": 0, "UNCERTAIN": 0}
            for res, n in conn.execute(f"SELECT result, COUNT(*) FROM {table} GROUP BY result"):
                c[_normalize_result(res)] += n
            total = sum(c.values())
            out[name] = {
                "total": total, "pass": c["PASS"], "fail": c["FAIL"], "uncertain": c["UNCERTAIN"],
                "pass_rate": round(100.0 * c["PASS"] / total, 1) if total else 0.0,
            }
    return out


def list_post_production(limit: int = 100) -> list[dict[str, Any]]:
    """Unified boresight/MTF/torque rows, newest first."""
    def fmt(v: Any, spec: str, unit: str) -> str:
        return f"{format(v, spec)} {unit}" if isinstance(v, (int, float)) else "-"

    def tol(prefix: str, v: Any, spec: str, unit: str) -> str:
        return f"{prefix} {fmt(v, spec, unit)}" if v is not None else "-"

    rows: list[dict[str, Any]] = []
    for r in list_boresight(limit=limit):
        rows.append({"_ts": r.get("ts") or "", "Seri No": r.get("serial_no"), "Muayeneci": r.get("inspector"),
                     "Test": "Optik Eksen", "Aşama": r.get("stage") or "-",
                     "Ölçüm": fmt(r.get("err_mrad"), ".2f", "mrad"),
                     "Tolerans/Spec": tol("≤", r.get("tolerance_mrad"), ".2f", "mrad"),
                     "_res": r.get("result")})
    for r in list_mtf(limit=limit):
        rows.append({"_ts": r.get("ts") or "", "Seri No": r.get("serial_no"), "Muayeneci": r.get("inspector"),
                     "Test": "MTF", "Aşama": "-",
                     "Ölçüm": "MTF50 " + fmt(r.get("mtf50_cyc_px"), ".2f", "cy/px"),
                     "Tolerans/Spec": tol("≥", r.get("spec_mtf50"), ".2f", "cy/px"),
                     "_res": r.get("result")})
    for r in list_torque(limit=limit):
        a = r.get("angle_deg")
        rows.append({"_ts": r.get("ts") or "", "Seri No": r.get("serial_no"), "Muayeneci": r.get("inspector"),
                     "Test": "Tork İşareti", "Aşama": r.get("stage") or "-",
                     "Ölçüm": fmt(abs(a) if isinstance(a, (int, float)) else None, ".1f", "°"),
                     "Tolerans/Spec": tol("≤", r.get("tolerance_deg"), ".1f", "°"),
                     "_res": r.get("result")})
    rows.sort(key=lambda x: x["_ts"], reverse=True)
    out = []
    for r in rows[:limit]:
        ts = r["_ts"]
        try:
            ts = datetime.fromisoformat(ts).astimezone().strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
        d = {"Zaman": ts}
        d.update({k: v for k, v in r.items() if not k.startswith("_")})
        d["Sonuç"] = _PP_LABEL[_normalize_result(r["_res"])]
        out.append(d)
    return out


class SerialStatus(dict):
    """Custom dictionary representing serial number quality gate status.

    Supports equality comparison with string status (e.g. status == 'SEVKE HAZIR')
    so that both `serial_status(sn) == 'SEVKE HAZIR'` and `serial_status(sn)['overall'] == 'SEVKE HAZIR'` work.
    """

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return (
                self.get("overall") == other
                or self.get("status") == other
                or self.get("genel_durum") == other
            )
        return super().__eq__(other)

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


def serial_status(serial_no: str) -> SerialStatus:
    """Evaluates the quality gate status for a given serial number.

    Returns a SerialStatus dictionary containing:
    - serial_no: serial number string
    - product_group: product group name or '-'
    - visual_decision: 'KABUL' | 'RET' | 'none'
    - boresight_result: 'GEÇTİ' | 'KALDI' | 'none'
    - drift_status: 'GEÇTİ' | 'KALDI' | 'none'
    - overall: 'SEVKE HAZIR' | 'BEKLEMEDE' | 'RET'
    - emoji: '🟢' | '🟡' | '🔴'

    Quality Gate Logic:
    - 'SEVKE HAZIR': only if visual human decision == ACCEPT and latest boresight == PASS
      (and drift within tolerance if both pre/post exist).
    - 'RET': if ANY stage failed (visual human decision == REJECT, latest boresight == FAIL,
      or vibration drift exceeds tolerance).
    - 'BEKLEMEDE': if a required stage is missing and no stage has failed.
    """
    init_db()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT * FROM inspections
            WHERE serial_no = ?
            ORDER BY id DESC
            """,
            (serial_no,),
        )
        insp_rows = [dict(r) for r in cursor.fetchall()]

        cursor.execute(
            """
            SELECT * FROM boresight_tests
            WHERE serial_no = ?
            ORDER BY id DESC
            """,
            (serial_no,),
        )
        bore_rows = [dict(r) for r in cursor.fetchall()]

        cursor.execute("SELECT * FROM mtf_tests WHERE serial_no = ? ORDER BY id DESC", (serial_no,))
        mtf_rows = [dict(r) for r in cursor.fetchall()]
        cursor.execute("SELECT * FROM torque_tests WHERE serial_no = ? ORDER BY id DESC", (serial_no,))
        torque_rows = [dict(r) for r in cursor.fetchall()]

    def _tr(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "none"
        return {"PASS": "GEÇTİ", "FAIL": "KALDI"}.get(_normalize_result(rows[0].get("result")), "BELİRSİZ")

    mtf_result = _tr(mtf_rows)
    torque_result = _tr(torque_rows)
    extra_failed = "KALDI" in (mtf_result, torque_result)
    extra_uncertain = "BELİRSİZ" in (mtf_result, torque_result)

    # Product group from latest inspection (or boresight params)
    product_group = "-"
    if insp_rows and insp_rows[0].get("product_group"):
        product_group = insp_rows[0]["product_group"]
    elif bore_rows:
        try:
            p = json.loads(bore_rows[0]["params"]) if isinstance(bore_rows[0].get("params"), str) else (bore_rows[0].get("params") or {})
            product_group = p.get("product_group", "-")
        except Exception:
            product_group = "-"

    # 1. Visual Inspection Human Decision
    visual_human_raw = None
    visual_decision = "none"
    visual_passed = False
    visual_failed = False
    visual_missing = True

    if insp_rows:
        visual_missing = False
        latest_insp = insp_rows[0]
        hum_dec = (latest_insp.get("human_decision") or "").strip().upper()
        visual_human_raw = hum_dec
        if hum_dec in ("ACCEPT", "KABUL", "PASS", "ONAY"):
            visual_decision = "KABUL"
            visual_passed = True
        elif hum_dec in ("REJECT", "RET", "FAIL"):
            visual_decision = "RET"
            visual_failed = True
        else:
            visual_decision = "none"

    # 2. Latest Boresight Test Result
    boresight_raw = None
    boresight_result = "none"
    boresight_passed = False
    boresight_failed = False
    boresight_missing = True

    if bore_rows:
        boresight_missing = False
        latest_bore = bore_rows[0]
        res = (latest_bore.get("result") or "").strip().upper()
        boresight_raw = res
        if res in ("PASS", "GEÇTI", "GECTI", "GEÇTİ"):
            boresight_result = "GEÇTİ"
            boresight_passed = True
        elif res in ("FAIL", "KALDI", "REJECT", "RET"):
            boresight_result = "KALDI"
            boresight_failed = True
        else:
            boresight_result = "none"

    # 3. Vibration Drift Check (if pre and post vibration records exist)
    pre_bore = None
    post_bore = None
    for b in bore_rows:
        stg = (b.get("stage") or "").lower()
        if ("öncesi" in stg or "oncesi" in stg or "pre" in stg) and pre_bore is None:
            pre_bore = b
        elif ("sonrası" in stg or "sonrasi" in stg or "post" in stg) and post_bore is None:
            post_bore = b

    drift_status = "none"
    drift_passed = None
    drift_failed = False
    drift_val = None

    if pre_bore is not None and post_bore is not None:
        tol = float(post_bore.get("tolerance_mrad") or 0.5)
        if post_bore.get("drift_mrad") is not None:
            drift_val = float(post_bore["drift_mrad"])
            if drift_val <= tol:
                drift_passed = True
                drift_status = "GEÇTİ"
            else:
                drift_passed = False
                drift_failed = True
                drift_status = "KALDI"
        else:
            try:
                p_pre = json.loads(pre_bore["params"]) if isinstance(pre_bore.get("params"), str) else (pre_bore.get("params") or {})
                p_post = json.loads(post_bore["params"]) if isinstance(post_bore.get("params"), str) else (post_bore.get("params") or {})
                az1 = p_pre.get("az_mrad")
                el1 = p_pre.get("el_mrad")
                az2 = p_post.get("az_mrad")
                el2 = p_post.get("el_mrad")

                if az1 is not None and el1 is not None and az2 is not None and el2 is not None:
                    drift_val = float(((float(az2) - float(az1))**2 + (float(el2) - float(el1))**2)**0.5)
                else:
                    pitch = float(p_post.get("pixel_pitch_um") or 3.45)
                    focal = float(p_post.get("focal_length_mm") or 50.0)
                    dx_diff = float(post_bore.get("dx_px", 0.0)) - float(pre_bore.get("dx_px", 0.0))
                    dy_diff = float(post_bore.get("dy_px", 0.0)) - float(pre_bore.get("dy_px", 0.0))
                    dr_px = (dx_diff**2 + dy_diff**2)**0.5
                    drift_val = float(np.arctan(dr_px * pitch * 1e-3 / focal) * 1000.0)

                if drift_val <= tol:
                    drift_passed = True
                    drift_status = "GEÇTİ"
                else:
                    drift_passed = False
                    drift_failed = True
                    drift_status = "KALDI"
            except Exception:
                drift_status = "none"

    # 4. Overall Decision
    if visual_failed or boresight_failed or drift_failed or extra_failed:
        overall = "RET"
        emoji = "🔴"
    elif visual_missing or boresight_missing or not visual_passed or not boresight_passed or extra_uncertain:
        overall = "BEKLEMEDE"
        emoji = "🟡"
    elif visual_passed and boresight_passed and (drift_passed is None or drift_passed is True):
        overall = "SEVKE HAZIR"
        emoji = "🟢"
    else:
        overall = "BEKLEMEDE"
        emoji = "🟡"

    return SerialStatus({
        "serial_no": serial_no,
        "product_group": product_group,
        "visual_decision": visual_decision,
        "boresight_result": boresight_result,
        "drift_status": drift_status,
        "drift_mrad": drift_val,
        "mtf_result": mtf_result,
        "torque_result": torque_result,
        "overall": overall,
        "status": overall,
        "emoji": emoji,
        "gorsel_muayene": visual_decision,
        "son_test": boresight_result,
        "kayma": drift_status,
        "genel_durum": f"{emoji} {overall}",
        "visual_raw": visual_human_raw,
        "boresight_raw": boresight_raw,
        "inspections": insp_rows,
        "boresight_tests": bore_rows,
        "mtf_tests": mtf_rows,
        "torque_tests": torque_rows,
    })


def list_serial_status() -> list[SerialStatus]:
    """Returns quality gate status for all distinct serial numbers."""
    init_db()
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT serial_no, MAX(ts) as max_ts FROM (
                SELECT serial_no, ts FROM inspections WHERE serial_no IS NOT NULL AND TRIM(serial_no) != ''
                UNION ALL
                SELECT serial_no, ts FROM boresight_tests WHERE serial_no IS NOT NULL AND TRIM(serial_no) != ''
                UNION ALL
                SELECT serial_no, ts FROM mtf_tests WHERE serial_no IS NOT NULL AND TRIM(serial_no) != ''
                UNION ALL
                SELECT serial_no, ts FROM torque_tests WHERE serial_no IS NOT NULL AND TRIM(serial_no) != ''
            ) GROUP BY serial_no ORDER BY max_ts DESC
            """
        )
        serials = [r[0] for r in cursor.fetchall()]
    return [serial_status(s) for s in serials]
