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


def export_csv(dest_path: Path | str | None = None) -> Path:
    """Exports all inspections to a CSV file.

    Returns the Path to the generated CSV.
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
        if count > 0:
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
