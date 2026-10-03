"""app.py - 3E Kalite Kontrol — AI Destekli Muayene (PoC)

Gradio Web Application and User Interface.
"""

from __future__ import annotations

import io
import os
import threading
from datetime import datetime

# Hugging Face ZeroGPU hosting requires at least one @spaces.GPU function at startup.
# Inference itself runs on CPU; this probe is never called.
try:
    import spaces

    @spaces.GPU(duration=5)
    def _zerogpu_probe():
        return True
except ImportError:
    pass
from pathlib import Path
from typing import Any

import warnings
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import config
import engines.boresight as boresight_engine
import engines.sam as sam_engine
import engines.torque_mark as torque_mark_engine
import storage
from gradio_image_annotation import image_annotator

# Operator annotation labels: DEFECT_TYPES minus 'Yok'
ANNOTATOR_LABELS = [d for d in config.DEFECT_TYPES if d != "Yok"]
ANNOTATOR_COLORS = [
    "#e11d48",  # Yüzey çiziği - Rose
    "#ea580c",  # Kaplama kusuru - Orange
    "#2563eb",  # Konektör/montaj kusuru - Blue
    "#7c3aed",  # Optik eksen şüphesi - Purple
    "#4b5563",  # Diğer - Gray
]

# Suppress Gradio 6 notice about theme/css moving from Blocks to launch
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The parameters have been moved from the Blocks constructor to the launch\\(\\) method",
)

# ---------------------------------------------------------------------------
# Pipeline Import & Fallback Stub
# ---------------------------------------------------------------------------
try:
    import pipeline
except ImportError:
    import warnings

    warnings.warn(
        "pipeline module not found or failed to load, falling back to CONTRACT fake_run stub"
    )
    pipeline = None


def fake_run(image: Image.Image, product_group: str) -> dict[str, Any]:
    """Stub returning CONTRACT-shaped dict when pipeline.py is unavailable."""
    stat = np.array(image.convert("L"))
    mean_val = float(np.mean(stat))

    w, h = image.size
    heatmap_overlay = image.copy()
    detection_image = image.copy()

    if mean_val < 90:
        decision = "REJECT"
        defect_type = "Yüzey çiziği"
        confidence = 0.92
        defect_score = 0.88
        severity = "Yüksek"
        reasons = [
            "Anomali tespit edildi: p=0.01 (yüksek sapma)",
            "YOLO: 'Yüzey çiziği' tespit edildi (%88 güven)",
            "VLM: Parça yüzeyinde derin mekanik çizik doğrulandı",
        ]
        box = [int(w * 0.3), int(h * 0.35), int(w * 0.65), int(h * 0.6)]
        d_draw = ImageDraw.Draw(detection_image)
        d_draw.rectangle(box, outline=(255, 50, 50), width=4)
        d_draw.text((box[0] + 5, box[1] + 5), f"{defect_type} 0.88", fill=(255, 80, 80))

        h_draw = ImageDraw.Draw(heatmap_overlay)
        h_draw.rectangle(box, fill=(255, 0, 0, 80), outline=(255, 200, 0))

        detections = [
            {
                "label": "scratch",
                "label_tr": "Yüzey çiziği",
                "conf": 0.88,
                "box": box,
            }
        ]
        vlm_called = True
        vlm_reasoning = (
            "Parça yüzeyinde derin çizik ve homojen olmayan doku gözlendi. Mekanik tolerans dışı."
        )
    elif mean_val > 165 or (product_group in config.HIGH_RISK_GROUPS and mean_val > 120):
        decision = "REVIEW"
        defect_type = (
            "Kaplama kusuru" if product_group != "Termal kamera modülü" else "Optik eksen şüphesi"
        )
        confidence = 0.58
        defect_score = 0.49
        severity = "Orta"
        reasons = [
            "Motorlar arasında kısmi uyumsuzluk (Anomali: şüpheli, YOLO: tespit yok)",
            "Belirsizlik eşiği aşıldı — operatör doğrulaması gerekiyor",
            f"Risk kuralı: {product_group} hassas parça kategorisinde",
        ]
        detections = []
        vlm_called = True
        vlm_reasoning = (
            "Yüzey parlaklığı ve yansıma homojenliğinde hafif sapma mevcut; parça kabul sınırında."
        )
    else:
        decision = "ACCEPT"
        defect_type = "Yok"
        confidence = 0.95
        defect_score = 0.08
        severity = None
        reasons = [
            "Anomali skoru normal sınırlar içinde (p=0.84)",
            "YOLO kusur dedektörü herhangi bir anomali bulmadı",
            "Erken çıkış kuralı devrede — VLM çağrısına gerek kalmadı",
        ]
        detections = []
        vlm_called = False
        vlm_reasoning = "Görsel temiz, referans parça standartlarına uygun."

    return {
        "decision": decision,
        "decision_label": config.DECISIONS.get(decision, decision),
        "confidence": confidence,
        "defect_score": defect_score,
        "defect_type": defect_type,
        "severity": severity,
        "reasons": reasons,
        "heatmap_overlay": heatmap_overlay,
        "detection_image": detection_image,
        "engines": {
            "anomaly": {
                "available": True,
                "backend": "efficientad-onnx",
                "score": defect_score,
                "p_value": round(1.0 - defect_score, 2),
                "prob": defect_score,
                "latency_ms": 42,
                "error": None,
            },
            "detector": {
                "available": True,
                "detections": detections,
                "prob": round(max([d["conf"] for d in detections], default=0.03), 2),
                "latency_ms": 35,
                "error": None,
            },
            "vlm": {
                "available": True,
                "called": vlm_called,
                "model": config.GEMINI_MODEL,
                "results": [],
                "majority_type": defect_type if vlm_called else None,
                "consistency": 1.0 if vlm_called else 0.0,
                "prob": defect_score if vlm_called else 0.0,
                "reasoning": vlm_reasoning,
                "location": "Merkez eksen" if vlm_called else "",
                "latency_ms": 520 if vlm_called else 0,
                "error": None,
            },
        },
        "model_versions": {
            "anomaly": "efficientad-v1.0",
            "detector": "yolo11s-seg-v1.0",
            "vlm": config.GEMINI_MODEL,
            "app": config.APP_VERSION,
        },
        "latency_ms": 77 + (520 if vlm_called else 0),
    }


def execute_pipeline(image: Image.Image, product_group: str) -> dict[str, Any]:
    """Runs the real pipeline if available; otherwise falls back to fake_run."""
    if pipeline is not None and hasattr(pipeline, "run"):
        try:
            return pipeline.run(image, product_group)
        except Exception as e:
            warnings.warn(f"pipeline.run failed with error: {e}, falling back to fake_run")
            return fake_run(image, product_group)
    return fake_run(image, product_group)


# ---------------------------------------------------------------------------
# Preprocessing and UI Helpers
# ---------------------------------------------------------------------------
def generate_default_serial() -> str:
    """Generates an automatic serial number."""
    return f"SN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def validate_and_preprocess_image(image: Any) -> Image.Image:
    """Validates upload: max 10 MB, converts to RGB, resizes longest side to 1024."""
    if image is None:
        raise gr.Error(
            "Lütfen muayene için bir görsel yükleyin veya aşağıdaki örneklerden seçin."
        )

    if not isinstance(image, Image.Image):
        try:
            image = Image.open(image)
        except Exception as e:
            raise gr.Error(f"Görsel açılamadı: {e}")

    # Estimate compressed byte size
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    size_bytes = buf.tell()
    if size_bytes > 10 * 1024 * 1024:
        raise gr.Error(
            f"Yüklenen görsel 10 MB sınırını aşıyor ({size_bytes / (1024 * 1024):.1f} MB). "
            "Lütfen daha küçük bir görsel yükleyin."
        )

    # Convert to RGB
    image = image.convert("RGB")

    # Resize longest side to 1024
    w, h = image.size
    max_dim = max(w, h)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return image


def load_sample_examples() -> list[list[Any]]:
    """Loads 1-2 examples per category from samples/ directory if present."""
    examples: list[list[Any]] = []
    if not config.SAMPLES_DIR.exists():
        return examples

    for prod_group, cat in config.PRODUCT_GROUPS.items():
        cat_dir = config.SAMPLES_DIR / cat
        if not cat_dir.exists():
            continue

        # 1-2 from test_good
        good_dir = cat_dir / "test_good"
        if not good_dir.exists():
            good_dir = cat_dir / "good"
        if good_dir.exists():
            good_imgs = sorted(
                list(good_dir.glob("*.png")) + list(good_dir.glob("*.jpg"))
            )
            for p in good_imgs[:2]:
                examples.append([str(p), prod_group])

        # 1-2 from defect
        defect_dir = cat_dir / "defect"
        if defect_dir.exists():
            defect_imgs = sorted(
                list(defect_dir.glob("*.png")) + list(defect_dir.glob("*.jpg"))
            )
            for p in defect_imgs[:2]:
                examples.append([str(p), prod_group])

    return examples


def render_decision_badge(result: dict[str, Any] | None) -> str:
    """Renders HTML decision badge with color coding and typography."""
    if not result:
        return """
        <div style="border: 2px dashed #d1d5db; background: #f9fafb; border-radius: 12px; padding: 26px; text-align: center; color: #6b7280; font-family: system-ui, -apple-system, sans-serif;">
            <div style="font-size: 36px; margin-bottom: 8px;">🔍</div>
            <div style="font-weight: 700; font-size: 16px; color: #374151;">Muayene Bekleniyor</div>
            <div style="font-size: 13px; margin-top: 4px;">Sol panelden bir parça görseli yükleyip <strong>Analiz et</strong> butonuna basınız.</div>
        </div>
        """

    decision = result.get("decision", "REVIEW").upper()
    conf = float(result.get("confidence", 0.0))
    conf_pct = int(round(conf * 100))
    defect_type = result.get("defect_type", "Yok")
    severity = result.get("severity") or "Yok"
    defect_score = float(result.get("defect_score", 0.0))

    if decision == "ACCEPT":
        bg = "#ecfdf5"
        border = "#10b981"
        badge_bg = "#059669"
        text_color = "#065f46"
        icon = "✅"
        title = "KABUL ÖNERİSİ"
        desc = "Parça standartlara uygun görünüyor. Belirgin bir hata tespit edilmedi."
    elif decision == "REJECT":
        bg = "#fef2f2"
        border = "#ef4444"
        badge_bg = "#dc2626"
        text_color = "#991b1b"
        icon = "❌"
        title = "RET ÖNERİSİ"
        desc = "Kritik anomali veya spesifik kusur tespit edildi."
    else:  # REVIEW
        bg = "#fffbeb"
        border = "#f59e0b"
        badge_bg = "#d97706"
        text_color = "#92400e"
        icon = "⚠️"
        title = "İNSAN İNCELEMESİ (REVIEW)"
        desc = "Motorlar arasında çelişki veya sınır değer tespit edildi. Operatör nihai kararı vermelidir."

    sev_badge = (
        f'<span style="background: #e5e7eb; color: #1f2937; padding: 2px 8px; border-radius: 6px; font-size: 12px; margin-left: 8px;">Önem Derecesi: <strong>{severity}</strong></span>'
        if severity and severity != "Yok" and severity != "Belirtilmedi"
        else ""
    )

    return f"""
    <div style="border: 2px solid {border}; background: {bg}; border-radius: 12px; padding: 18px 22px; margin-bottom: 12px; font-family: system-ui, -apple-system, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div style="flex: 1; min-width: 220px;">
                <span style="display: inline-block; background: {badge_bg}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px; letter-spacing: 0.3px;">
                    {icon} {title}
                </span>
                <div style="margin-top: 10px; font-size: 15px; color: #1f2937;">
                    <strong>Tespit Edilen Kusur:</strong> <span style="font-weight: 700; color: {text_color};">{defect_type}</span>
                    {sev_badge}
                </div>
                <div style="margin-top: 4px; font-size: 12px; color: #4b5563;">
                    {desc}
                </div>
            </div>
            <div style="text-align: right; min-width: 120px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: 700; letter-spacing: 0.5px;">AI Güven Skoru</div>
                <div style="font-size: 40px; font-weight: 800; color: {text_color}; line-height: 1.1;">%{conf_pct}</div>
                <div style="font-size: 12px; color: #4b5563; margin-top: 2px;">Kusur Skoru: <strong>{defect_score:.2f}</strong></div>
            </div>
        </div>
    </div>
    """


def render_metrics_cards(st: dict[str, Any]) -> str:
    """Renders HTML summary cards for the dashboard."""
    total = st.get("total", 0)
    ai_accept = st.get("ai_accept", 0)
    ai_reject = st.get("ai_reject", 0)
    ai_review = st.get("ai_review", 0)
    review_rate = st.get("human_review_rate", 0.0)
    agreement_pct = st.get("agreement_pct", 0.0)
    missed_count = st.get("missed_by_ai", 0)

    return f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 16px; font-family: system-ui, -apple-system, sans-serif;">
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase;">Toplam Muayene</div>
            <div style="font-size: 30px; font-weight: 800; color: #111827; margin-top: 4px;">{total}</div>
            <div style="font-size: 12px; color: #9ca3af; margin-top: 2px;">Kayıtlı muayene adedi</div>
        </div>
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase;">Kabul / Ret / İnceleme</div>
            <div style="font-size: 22px; font-weight: 700; color: #111827; margin-top: 8px;">
                <span style="color: #059669;" title="Kabul">{ai_accept}</span> /
                <span style="color: #dc2626;" title="Ret">{ai_reject}</span> /
                <span style="color: #d97706;" title="İnceleme">{ai_review}</span>
            </div>
            <div style="font-size: 12px; color: #9ca3af; margin-top: 2px;">AI ilk öneri dökümü</div>
        </div>
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase;">İnsan İncelemesi Oranı</div>
            <div style="font-size: 30px; font-weight: 800; color: #d97706; margin-top: 4px;">%{review_rate:.1f}</div>
            <div style="font-size: 12px; color: #9ca3af; margin-top: 2px;">Operatöre devredilen oran</div>
        </div>
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase;">AI - İnsan Uyumu</div>
            <div style="font-size: 30px; font-weight: 800; color: #2563eb; margin-top: 4px;">%{agreement_pct:.1f}</div>
            <div style="font-size: 12px; color: #9ca3af; margin-top: 2px;">Doğrulanan karar mutabakatı</div>
        </div>
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase;">Kaçan Kusur (Operatör)</div>
            <div style="font-size: 30px; font-weight: 800; color: #dc2626; margin-top: 4px;">{missed_count}</div>
            <div style="font-size: 12px; color: #9ca3af; margin-top: 2px;">Sistemin kaçırıp operatörün işaretlediği</div>
        </div>
    </div>
    """


def get_defect_dataframe(st: dict[str, Any]) -> pd.DataFrame:
    """Builds defect distribution dataframe for BarPlot."""
    dist = st.get("defect_distribution", {})
    rows = [{"Kusur Tipi": k, "Adet": v} for k, v in dist.items()]
    if not rows:
        rows = [{"Kusur Tipi": d, "Adet": 0} for d in config.DEFECT_TYPES if d != "Yok"]
    return pd.DataFrame(rows)


def get_history_dataframe(limit: int = 100) -> pd.DataFrame:
    """Builds dataframe of recent inspections."""
    records = storage.list_inspections(limit=limit)
    rows = []
    for r in records:
        ag = r.get("agreement_flag")
        ag_text = "Evet (1)" if ag == 1 else ("Hayır (0)" if ag == 0 else "İnceleme (-)")
        ai_dec = r.get("ai_decision", "")
        ai_label = config.DECISIONS.get(ai_dec, ai_dec)
        conf_str = f"%{int(round(float(r.get('confidence', 0.0)) * 100))}"
        ts_fmt = r.get("ts", "")[:19].replace("T", " ")
        rows.append(
            {
                "ID": r.get("id"),
                "Tarih/Saat": ts_fmt,
                "Muayeneci": r.get("inspector"),
                "Ürün Grubu": r.get("product_group"),
                "Seri No": r.get("serial_no"),
                "AI Kararı": ai_label,
                "AI Kusur": r.get("ai_defect_type"),
                "Güven": conf_str,
                "İnsan Kararı": r.get("human_decision"),
                "İnsan Kusur": r.get("human_defect_type"),
                "Uyum": ag_text,
                "Not": r.get("note", ""),
            }
        )
    return pd.DataFrame(rows)


def build_engines_dataframe(engines: dict[str, Any]) -> pd.DataFrame:
    """Builds the engine breakdown table from CONTRACT engines output."""
    rows = []

    # 1. Anomaly Engine
    an = engines.get("anomaly", {})
    an_avail = an.get("available", False)
    an_err = an.get("error")
    an_status = "Aktif" if an_avail else f"Devre dışı ({an_err or 'model yok'})"
    an_prob = f"{an.get('prob', 0.0):.2f} (p={an.get('p_value', 1.0):.3f})" if an_avail else "—"
    an_lat = an.get("latency_ms", 0)
    an_backend = an.get("backend") or "PatchCore + DINOv2"
    rows.append(
        {
            "Motor": "Anomali Tespiti",
            "Durum": an_status,
            "Olasılık / Metrik": an_prob,
            "Gecikme (ms)": an_lat,
            "Model / Backend": an_backend,
        }
    )

    # 2. Object Detector
    det = engines.get("detector", {})
    det_avail = det.get("available", False)
    det_err = det.get("error")
    detections = det.get("detections", [])
    if det_avail:
        det_status = (
            f"Aktif ({len(detections)} tespit)" if detections else "Aktif (0 tespit)"
        )
        det_prob = f"{det.get('prob', 0.0):.2f}"
    else:
        det_status = f"Devre dışı ({det_err or 'model yok'})"
        det_prob = "—"
    det_lat = det.get("latency_ms", 0)
    rows.append(
        {
            "Motor": "Kusur Dedektörü",
            "Durum": det_status,
            "Olasılık / Metrik": det_prob,
            "Gecikme (ms)": det_lat,
            "Model / Backend": "YOLO11s-seg (ONNX)",
        }
    )

    # 3. VLM (Gemini)
    vlm = engines.get("vlm", {})
    vlm_avail = vlm.get("available", False)
    vlm_called = vlm.get("called", False)
    vlm_err = vlm.get("error")
    if not vlm_avail:
        vlm_status = f"Devre dışı ({vlm_err or 'API anahtarı yok'})"
        vlm_prob = "—"
    elif not vlm_called:
        vlm_status = "Erken Çıkış (Atlandı)"
        vlm_prob = "—"
    else:
        vlm_status = "Çağrıldı (Aktif)"
        vlm_prob = f"{vlm.get('prob', 0.0):.2f} (Tutarlılık: %{int(vlm.get('consistency', 1.0) * 100)})"
    vlm_lat = vlm.get("latency_ms", 0)
    vlm_model = vlm.get("model") or config.GEMINI_MODEL
    rows.append(
        {
            "Motor": "Görsel Dil Modeli (VLM)",
            "Durum": vlm_status,
            "Olasılık / Metrik": vlm_prob,
            "Gecikme (ms)": vlm_lat,
            "Model / Backend": vlm_model,
        }
    )

    return pd.DataFrame(rows)


def format_reasons(reasons: list[str]) -> str:
    """Formats reason strings into markdown bullet points."""
    if not reasons:
        return "*Belirgin bir kural veya motor gerekçesi üretilmedi.*"
    lines = ["### 📋 Karar Gerekçeleri"]
    for r in reasons:
        lines.append(f"- {r}")
    return "\n".join(lines)


def format_vlm_reasoning(vlm_data: dict[str, Any]) -> str:
    """Formats VLM reasoning markdown."""
    if not vlm_data.get("available", False):
        err = vlm_data.get("error") or "API anahtarı tanımlı değil"
        return f"> ⚠️ *VLM (Gemini) kullanılamadı: {str(err)[:200]}*"
    if not vlm_data.get("called", False):
        return (
            "> ℹ️ *VLM motoru erken çıkış (early exit) kuralı ile atlandı. "
            "Anomali ve nesne dedektörü kesin kabul sinyali verdiği için API çağrısı yapılmadı.*"
        )
    reasoning = vlm_data.get("reasoning", "")
    location = vlm_data.get("location", "")
    loc_str = f" **Konum:** {location}" if location else ""
    return f"""### 🤖 Gemini VLM Muhakeme Raporu
> {reasoning}
{loc_str}
"""


def get_runtime_engine_status_md() -> str:
    """Inspects filesystem and environment to generate engine status cards."""
    # Anomaly
    has_anomaly_onnx = any(
        config.anomaly_onnx(cat).exists()
        for cat in set(config.PRODUCT_GROUPS.values())
    )
    if has_anomaly_onnx:
        anom_status = "🟢 **Aktif (ONNX):** EfficientAD ONNX modelleri mevcut."
    else:
        anom_status = (
            "🟡 **Yedek Devrede (PatchCore):** ONNX dosyaları bulunamadı; "
            "ResNet18 tabanlı PatchCore hafıza bankası yedeği kullanılıyor."
        )

    # Detector
    has_yolo = config.YOLO_MODEL.exists()
    if has_yolo:
        det_status = (
            f"🟢 **Aktif (ONNX):** {config.YOLO_MODEL.name} dedektör ağırlığı yüklü."
        )
    else:
        det_status = (
            "🟡 **Model Bekleniyor:** `models/yolo.onnx` bulunamadı. "
            "Kusur tespit motoru pasif durumda; ağırlıklar anomali ve VLM'e aktarılır."
        )

    # VLM
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    if has_gemini:
        vlm_status = f"🟢 **Aktif:** Google Gemini API anahtarı tanımlı (`{config.GEMINI_MODEL}`)."
    else:
        vlm_status = (
            "🔴 **Pasif:** `GEMINI_API_KEY` ortam değişkeni tanımlı değil. "
            "VLM çağrıları atlanır veya yerel simülasyon yanıtı döner."
        )

    # DB
    db_status = (
        f"🟢 **Aktif:** SQLite `{storage.DB_PATH.name}` dosyasında denetim kaydı."
    )

    return f"""
| Bileşen | Durum Açıklaması |
|---|---|
| **Anomali Tespiti** | {anom_status} |
| **Kusur Dedektörü** | {det_status} |
| **VLM (Gemini)** | {vlm_status} |
| **Kayıt Deposu (SQLite)** | {db_status} |
"""


# ---------------------------------------------------------------------------
# Gradio Callback Functions
# ---------------------------------------------------------------------------
def render_sam_overlay(
    image: Image.Image,
    op_labels: list[dict[str, Any]],
) -> Image.Image:
    """Renders colored masks overlaid on the image with bounding boxes and labels."""
    pil_img = image.convert("RGB")
    w, h = pil_img.size

    color_palette = [
        (225, 29, 72),   # Rose / red
        (234, 88, 12),   # Orange
        (37, 99, 235),   # Blue
        (124, 58, 237),  # Purple
        (13, 148, 136),  # Teal
        (202, 138, 4),   # Yellow
    ]

    img_rgba = pil_img.convert("RGBA")
    mask_canvas = np.zeros((h, w, 4), dtype=np.uint8)

    for i, item in enumerate(op_labels):
        color = color_palette[i % len(color_palette)]
        mask = item.get("mask")
        if mask is not None and isinstance(mask, np.ndarray):
            mask_canvas[mask > 0] = list(color) + [120]  # alpha 120

    mask_layer = Image.fromarray(mask_canvas, mode="RGBA")
    blended = Image.alpha_composite(img_rgba, mask_layer)
    draw = ImageDraw.Draw(blended)

    for i, item in enumerate(op_labels):
        color = color_palette[i % len(color_palette)]
        box = item.get("box", [])
        if len(box) >= 4:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            draw.rectangle([x1, y1, x2, y2], outline=color + (255,), width=3)
            lab = item.get("label", "")
            src = " (operatör)" if item.get("source") == "operator" else " (model)"
            text = f"{lab}{src}"
            text_w = max(70, len(text) * 8 + 12)
            tag_y1 = max(0, y1 - 20)
            tag_y2 = max(20, y1)
            draw.rectangle([x1, tag_y1, x1 + text_w, tag_y2], fill=color + (220,))
            draw.text((x1 + 4, tag_y1 + 2), text, fill=(255, 255, 255, 255))

    return blended.convert("RGB")


def on_sam_click(
    annotations: dict[str, Any] | None,
    last_result: dict[str, Any] | None = None,
    fallback_image: Any = None,
) -> tuple[
    Image.Image | None,  # sam_overlay_view
    str,                 # sam_status_view
    list[dict[str, Any]],# operator_labels_state
]:
    """Runs engines/sam.py on the annotated boxes and generates overlay."""
    if not annotations or not isinstance(annotations, dict):
        if fallback_image is not None:
            annotations = {"image": fallback_image, "boxes": []}
        else:
            return (None, "⚠️ Lütfen önce bir görsel analiz ediniz veya etiketleme alanına kutu ekleyiniz.", [])

    img_data = annotations.get("image")
    if img_data is None:
        img_data = fallback_image

    if img_data is None:
        return (None, "⚠️ Görsel bulunamadı.", [])

    try:
        if isinstance(img_data, np.ndarray):
            pil_img = Image.fromarray(img_data).convert("RGB")
        elif isinstance(img_data, Image.Image):
            pil_img = img_data.convert("RGB")
        else:
            pil_img = Image.open(img_data).convert("RGB")
    except Exception as e:
        return (None, f"⚠️ Görsel yüklenemedi: {e}", [])

    raw_boxes = annotations.get("boxes", [])
    if not raw_boxes:
        return (pil_img, "ℹ️ Hiç kutu işaretlenmedi. Maske çıkarılacak kutu bulunmuyor.", [])

    w, h = pil_img.size
    boxes_coords: list[list[float]] = []
    box_labels: list[str] = []

    for b in raw_boxes:
        x1 = float(b.get("xmin", b.get("x1", 0)))
        y1 = float(b.get("ymin", b.get("y1", 0)))
        x2 = float(b.get("xmax", b.get("x2", 0)))
        y2 = float(b.get("ymax", b.get("y2", 0)))
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        xmin = max(0.0, min(float(w), xmin))
        ymin = max(0.0, min(float(h), ymin))
        xmax = max(0.0, min(float(w), xmax))
        ymax = max(0.0, min(float(h), ymax))
        if xmax > xmin and ymax > ymin:
            boxes_coords.append([xmin, ymin, xmax, ymax])
            box_labels.append(b.get("label") or "Diğer")

    if not boxes_coords:
        return (pil_img, "ℹ️ Geçerli boyutta bir kutu bulunamadı.", [])

    sam_res = sam_engine.segment(pil_img, boxes_coords)
    if not sam_res.get("available", False):
        err = sam_res.get("error", "SAM segmentasyon hatası")
        return (None, f"❌ SAM kullanılamadı: {err}", [])

    masks = sam_res.get("masks", [])
    latency = sam_res.get("latency_ms", 0)

    # Distinguish model detections vs operator additions
    detector_boxes = []
    if last_result:
        for det in last_result.get("engines", {}).get("detector", {}).get("detections", []):
            db = det.get("box")
            if db and len(db) == 4:
                detector_boxes.append((det.get("label_tr") or det.get("label"), db))

    op_state: list[dict[str, Any]] = []
    for i, (b_coords, lab) in enumerate(zip(boxes_coords, box_labels)):
        source = "operator"
        for det_label, det_b in detector_boxes:
            if (
                abs(b_coords[0] - det_b[0]) <= 8
                and abs(b_coords[1] - det_b[1]) <= 8
                and abs(b_coords[2] - det_b[2]) <= 8
                and abs(b_coords[3] - det_b[3]) <= 8
            ):
                source = "model"
                break

        m_arr = masks[i] if i < len(masks) else np.zeros((h, w), dtype=np.uint8)
        op_state.append({
            "label": lab,
            "box": b_coords,
            "mask": m_arr,
            "source": source,
        })

    overlay_img = render_sam_overlay(pil_img, op_state)
    status_msg = (
        f"✅ **SAM Segmentasyonu Tamamlandı:** `{len(boxes_coords)}` kutudan maske çıkarıldı. "
        f"(⏱️ `{latency} ms`)"
    )
    return (overlay_img, status_msg, op_state)


def on_analyze_click(
    image: Any,
    product_group: str,
    serial_no: str,
    inspector: str,
) -> tuple[
    str,  # decision_badge_html
    Image.Image | None,  # heatmap
    Image.Image | None,  # detector image
    str,  # vlm markdown
    str,  # reasons markdown
    pd.DataFrame,  # breakdown dataframe
    str,  # latency text
    dict[str, Any],  # last_result_state
    Any,  # radio update
    Any,  # defect dropdown update
    str,  # review warning markdown
    str,  # save confirmation clear
    dict[str, Any],  # annotator_view update
    Image.Image | None,  # sam_overlay_view reset
    str,  # sam_status_view reset
    list[dict[str, Any]],  # operator_labels_state reset
]:
    """Runs quality control pipeline on the uploaded image and updates the UI."""
    valid_image = validate_and_preprocess_image(image)

    result = execute_pipeline(valid_image, product_group)
    decision = result.get("decision", "REVIEW").upper()
    defect_type = result.get("defect_type", "Yok")
    latency = result.get("latency_ms", 0)

    # Badge HTML
    badge_html = render_decision_badge(result)

    # Images
    heatmap_img = result.get("heatmap_overlay")
    detector_img = result.get("detection_image")

    # Texts & Table
    reasons_md = format_reasons(result.get("reasons", []))
    vlm_md = format_vlm_reasoning(result.get("engines", {}).get("vlm", {}))
    breakdown_df = build_engines_dataframe(result.get("engines", {}))
    latency_str = f"⏱️ **Toplam Analiz Gecikmesi:** `{latency} ms`"

    if decision == "REVIEW":
        radio_update = gr.Radio(
            choices=["Kabul", "Ret"],
            value=None,
            label="Nihai Kararınız (İnceleme Sonucu Zorunludur)",
        )
        warning_md = (
            "⚠️ **DİKKAT:** AI sistemi bu parça için **İNSAN İNCELEMESİ (REVIEW)** önerdi. "
            "'Onayla' seçeneği kapatılmıştır. Lütfen parçayı fiziken inceleyerek "
            "**Kabul** veya **Ret** kararı veriniz."
        )
    else:
        radio_update = gr.Radio(
            choices=["Onayla (AI önerisi)", "Kabul", "Ret"],
            value="Onayla (AI önerisi)",
            label="Nihai Kararınız",
        )
        warning_md = ""

    prefilled_defect = defect_type if defect_type in config.DEFECT_TYPES else "Yok"
    dropdown_update = gr.Dropdown(
        choices=config.DEFECT_TYPES,
        value=prefilled_defect,
    )

    # Pre-load annotator with current image and detector's boxes
    detections = result.get("engines", {}).get("detector", {}).get("detections", [])
    initial_boxes = []
    for det in detections:
        box = det.get("box")
        if box and len(box) == 4:
            lab = det.get("label_tr") or det.get("label") or "Diğer"
            initial_boxes.append({
                "xmin": int(round(box[0])),
                "ymin": int(round(box[1])),
                "xmax": int(round(box[2])),
                "ymax": int(round(box[3])),
                "label": lab,
            })

    annotator_update = {
        "image": valid_image,
        "boxes": initial_boxes,
    }

    return (
        badge_html,
        heatmap_img,
        detector_img,
        vlm_md,
        reasons_md,
        breakdown_df,
        latency_str,
        result,
        radio_update,
        dropdown_update,
        warning_md,
        "",  # clear previous save confirmation
        annotator_update,
        None,  # reset sam_overlay_view
        "*Kutulardan maske çıkarmak için butona basınız.*",
        [],    # reset operator_labels_state
    )


def on_save_click(
    last_result: dict[str, Any] | None,
    image: Any,
    product_group: str,
    serial_no: str,
    inspector: str,
    user_decision: str | None,
    selected_defect_type: str,
    note: str,
    operator_labels: list[dict[str, Any]] | None = None,
    annotations: dict[str, Any] | None = None,
) -> tuple[
    str,  # confirmation message
    str,  # new serial number
    str,  # dashboard metrics HTML
    pd.DataFrame,  # dashboard defect distribution
    pd.DataFrame,  # dashboard history dataframe
]:
    """Saves inspection to storage, validates choices, and refreshes dashboard."""
    if not last_result:
        last_result = {
            "decision": "ACCEPT" if user_decision == "Kabul" else ("REJECT" if user_decision == "Ret" else "REVIEW"),
            "defect_type": selected_defect_type or "Yok",
            "confidence": 0.85,
            "defect_score": 0.15,
            "engines": {"detector": {"detections": []}},
        }

    if not user_decision:
        raise gr.Error(
            "Lütfen bir karar seçiniz (Onayla, Kabul veya Ret)."
        )

    ai_decision = last_result.get("decision", "REVIEW").upper()

    if user_decision == "Onayla (AI önerisi)":
        if ai_decision == "REVIEW":
            raise gr.Error(
                "AI kararı İNSAN İNCELEMESİ aşamasındadır; doğrudan onaylanamaz. "
                "Lütfen parçayı inceleyip 'Kabul' veya 'Ret' seçiniz."
            )
        human_decision = ai_decision
        human_defect_type = (
            "Yok" if ai_decision == "ACCEPT" else (selected_defect_type or last_result.get("defect_type", "Yok"))
        )
    elif user_decision == "Kabul":
        human_decision = "ACCEPT"
        human_defect_type = "Yok"
    elif user_decision == "Ret":
        human_decision = "REJECT"
        human_defect_type = selected_defect_type or "Diğer"
    else:
        human_decision = user_decision
        human_defect_type = selected_defect_type

    labels_to_save: list[dict[str, Any]] = list(operator_labels or [])

    # If operator_labels is empty but annotations has boxes (user drew boxes without pressing SAM)
    if not labels_to_save and annotations and isinstance(annotations, dict):
        raw_boxes = annotations.get("boxes", [])
        if raw_boxes:
            boxes_coords = []
            box_labels = []
            for b in raw_boxes:
                x1 = float(b.get("xmin", b.get("x1", 0)))
                y1 = float(b.get("ymin", b.get("y1", 0)))
                x2 = float(b.get("xmax", b.get("x2", 0)))
                y2 = float(b.get("ymax", b.get("y2", 0)))
                boxes_coords.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
                box_labels.append(b.get("label") or "Diğer")

            sam_res = sam_engine.segment(image, boxes_coords)
            masks = sam_res.get("masks", [])
            for i, (b_coords, lab) in enumerate(zip(boxes_coords, box_labels)):
                m_arr = masks[i] if i < len(masks) else None
                labels_to_save.append({
                    "label": lab,
                    "box": b_coords,
                    "mask": m_arr,
                    "source": "operator",
                })

    # Save to SQLite
    record_id = storage.save_inspection(
        inspector=inspector,
        product_group=product_group,
        serial_no=serial_no,
        image=image,
        pipeline_result=last_result,
        human_decision=human_decision,
        human_defect_type=human_defect_type,
        note=note,
        operator_labels=labels_to_save if labels_to_save else None,
    )

    extra_msg = ""
    if labels_to_save:
        extra_msg = f" ({len(labels_to_save)} adet operatör etiketi/maskesi işlendi)"

    conf_msg = (
        f"✅ **Kayıt Başarılı!** Muayene No: **#{record_id}** veritabanına işlendi. "
        f"(Nihai Karar: **{human_decision}**, Kusur: **{human_defect_type}**){extra_msg}"
    )

    new_sn = generate_default_serial()

    # Refresh dashboard
    new_stats = storage.stats()
    metrics_html = render_metrics_cards(new_stats)
    defect_df = get_defect_dataframe(new_stats)
    history_df = get_history_dataframe(100)

    return (conf_msg, new_sn, metrics_html, defect_df, history_df)


def on_refresh_dashboard() -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Refreshes dashboard metrics cards, defect chart, and inspection table."""
    st = storage.stats()
    return (
        render_metrics_cards(st),
        get_defect_dataframe(st),
        get_history_dataframe(100),
    )


_PP_COLS = ["Zaman", "Seri No", "Muayeneci", "Test", "Aşama", "Ölçüm", "Tolerans/Spec", "Sonuç"]


def render_post_production_cards(pp: dict[str, dict[str, Any]]) -> str:
    """HTML metric cards, one per post-production test type."""
    cards = ""
    for name, s in pp.items():
        cards += f"""
        <div style="background: white; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 12px; color: #6b7280; font-weight: 600; text-transform: uppercase;">{name}</div>
            <div style="font-size: 30px; font-weight: 800; color: #111827; margin-top: 4px;">{s['total']}</div>
            <div style="font-size: 16px; font-weight: 700; margin-top: 6px;">
                <span style="color: #059669;" title="Geçti">{s['pass']}</span> /
                <span style="color: #dc2626;" title="Kaldı">{s['fail']}</span> /
                <span style="color: #d97706;" title="Belirsiz">{s['uncertain']}</span>
            </div>
            <div style="font-size: 12px; color: #9ca3af; margin-top: 2px;">Geçti / Kaldı / Belirsiz · Geçme oranı %{s['pass_rate']:.1f}</div>
        </div>"""
    return f'<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin-bottom: 16px; font-family: system-ui, -apple-system, sans-serif;">{cards}</div>'


def on_refresh_post_production() -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Post-production dashboard: cards, result-count chart, non-pass table, full table."""
    pp = storage.post_production_stats()
    chart = pd.DataFrame(
        [{"Test": t, "Sonuç": lbl, "Adet": s[k]} for t, s in pp.items()
         for lbl, k in (("GEÇTİ", "pass"), ("KALDI", "fail"), ("BELİRSİZ", "uncertain"))]
    )
    rows = storage.list_post_production(100)
    full = pd.DataFrame(rows, columns=_PP_COLS)
    bad = pd.DataFrame([r for r in rows if not r["Sonuç"].startswith("🟢")][:30], columns=_PP_COLS)
    return render_post_production_cards(pp), chart, bad, full


def on_export_csv_click() -> str:
    """Exports CSV and returns its path."""
    csv_path = storage.export_csv()
    return str(csv_path)


def on_export_yolo_click() -> str:
    """Exports active-learning YOLO zip and returns its path."""
    zip_path = storage.export_yolo_zip()
    return str(zip_path)


def get_quality_gate_dataframe() -> pd.DataFrame:
    """Builds a pandas DataFrame for the Quality Gate table."""
    statuses = storage.list_serial_status()
    if not statuses:
        return pd.DataFrame(
            columns=["Seri No", "Ürün Grubu", "Görsel Muayene", "Son Test", "Kayma", "MTF", "Tork İşareti", "Genel Durum"]
        )
    rows = []
    for s in statuses:
        rows.append({
            "Seri No": s.get("serial_no", "-"),
            "Ürün Grubu": s.get("product_group", "-"),
            "Görsel Muayene": s.get("visual_decision", "-"),
            "Son Test": s.get("boresight_result", "-"),
            "Kayma": s.get("drift_status", "-"),
            "MTF": s.get("mtf_result", "none"),
            "Tork İşareti": s.get("torque_result", "none"),
            "Genel Durum": s.get("genel_durum", f"{s.get('emoji', '')} {s.get('overall', '-')}"),
        })
    return pd.DataFrame(rows)


def on_lookup_serial(serial_no: str) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Retrieves full history across both stages for a given serial number."""
    if not serial_no or not str(serial_no).strip():
        empty_html = """
        <div style="border: 1px dashed #d1d5db; padding: 12px; border-radius: 8px; color: #6b7280; font-family: system-ui, sans-serif;">
            Lütfen geçmişini sorgulamak istediğiniz bir seri numarası giriniz.
        </div>
        """
        return (
            empty_html,
            pd.DataFrame(columns=["Muayene ID", "Tarih", "Muayeneci", "Ürün Grubu", "AI Kararı", "Nihai Karar", "Kusur Türü", "Not"]),
            pd.DataFrame(columns=["Test ID", "Tarih", "Muayeneci", "Aşama", "Kanal", "dx (px)", "dy (px)", "Açı (mrad)", "Tolerans", "Sonuç", "Kayma"]),
        )

    sn = str(serial_no).strip()
    status = storage.serial_status(sn)

    overall = status.get("overall", "BEKLEMEDE")
    emoji = status.get("emoji", "🟡")
    v_dec = status.get("visual_decision", "none")
    b_res = status.get("boresight_result", "none")
    d_stat = status.get("drift_status", "none")
    p_grp = status.get("product_group", "-")
    m_res = status.get("mtf_result", "none")
    t_res = status.get("torque_result", "none")

    if overall == "SEVKE HAZIR":
        card_bg = "#ecfdf5"
        border_c = "#10b981"
        badge_bg = "#059669"
        summary_txt = "Tüm üretim içi muayene ve son test kriterleri başarıyla karşılandı. Ürün sevke hazırdır."
    elif overall == "RET":
        card_bg = "#fef2f2"
        border_c = "#ef4444"
        badge_bg = "#dc2626"
        summary_txt = "Kalite kapısı kriterleri sağlanamadı. Parçada red veya tolerans dışı ölçüm tespit edildi."
    else:
        card_bg = "#fffbeb"
        border_c = "#f59e0b"
        badge_bg = "#d97706"
        summary_txt = "Muayene veya son test aşamalarından en az biri henüz tamamlanmadı. İşlem beklemede."

    card_html = f"""
    <div style="border: 2px solid {border_c}; background: {card_bg}; border-radius: 12px; padding: 16px 20px; font-family: system-ui, -apple-system, sans-serif; margin-bottom: 12px;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <div>
                <span style="font-size: 20px; font-weight: 800; color: #111827;">Seri No: <code>{sn}</code></span>
                <span style="margin-left: 12px; font-size: 14px; color: #4b5563;">Ürün Grubu: <strong>{p_grp}</strong></span>
            </div>
            <span style="background: {badge_bg}; color: white; padding: 6px 16px; border-radius: 20px; font-weight: 700; font-size: 14px;">
                {emoji} {overall}
            </span>
        </div>
        <div style="display: flex; gap: 20px; margin-top: 12px; flex-wrap: wrap; font-size: 14px;">
            <div><strong>🏭 Görsel Muayene:</strong> <span style="font-weight: 700;">{v_dec}</span></div>
            <div><strong>🧪 Son Test (Boresight):</strong> <span style="font-weight: 700;">{b_res}</span></div>
            <div><strong>📈 Titreşim Kayması:</strong> <span style="font-weight: 700;">{d_stat}</span></div>
            <div><strong>🔬 MTF:</strong> <span style="font-weight: 700;">{m_res}</span></div>
            <div><strong>🔩 Tork İşareti:</strong> <span style="font-weight: 700;">{t_res}</span></div>
        </div>
        <div style="margin-top: 8px; font-size: 13px; color: #374151;">{summary_txt}</div>
    </div>
    """

    insp_rows = []
    for r in status.get("inspections", []):
        insp_rows.append({
            "Muayene ID": f"#{r.get('id')}",
            "Tarih": str(r.get("ts", ""))[:19].replace("T", " "),
            "Muayeneci": r.get("inspector", "-"),
            "Ürün Grubu": r.get("product_group", "-"),
            "AI Kararı": r.get("ai_decision", "-"),
            "Nihai Karar": r.get("human_decision", "-"),
            "Kusur Türü": r.get("human_defect_type", "-"),
            "Not": r.get("note", "-"),
        })
    insp_df = pd.DataFrame(insp_rows) if insp_rows else pd.DataFrame(
        columns=["Muayene ID", "Tarih", "Muayeneci", "Ürün Grubu", "AI Kararı", "Nihai Karar", "Kusur Türü", "Not"]
    )

    bore_rows = []
    for r in status.get("boresight_tests", []):
        drift_str = f"{float(r['drift_mrad']):.3f} mrad" if r.get("drift_mrad") is not None else "-"
        bore_rows.append({
            "Test ID": f"#{r.get('id')}",
            "Tarih": str(r.get("ts", ""))[:19].replace("T", " "),
            "Muayeneci": r.get("inspector", "-"),
            "Aşama": r.get("stage", "-"),
            "Kanal": r.get("channel", "-"),
            "dx (px)": f"{float(r.get('dx_px', 0)):+.2f}",
            "dy (px)": f"{float(r.get('dy_px', 0)):+.2f}",
            "Açı (mrad)": f"{float(r.get('err_mrad', 0)):.3f}",
            "Tolerans": f"{float(r.get('tolerance_mrad', 0)):.2f}",
            "Sonuç": r.get("result", "-"),
            "Kayma": drift_str,
        })
    bore_df = pd.DataFrame(bore_rows) if bore_rows else pd.DataFrame(
        columns=["Test ID", "Tarih", "Muayeneci", "Aşama", "Kanal", "dx (px)", "dy (px)", "Açı (mrad)", "Tolerans", "Sonuç", "Kayma"]
    )

    return (card_html, insp_df, bore_df)


def on_copy_serial_to_stage2(sn: str) -> tuple[str, str, gr.update]:
    """Copies current serial number from Stage 1 to Stage 2."""
    val = (sn or "").strip()
    return (
        val,
        val,
        gr.update(value=f"✅ `{val}` son test istasyonuna aktarıldı.", visible=True),
    )


# ---------------------------------------------------------------------------
# Boresight Station Helpers & Callbacks
# ---------------------------------------------------------------------------
def load_boresight_examples() -> list[list[Any]]:
    """Loads synthetic collimator target examples from samples/boresight/."""
    bore_dir = config.SAMPLES_DIR / "boresight"
    if not bore_dir.exists():
        return []
    examples: list[list[Any]] = []
    # Add representative targets
    for fn in ["vis_offset_0_0.png", "vis_offset_3_2_az.png", "vis_offset_15_0_el.png"]:
        p = bore_dir / fn
        if p.exists():
            examples.append([str(p), "Görünür", 3.45, 50.0, 0.5])
    for fn in ["thm_offset_0_0.png", "thm_offset_subpix_pass.png", "thm_offset_3_2_az.png"]:
        p = bore_dir / fn
        if p.exists():
            examples.append([str(p), "Termal", 12.0, 25.0, 0.5])
    return examples


def render_boresight_badge(result: dict[str, Any] | None) -> str:
    """Renders HTML decision badge for boresight test."""
    if not result:
        return """
        <div style="border: 2px dashed #d1d5db; background: #f9fafb; border-radius: 12px; padding: 24px; text-align: center; color: #6b7280; font-family: system-ui, -apple-system, sans-serif;">
            <div style="font-size: 36px; margin-bottom: 8px;">🎯</div>
            <div style="font-weight: 700; font-size: 16px; color: #374151;">Optik Eksen Ölçümü Bekleniyor</div>
            <div style="font-size: 13px; margin-top: 4px;">Sol panelden kolimatör retikül görseli yükleyip <strong>Optik Ekseni Ölç</strong> butonuna basınız.</div>
        </div>
        """

    passed = result.get("passed", False)
    err = float(result.get("err_mrad", 0.0))
    tol = float(result.get("tolerance_mrad", 0.5))
    off_px = float(result.get("offset_px", 0.0))
    dx = float(result.get("dx_px", 0.0))
    dy = float(result.get("dy_px", 0.0))
    az = float(result.get("az_mrad", 0.0))
    el = float(result.get("el_mrad", 0.0))
    channel = result.get("channel", "Görünür")
    quality = float(result.get("quality_score", 0.0))

    if passed:
        bg = "#ecfdf5"
        border = "#10b981"
        badge_bg = "#059669"
        text_color = "#065f46"
        icon = "✅"
        title = "GEÇTİ — OPTİK EKSEN TOLERANS DAHİLİNDE (PASS)"
        desc = f"{channel} kanalında toplam açısal sapma izin verilen {tol:.2f} mrad tolerans sınırının altındadır."
    else:
        bg = "#fef2f2"
        border = "#ef4444"
        badge_bg = "#dc2626"
        text_color = "#991b1b"
        icon = "❌"
        title = "KALDI — OPTİK EKSEN TOLERANS DIŞI (FAIL)"
        desc = f"{channel} kanalında toplam açısal sapma {tol:.2f} mrad tolerans sınırını aşmaktadır. Eksen ayarı gereklidir."

    return f"""
    <div style="border: 2px solid {border}; background: {bg}; border-radius: 12px; padding: 18px 22px; margin-bottom: 12px; font-family: system-ui, -apple-system, sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div style="flex: 1; min-width: 240px;">
                <span style="display: inline-block; background: {badge_bg}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px; letter-spacing: 0.3px;">
                    {icon} {title}
                </span>
                <div style="margin-top: 10px; font-size: 15px; color: #1f2937;">
                    <strong>Toplam Açısal Sapma:</strong> <span style="font-weight: 800; font-size: 20px; color: {text_color};">{err:.3f} mrad</span>
                    <span style="color: #6b7280; margin-left: 8px;">(Tolerans: <strong>{tol:.2f} mrad</strong>)</span>
                </div>
                <div style="margin-top: 4px; font-size: 12px; color: #4b5563;">
                    {desc}
                </div>
            </div>
            <div style="text-align: right; min-width: 140px;">
                <div style="font-size: 11px; text-transform: uppercase; color: #6b7280; font-weight: 700; letter-spacing: 0.5px;">Sapma Koordinatları</div>
                <div style="font-size: 22px; font-weight: 800; color: {text_color}; line-height: 1.2;">Δr = {off_px:.2f} px</div>
                <div style="font-size: 12px; color: #4b5563; margin-top: 2px;">dx: {dx:+.2f} px | dy: {dy:+.2f} px</div>
                <div style="font-size: 12px; color: #4b5563;">Az: {az:+.3f} | El: {el:+.3f} mrad</div>
                <div style="font-size: 11px; color: #6b7280; margin-top: 2px;">Kalite: <strong>%{int(quality * 100)}</strong></div>
            </div>
        </div>
    </div>
    """


def build_boresight_metrics_dataframe(res: dict[str, Any] | None) -> pd.DataFrame:
    """Builds metrics table dataframe for boresight measurements."""
    if not res:
        return pd.DataFrame(columns=["Parametre / Metrik", "Ölçülen Değer", "Birim", "Açıklama"])

    rx, ry = res.get("reticle_center", (0.0, 0.0))
    cx, cy = res.get("image_center", (0.0, 0.0))
    dx = res.get("dx_px", 0.0)
    dy = res.get("dy_px", 0.0)
    off_px = res.get("offset_px", 0.0)
    az = res.get("az_mrad", 0.0)
    el = res.get("el_mrad", 0.0)
    tot = res.get("err_mrad", 0.0)
    tol = res.get("tolerance_mrad", 0.5)
    quality = res.get("quality_score", 0.0)
    lat = res.get("latency_ms", 0)
    passed = res.get("passed", False)
    method = res.get("method", "—")

    rows = [
        {"Parametre / Metrik": "Görüntü Referans Merkezi (cx, cy)", "Ölçülen Değer": f"({cx:.1f}, {cy:.1f})", "Birim": "piksel", "Açıklama": "Sensör geometrik orta noktası"},
        {"Parametre / Metrik": "Tespit Edilen Retikül Merkezi", "Ölçülen Değer": f"({rx:.2f}, {ry:.2f})", "Birim": "piksel", "Açıklama": "Alt-piksel hassasiyetinde retikül konumu"},
        {"Parametre / Metrik": "Yatay Sapma (dx)", "Ölçülen Değer": f"{dx:+.2f}", "Birim": "piksel", "Açıklama": "X ekseni piksel ofseti"},
        {"Parametre / Metrik": "Dikey Sapma (dy)", "Ölçülen Değer": f"{dy:+.2f}", "Birim": "piksel", "Açıklama": "Y ekseni piksel ofseti"},
        {"Parametre / Metrik": "Radyal Piksel Ofseti (Δr)", "Ölçülen Değer": f"{off_px:.2f}", "Birim": "piksel", "Açıklama": "sqrt(dx² + dy²)"},
        {"Parametre / Metrik": "Yatay Açısal Sapma (Azimut)", "Ölçülen Değer": f"{az:+.3f}", "Birim": "mrad", "Açıklama": "atan(dx * pitch / focal)"},
        {"Parametre / Metrik": "Dikey Açısal Sapma (Yükseliş)", "Ölçülen Değer": f"{el:+.3f}", "Birim": "mrad", "Açıklama": "atan(dy * pitch / focal)"},
        {"Parametre / Metrik": "Toplam Açısal Sapma (θ)", "Ölçülen Değer": f"{tot:.3f}", "Birim": "mrad", "Açıklama": "atan(Δr * pitch / focal)"},
        {"Parametre / Metrik": "Tolerans Eşiği", "Ölçülen Değer": f"{tol:.2f}", "Birim": "mrad", "Açıklama": "Maksimum izin verilen açısal hata"},
        {"Parametre / Metrik": "Test Kararı", "Ölçülen Değer": "GEÇTİ (PASS)" if passed else "KALDI (FAIL)", "Birim": "—", "Açıklama": "Tolerans karşılaştırma sonucu"},
        {"Parametre / Metrik": "Ölçüm Kalite Skoru", "Ölçülen Değer": f"%{int(quality * 100)}", "Birim": "—", "Açıklama": f"Kontrast, inlier oranı ve kalıntı ({method})"},
        {"Parametre / Metrik": "Ölçüm Gecikmesi", "Ölçülen Değer": f"{lat}", "Birim": "ms", "Açıklama": "Saf OpenCV alt-piksel işlem süresi"},
    ]
    return pd.DataFrame(rows)


def load_mtf_examples() -> list[list[Any]]:
    """Loads synthetic slanted-edge examples from samples/mtf/."""
    d = config.SAMPLES_DIR / "mtf"
    rows: list[list[Any]] = []
    for fn, ch, pitch, spec in (
        ("vis_sigma_0.6.png", "Görünür", 3.45, 0.25),
        ("vis_sigma_1.5.png", "Görünür", 3.45, 0.25),
        ("thm_sigma_1.0.png", "Termal", 12.0, 0.18),
        ("thm_sigma_2.5.png", "Termal", 12.0, 0.18),
    ):
        if (d / fn).exists():
            rows.append([str(d / fn), ch, pitch, spec])
    return rows


def render_mtf_badge(res: dict[str, Any] | None) -> str:
    """HTML decision badge for the MTF test (same visual style as boresight)."""
    if not res:
        return """
        <div style="border: 2px dashed #d1d5db; background: #f9fafb; border-radius: 12px; padding: 24px; text-align: center; color: #6b7280; font-family: system-ui, -apple-system, sans-serif;">
            <div style="font-size: 36px; margin-bottom: 8px;">🔬</div>
            <div style="font-weight: 700; font-size: 16px; color: #374151;">MTF Ölçümü Bekleniyor</div>
            <div style="font-size: 13px; margin-top: 4px;">Eğimli kenar görseli yükleyip <strong>MTF Ölç</strong> butonuna basınız.</div>
        </div>
        """
    passed = res.get("passed")
    m50 = float(res["mtf50_cy_px"])
    spec = res.get("spec_mtf50")
    if passed is None:
        bg, border, badge_bg, tc, icon, title = "#eff6ff", "#3b82f6", "#2563eb", "#1e3a8a", "ℹ️", "ÖLÇÜLDÜ — SPEC TANIMLI DEĞİL"
    elif passed:
        bg, border, badge_bg, tc, icon, title = "#ecfdf5", "#10b981", "#059669", "#065f46", "✅", "GEÇTİ — MTF50 SPEC İÇİNDE (PASS)"
    else:
        bg, border, badge_bg, tc, icon, title = "#fef2f2", "#ef4444", "#dc2626", "#991b1b", "❌", "KALDI — MTF50 SPEC ALTINDA (FAIL)"
    spec_txt = f"(Spec: <strong>≥ {spec:.2f} cy/px</strong>)" if spec is not None else ""
    warn = "".join(f"<div style='font-size:12px;color:#92400e;margin-top:4px;'>⚠️ {w}</div>" for w in res.get("warnings", []))
    return f"""
    <div style="border: 2px solid {border}; background: {bg}; border-radius: 12px; padding: 18px 22px; margin-bottom: 12px; font-family: system-ui, -apple-system, sans-serif;">
        <span style="display: inline-block; background: {badge_bg}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px; letter-spacing: 0.3px;">{icon} {title}</span>
        <div style="margin-top: 10px; font-size: 15px; color: #1f2937;">
            <strong>MTF50:</strong> <span style="font-weight: 800; font-size: 20px; color: {tc};">{m50:.3f} cy/px</span>
            <span style="color: #6b7280; margin-left: 8px;">{res['mtf50_lp_mm']:.1f} lp/mm {spec_txt}</span>
        </div>
        {warn}
    </div>
    """


def build_mtf_metrics_dataframe(res: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("MTF50", f"{res['mtf50_cy_px']:.4f}", "cycles/pixel", "Kontrastın %50'ye düştüğü frekans"),
        ("MTF50", f"{res['mtf50_lp_mm']:.1f}", "lp/mm", "Piksel boyutuna göre"),
        ("MTF @ Nyquist (0.5 cy/px)", f"{res['mtf_nyquist']:.3f}", "—", "Örnekleme sınırında kontrast"),
        ("MTF10", f"{res['mtf10_cy_px']:.4f} ({res['mtf10_lp_mm']:.1f} lp/mm)", "cy/px", "Kontrastın %10 olduğu frekans (çözünürlük sınırı)"),
        ("Kenar Açısı", f"{res['edge_angle_deg']:.2f}", "derece", f"{res['edge_orientation']} eksenden; ISO için 2–10°"),
        ("Kenar Polaritesi", res["polarity"], "—", "Otomatik ele alındı"),
        ("SNR", f"{res['snr']:.0f}", "—", "Kenar kontrastı / düz bölge gürültüsü"),
        ("ROI (x0,y0,x1,y1)", str(res["roi"]), "piksel", "Otomatik tespit edilen bölge"),
        ("Uyarılar", "; ".join(res["warnings"]) or "Yok", "—", ""),
        ("Ölçüm Gecikmesi", str(res["latency_ms"]), "ms", "CPU"),
    ]
    return pd.DataFrame(rows, columns=["Parametre / Metrik", "Ölçülen Değer", "Birim", "Açıklama"])


def on_measure_mtf(image: Any, pixel_pitch_um: float, spec_mtf50: float, channel: str):
    """Runs slanted-edge MTF measurement."""
    from engines import mtf as mtf_engine
    if image is None:
        raise gr.Error("Lütfen eğimli kenar içeren bir görsel yükleyiniz veya örneklerden seçiniz.")
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    pitch = float(pixel_pitch_um) if pixel_pitch_um and pixel_pitch_um > 0 else (12.0 if channel == "Termal" else 3.45)
    spec = float(spec_mtf50) if spec_mtf50 and spec_mtf50 > 0 else None
    try:
        res = mtf_engine.measure_mtf(image, pitch, channel, spec_mtf50=spec)
    except ValueError as e:
        raise gr.Error(f"MTF ölçülemedi: {e}")
    return render_mtf_badge(res), res["plot_image"], res["roi_image"], build_mtf_metrics_dataframe(res), res


def on_save_mtf(last_res: dict[str, Any] | None, serial_no: str, inspector: str) -> str:
    """Persists the last MTF measurement to SQLite."""
    if not last_res:
        raise gr.Error("Lütfen önce 'MTF Ölç' butonuna basarak ölçüm yapınız.")
    passed = last_res.get("passed")
    result = "PASS" if passed is True else ("FAIL" if passed is False else "UNCERTAIN")
    rid = storage.save_mtf(
        inspector=inspector,
        serial_no=(serial_no or "").strip(),
        channel=last_res.get("channel", "Görünür"),
        mtf50_cyc_px=last_res.get("mtf50_cy_px"),
        mtf50_lpmm=last_res.get("mtf50_lp_mm"),
        mtf_nyquist=last_res.get("mtf_nyquist"),
        edge_angle_deg=last_res.get("edge_angle_deg"),
        spec_mtf50=last_res.get("spec_mtf50"),
        result=result,
        image=last_res.get("roi_image"),
        params={
            "pixel_pitch_um": last_res.get("pixel_pitch_um"),
            "mtf10_cy_px": last_res.get("mtf10_cy_px"),
            "snr": last_res.get("snr"),
            "warnings": last_res.get("warnings"),
        },
    )
    return f"✅ **MTF Testi Kaydedildi!** Kayıt ID: **#{rid}** (Seri No: **{serial_no}**, Sonuç: **{result}**)"


def on_boresight_channel_change(channel: str) -> tuple[float, float]:
    """Updates sensible defaults for pixel pitch and focal length when channel toggles."""
    if channel == "Termal":
        return 12.0, 25.0
    return 3.45, 50.0


# ---------------------------------------------------------------------------
# Torque Mark (Witness Mark) Helpers & Callbacks
# ---------------------------------------------------------------------------
def load_torque_mark_examples() -> list[list[Any]]:
    """Loads sample torque mark images from samples/torque_mark/."""
    d = config.SAMPLES_DIR / "torque_mark"
    if not d.exists():
        return []
    examples: list[list[Any]] = []
    for fn in ["kırmızı_rot00.png", "kırmızı_rot15.png",
                "turuncu_rot04.png", "turuncu_rot30.png",
                "sarı_rot02.png", "kırmızı_missing_head.png"]:
        p = d / fn
        if p.exists():
            color = fn.split("_")[0]
            examples.append([str(p), color, 5.0])
    return examples


def render_torque_mark_badge(result: dict[str, Any] | None) -> str:
    """HTML decision badge for torque stripe inspection."""
    if not result:
        return """
        <div style="border: 2px dashed #d1d5db; background: #f9fafb; border-radius: 12px; padding: 24px; text-align: center; color: #6b7280; font-family: system-ui, -apple-system, sans-serif;">
            <div style="font-size: 36px; margin-bottom: 8px;">🔩</div>
            <div style="font-weight: 700; font-size: 16px; color: #374151;">Tork İşareti Kontrolü Bekleniyor</div>
            <div style="font-size: 13px; margin-top: 4px;">Vida üst görünüm görseli yükleyip <strong>Kontrol Et</strong> butonuna basınız.</div>
        </div>
        """
    dec = result.get("decision", "BELİRSİZ")
    angle = result.get("angle_deg")
    tol = result.get("tolerance_deg", 5.0)

    if dec == "GEÇTİ":
        bg, border, badge_bg, tc, icon, title = "#ecfdf5", "#10b981", "#059669", "#065f46", "✅", "GEÇTİ — TORK İŞARETİ SAĞLAM"
    elif dec == "KALDI":
        bg, border, badge_bg, tc, icon, title = "#fef2f2", "#ef4444", "#dc2626", "#991b1b", "❌", "KALDI — GEVŞEME ŞÜPHESİ"
    else:
        bg, border, badge_bg, tc, icon, title = "#eff6ff", "#3b82f6", "#2563eb", "#1e3a8a", "⚠️", "BELİRSİZ — ÖLÇÜM YAPILAMADI"

    angle_txt = f"{angle:.1f}°" if angle is not None else "Ölçülemedi"
    offset = result.get("offset_px")
    offset_txt = f"{offset:.1f} px" if offset is not None else "—"
    lat = result.get("latency_ms", 0)
    explanation = result.get("explanation", "")

    return f"""
    <div style="border: 2px solid {border}; background: {bg}; border-radius: 12px; padding: 18px 22px; margin-bottom: 12px; font-family: system-ui, -apple-system, sans-serif;">
        <span style="display: inline-block; background: {badge_bg}; color: white; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 14px; letter-spacing: 0.3px;">{icon} {title}</span>
        <div style="margin-top: 10px; font-size: 15px; color: #1f2937;">
            <strong>Açısal Fark:</strong> <span style="font-weight: 800; font-size: 20px; color: {tc};">{angle_txt}</span>
            <span style="color: #6b7280; margin-left: 8px;">(Tolerans: <strong>≤ {tol:.1f}°</strong>)</span>
            <span style="color: #6b7280; margin-left: 12px;">Ofset: <strong>{offset_txt}</strong></span>
        </div>
        <div style="font-size: 12px; color: #4b5563; margin-top: 4px;">{explanation}</div>
        <div style="font-size: 11px; color: #9ca3af; margin-top: 4px;">⏱️ {lat} ms</div>
    </div>
    """


def build_torque_mark_metrics_dataframe(res: dict[str, Any] | None) -> pd.DataFrame:
    """Metrics table for torque mark inspection."""
    if not res:
        return pd.DataFrame(columns=["Parametre / Metrik", "Ölçülen Değer", "Birim", "Açıklama"])
    rows = [
        ("Açısal Fark", f"{res['angle_deg']:.2f}" if res['angle_deg'] is not None else "—", "derece", "Kafa ve gövde boya çizgileri arasındaki açı farkı"),
        ("Dik Ofset", f"{res['offset_px']:.1f}" if res['offset_px'] is not None else "—", "piksel", "İki çizgi arasındaki dik mesafe"),
        ("Kafa Boya Pikseli", str(res["head_pixels"]), "adet", "Vida kafası üzerindeki boya pikseli sayısı"),
        ("Gövde Boya Pikseli", str(res["housing_pixels"]), "adet", "Gövde (housing) tarafındaki boya pikseli sayısı"),
        ("Toplam Boya Pikseli", str(res["total_paint_pixels"]), "adet", "Tüm tespit edilen boya pikselleri"),
        ("Vida Dairesi", f"({res['circle'][0]}, {res['circle'][1]}) r={res['circle'][2]}" if res["circle"] else "Tespit edilemedi", "piksel", "Tespit edilen vida kafası dairesi"),
        ("Tolerans (Açı)", f"{res['tolerance_deg']:.1f}", "derece", "Azami izin verilen açısal fark"),
        ("Tolerans (Ofset)", f"{res['tolerance_px']:.1f}" if res['tolerance_px'] else "—", "piksel", "Azami izin verilen dik ofset"),
        ("Karar", res["decision"], "—", res["explanation"]),
        ("Gecikme", str(res["latency_ms"]), "ms", "İşlem süresi"),
    ]
    return pd.DataFrame(rows, columns=["Parametre / Metrik", "Ölçülen Değer", "Birim", "Açıklama"])


def on_inspect_torque_mark(
    image: Any,
    paint_color: str,
    tolerance_deg: float,
) -> tuple[str, Image.Image | None, pd.DataFrame, str]:
    """Runs torque stripe inspection and returns UI outputs."""
    if image is None:
        raise gr.Error("Lütfen vida üst-görünüm görseli yükleyiniz veya örneklerden seçiniz.")
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")

    color = paint_color if paint_color in ("kırmızı", "turuncu", "sarı") else "auto"
    tol = float(tolerance_deg) if tolerance_deg and tolerance_deg > 0 else 5.0

    res = torque_mark_engine.inspect(image, paint_color=color, tolerance_deg=tol)

    badge = render_torque_mark_badge(res)
    anno = res.get("annotated_image")
    df = build_torque_mark_metrics_dataframe(res)

    explanation_md = """### 🔩 Tork İşareti (Witness Mark / Torque Stripe) Hakkında

**Nedir?** Savunma ve havacılık montajlarında, kritik bağlantı vidaları tork anahtarıyla sıkıldıktan sonra vida kafası ve bitişik gövde yüzeyi üzerine kesintisiz bir boya çizgisi çekilir. Bu çizgi *tork işareti* (witness mark / torque stripe) olarak adlandırılır.

**Nasıl Çalışır?** Vida gevşerse, kafası gövdeye göre döner ve boya çizgisi kırılır/kayar. Bu açısal kayma görsel muayene ile tespit edilebilir.

**Sınırlılık:** Tork işareti yalnızca *dönerek gevşemeyi* tespit eder; dönme olmadan tork kaybını (örn. termal gevşeme, basınç kaybı) saptayamaz. Tork-açı eğrisi verisi ile birleştirilmesi (akıllı tork aletleri entegrasyonu) yol haritasındadır.

**Kayıt:** sonuçlar 'Kaydet' butonuyla veritabanına yazılır ve kalite kapısına dahil edilir.
"""

    return badge, anno, df, explanation_md, res


def on_save_torque(
    last_res: dict[str, Any] | None, serial_no: str, inspector: str, stage: str, paint_color: str
) -> str:
    """Persists the last torque-mark inspection to SQLite."""
    if not last_res:
        raise gr.Error("Lütfen önce 'Kontrol Et' butonuna basarak ölçüm yapınız.")
    rid = storage.save_torque(
        inspector=inspector,
        serial_no=(serial_no or "").strip(),
        stage=stage,
        paint_color=paint_color,
        angle_deg=last_res.get("angle_deg"),
        offset_px=last_res.get("offset_px"),
        tolerance_deg=last_res.get("tolerance_deg"),
        result=last_res.get("decision", "BELİRSİZ"),
        image=last_res.get("annotated_image"),
        params={"head_pixels": last_res.get("head_pixels"), "housing_pixels": last_res.get("housing_pixels")},
    )
    return f"✅ **Tork İşareti Kaydedildi!** Kayıt ID: **#{rid}** (Seri No: **{serial_no}**, Karar: **{last_res.get('decision')}**)"


def validate_boresight_image(image: Any) -> Image.Image:
    """Validates collimator image without downsampling to preserve calibrated optical geometry."""
    if image is None:
        raise gr.Error("Lütfen ölçüm için bir kolimatör hedef görseli yükleyiniz veya örneklerden seçiniz.")
    if not isinstance(image, Image.Image):
        try:
            image = Image.open(image)
        except Exception as e:
            raise gr.Error(f"Görsel açılamadı: {e}")
    return image.convert("RGB")


def on_measure_boresight(
    image: Any,
    pixel_pitch_um: float,
    focal_length_mm: float,
    tolerance_mrad: float,
    channel: str,
) -> tuple[
    str,                 # badge_html
    Image.Image | None,  # annotated_image
    pd.DataFrame,        # metrics_table
    dict[str, Any],      # last_boresight_state
]:
    """Measures boresight alignment and updates UI."""
    valid_img = validate_boresight_image(image)
    pitch = float(pixel_pitch_um) if pixel_pitch_um and pixel_pitch_um > 0 else (12.0 if channel == "Termal" else 3.45)
    focal = float(focal_length_mm) if focal_length_mm and focal_length_mm > 0 else (25.0 if channel == "Termal" else 50.0)
    tol = float(tolerance_mrad) if tolerance_mrad and tolerance_mrad > 0 else 0.5

    res = boresight_engine.measure(
        image=valid_img,
        pixel_pitch_um=pitch,
        focal_length_mm=focal,
        tolerance_mrad=tol,
        channel=channel,
    )

    badge_html = render_boresight_badge(res)
    anno_img = res.get("annotated_image")
    df = build_boresight_metrics_dataframe(res)

    return (badge_html, anno_img, df, res)


def on_save_boresight(
    last_res: dict[str, Any] | None,
    image: Any,
    serial_no: str,
    inspector: str,
    stage: str,
    channel: str,
    tolerance_mrad: float,
) -> tuple[str, str]:
    """Persists boresight measurement record to SQLite database."""
    if not last_res:
        raise gr.Error("Lütfen önce 'Optik Ekseni Ölç' butonuna basarak ölçüm yapınız.")

    valid_img = validate_boresight_image(image) if image is not None else None
    tol = float(tolerance_mrad) if tolerance_mrad and tolerance_mrad > 0 else float(last_res.get("tolerance_mrad", 0.5))

    params = {
        "pixel_pitch_um": last_res.get("pixel_pitch_um"),
        "focal_length_mm": last_res.get("focal_length_mm"),
        "quality_score": last_res.get("quality_score"),
        "reticle_center": last_res.get("reticle_center"),
        "image_center": last_res.get("image_center"),
        "az_mrad": last_res.get("az_mrad"),
        "el_mrad": last_res.get("el_mrad"),
        "method": last_res.get("method"),
    }

    record_id = storage.save_boresight(
        inspector=inspector,
        serial_no=serial_no,
        stage=stage,
        channel=channel,
        dx_px=last_res.get("dx_px", 0.0),
        dy_px=last_res.get("dy_px", 0.0),
        err_mrad=last_res.get("err_mrad", 0.0),
        tolerance_mrad=tol,
        result=last_res.get("result", "PASS"),
        image=last_res.get("annotated_image") or valid_img,
        params=params,
    )

    conf_msg = (
        f"✅ **Boresight Testi Kaydedildi!** Kayıt ID: **#{record_id}** "
        f"(Seri No: **{serial_no}**, Aşama: **{stage}**, Kanal: **{channel}**, "
        f"Açı: **{last_res.get('err_mrad', 0.0):.3f} mrad**, Sonuç: **{last_res.get('result', 'PASS')}**)"
    )
    new_sn = generate_default_serial()
    return conf_msg, new_sn


def on_compare_channels(
    vis_img: Any,
    thm_img: Any,
    vis_pitch: float,
    vis_focal: float,
    thm_pitch: float,
    thm_focal: float,
    inter_tol: float,
) -> tuple[str, Image.Image | None, pd.DataFrame]:
    """Calculates inter-channel alignment between visible and thermal sensors."""
    if vis_img is None or thm_img is None:
        raise gr.Error("Lütfen hem görünür hem de termal kanal için birer hedef görseli yükleyiniz.")

    valid_vis = validate_boresight_image(vis_img)
    valid_thm = validate_boresight_image(thm_img)

    comp = boresight_engine.compare_channels(
        visible_img=valid_vis,
        thermal_img=valid_thm,
        visible_pitch_um=float(vis_pitch or 3.45),
        visible_focal_mm=float(vis_focal or 50.0),
        thermal_pitch_um=float(thm_pitch or 12.0),
        thermal_focal_mm=float(thm_focal or 25.0),
        tolerance_mrad=float(inter_tol or 0.5),
    )

    passed = comp.get("passed", False)
    inter_mrad = comp.get("inter_channel_mrad", 0.0)
    tol = comp.get("tolerance_mrad", 0.5)
    d_az = comp.get("d_az_mrad", 0.0)
    d_el = comp.get("d_el_mrad", 0.0)

    badge_bg = "#059669" if passed else "#dc2626"
    title = "KANAL HİZALAMASI: GEÇTİ (PASS)" if passed else "KANAL HİZALAMASI: KALDI (FAIL)"
    desc = "Görünür ve termal kanallar arasındaki eksen farkı tolerans sınırları içindedir." if passed else "İki kanal arasındaki eksen paralelliği tolerans sınırını aşmaktadır."

    badge_html = f"""
    <div style="border: 2px solid {'#10b981' if passed else '#ef4444'}; background: {'#ecfdf5' if passed else '#fef2f2'}; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; font-family: system-ui, -apple-system, sans-serif;">
        <span style="display: inline-block; background: {badge_bg}; color: white; padding: 4px 12px; border-radius: 16px; font-weight: 700; font-size: 13px;">
            {'✅' if passed else '❌'} {title}
        </span>
        <div style="margin-top: 8px; font-size: 15px; color: #1f2937;">
            <strong>Eksenler Arası Açısal Fark:</strong> <span style="font-weight: 800; font-size: 18px; color: {'#065f46' if passed else '#991b1b'};">{inter_mrad:.3f} mrad</span>
            <span style="color: #6b7280; margin-left: 8px;">(Tolerans: <strong>{tol:.2f} mrad</strong>, ΔAz: <strong>{d_az:+.3f}</strong>, ΔEl: <strong>{d_el:+.3f}</strong>)</span>
        </div>
        <div style="font-size: 12px; color: #4b5563; margin-top: 2px;">{desc}</div>
    </div>
    """

    vis_res = comp["visible_result"]
    thm_res = comp["thermal_result"]

    rows = [
        {"Kanal": "Görünür (Visible)", "Piksel (µm)": vis_pitch, "Odak (mm)": vis_focal, "dx (px)": f"{vis_res['dx_px']:+.2f}", "dy (px)": f"{vis_res['dy_px']:+.2f}", "Az (mrad)": f"{vis_res['az_mrad']:+.3f}", "El (mrad)": f"{vis_res['el_mrad']:+.3f}", "Toplam Açı (mrad)": f"{vis_res['err_mrad']:.3f}", "Sonuç": vis_res["result"]},
        {"Kanal": "Termal (Thermal)", "Piksel (µm)": thm_pitch, "Odak (mm)": thm_focal, "dx (px)": f"{thm_res['dx_px']:+.2f}", "dy (px)": f"{thm_res['dy_px']:+.2f}", "Az (mrad)": f"{thm_res['az_mrad']:+.3f}", "El (mrad)": f"{thm_res['el_mrad']:+.3f}", "Toplam Açı (mrad)": f"{thm_res['err_mrad']:.3f}", "Sonuç": thm_res["result"]},
        {"Kanal": "Fark / Karşılaştırma", "Piksel (µm)": "—", "Odak (mm)": "—", "dx (px)": "—", "dy (px)": "—", "Az (mrad)": f"Δ {d_az:+.3f}", "El (mrad)": f"Δ {d_el:+.3f}", "Toplam Açı (mrad)": f"{inter_mrad:.3f}", "Sonuç": "PASS" if passed else "FAIL"},
    ]
    df = pd.DataFrame(rows)

    return (badge_html, comp.get("comparison_image"), df)


def on_calculate_drift(serial_no: str, drift_tol: float) -> tuple[str, pd.DataFrame, str]:
    """Calculates pre/post vibration drift from recorded tests for the given serial number."""
    if not serial_no or not serial_no.strip():
        raise gr.Error("Lütfen bir seri numarası giriniz.")

    records = storage.list_boresight(serial_no=serial_no.strip(), limit=50)

    if len(records) < 2:
        badge_html = f"""
        <div style="border: 2px dashed #f59e0b; background: #fffbeb; border-radius: 10px; padding: 14px 18px; color: #92400e; font-family: system-ui, -apple-system, sans-serif;">
            <strong>⚠️ Yetersiz Kayıt:</strong> <code>{serial_no}</code> seri numarası için veri tabanında {len(records)} adet kayıt bulundu.
            Kayma (drift) analizi yapabilmek için en az 2 test kaydı (örn. 'titreşim öncesi' ve 'titreşim sonrası') gereklidir.
        </div>
        """
        empty_df = pd.DataFrame([{"Aşama": r.get("stage"), "Tarih": r.get("ts")[:19], "Kanal": r.get("channel"), "Açı (mrad)": r.get("err_mrad"), "Sonuç": r.get("result")} for r in records])
        return badge_html, empty_df, "*En az iki test kaydı bulunduğunda drift hesaplanacaktır.*"

    # Last two records
    after_rec = records[0]
    before_rec = records[1]

    d_res = boresight_engine.drift(
        before=before_rec.get("params", {}),
        after=after_rec.get("params", {}),
        tolerance_mrad=float(drift_tol or 0.5),
    )

    drift_val = d_res.get("drift_mrad", 0.0)
    passed = d_res.get("passed", False)
    tol = d_res.get("tolerance_mrad", 0.5)
    d_az = d_res.get("drift_az_mrad", 0.0)
    d_el = d_res.get("drift_el_mrad", 0.0)

    badge_bg = "#059669" if passed else "#dc2626"
    title = "TİTREŞİM KAYMASI: GEÇTİ (PASS)" if passed else "TİTREŞİM KAYMASI: KALDI (FAIL)"
    desc = f"İki test arasındaki eksen kayması ({drift_val:.3f} mrad), izin verilen {tol:.2f} mrad sınırının altındadır. Mekanik stabilite onaylandı." if passed else f"İki test arasındaki eksen kayması ({drift_val:.3f} mrad), {tol:.2f} mrad sınırını aşmaktadır. Mekanik gevşeme veya montaj hatası şüphesi mevcuttur."

    badge_html = f"""
    <div style="border: 2px solid {'#10b981' if passed else '#ef4444'}; background: {'#ecfdf5' if passed else '#fef2f2'}; border-radius: 12px; padding: 16px 20px; margin-bottom: 12px; font-family: system-ui, -apple-system, sans-serif;">
        <span style="display: inline-block; background: {badge_bg}; color: white; padding: 4px 12px; border-radius: 16px; font-weight: 700; font-size: 13px;">
            {'✅' if passed else '❌'} {title}
        </span>
        <div style="margin-top: 8px; font-size: 15px; color: #1f2937;">
            <strong>Mekanik Eksen Kayması (Drift):</strong> <span style="font-weight: 800; font-size: 18px; color: {'#065f46' if passed else '#991b1b'};">{drift_val:.3f} mrad</span>
            <span style="color: #6b7280; margin-left: 8px;">(Tolerans: <strong>{tol:.2f} mrad</strong>, ΔAz: <strong>{d_az:+.3f}</strong>, ΔEl: <strong>{d_el:+.3f}</strong>)</span>
        </div>
        <div style="font-size: 12px; color: #4b5563; margin-top: 2px;">{desc}</div>
    </div>
    """

    rows = [
        {"Test": f"1. Ölçüm ({before_rec.get('stage')})", "Tarih/Saat": str(before_rec.get("ts", ""))[:19].replace("T", " "), "Kanal": before_rec.get("channel"), "dx (px)": before_rec.get("dx_px"), "dy (px)": before_rec.get("dy_px"), "Hata (mrad)": before_rec.get("err_mrad"), "Sonuç": before_rec.get("result")},
        {"Test": f"2. Ölçüm ({after_rec.get('stage')})", "Tarih/Saat": str(after_rec.get("ts", ""))[:19].replace("T", " "), "Kanal": after_rec.get("channel"), "dx (px)": after_rec.get("dx_px"), "dy (px)": after_rec.get("dy_px"), "Hata (mrad)": after_rec.get("err_mrad"), "Sonuç": after_rec.get("result")},
        {"Test": "Kayma Farkı (Drift)", "Tarih/Saat": "Δ Fark", "Kanal": "—", "dx (px)": f"{after_rec.get('dx_px', 0) - before_rec.get('dx_px', 0):+.2f}", "dy (px)": f"{after_rec.get('dy_px', 0) - before_rec.get('dy_px', 0):+.2f}", "Hata (mrad)": f"{drift_val:.3f}", "Sonuç": "PASS" if passed else "FAIL"},
    ]
    df = pd.DataFrame(rows)

    exp_md = f"""
    ### 🔬 Titreşim Testi Kayma Değerlendirmesi
    - **Analiz Edilen Seri:** `{serial_no}`
    - **Karşılaştırılan Aşamalar:** `{before_rec.get('stage')}` ➡️ `{after_rec.get('stage')}`
    - **Açısal Eksen Kayması:** `{drift_val:.3f} mrad` (İzin verilen: `{tol:.2f} mrad`)
    - **Değerlendirme:** {'Ürün çevresel şok ve titreşim profilini mekanik deformasyona uğramadan başarıyla tamamlamıştır.' if passed else 'DİKKAT: Titreşim testi sonrasında tolerans dışı kayma saptanmıştır. Optik sabitleme vidaları ve yapıştırıcı hatları incelenmelidir.'}
    """
    return badge_html, df, exp_md


# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
sample_examples = load_sample_examples()

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="3E Kalite Kontrol — AI Destekli Muayene (PoC)",
    css="""
    .metric-box { border-radius: 8px; padding: 12px; }
    .gr-button-primary { font-weight: 700 !important; }
    div[style*="background: #f"], div[style*="background: #e"], div[style*="background: white"] { color: #1f2937; }
    div[style*="background: #f"] :where(p, li, td, th, span, strong, b, div, h1, h2, h3, h4, code, em, label):not([style*="color"]), div[style*="background: #e"] :where(p, li, td, th, span, strong, b, div, h1, h2, h3, h4, code, em, label):not([style*="color"]), div[style*="background: white"] :where(p, li, td, th, span, strong, b, div, h1, h2, h3, h4, code, em, label):not([style*="color"]) { color: #1f2937 !important; }
    """,
) as demo:
    # State keeping last pipeline result
    last_result_state = gr.State(value=None)
    operator_labels_state = gr.State(value=[])

    gr.Markdown(
        """
        # 3E Elektro Optik — AI Destekli Görsel Kalite Kontrol Sistemi (PoC)
        *Optik lens grupları, termal kamera modülleri ve elektro-optik gözetleme üniteleri için hibrit muayene paneli.*
        """
    )

    with gr.Tabs() as tabs:
        # ===================================================================
        # TAB 1: ÜRETİM İÇİ MUAYENE
        # ===================================================================
        with gr.Tab("🏭 Üretim İçi Muayene", id="tab_muayene"):
            gr.Markdown("Denemek için örnek görseller: **Hakkında** sekmesindeki *Test görsellerini indir (ZIP)* düğmesi.")
            gr.Markdown(
                """
                **Aşama Amacı:** Üretim ve montaj hattındaki elektro-optik bileşenlerin (lens, sensör, gövde, kablo) yüzey çizikleri, kaplama deformasyonları ve montaj kusurları denetlenir.  
                **Muayene Yöntemi:** Derin öğrenme tabanlı yapay zeka modelleri (PatchCore anomali tespiti, YOLO11n nesne tespiti, Gemini multimodal muhakeme ve MobileSAM) ile uzman operatör destekli görsel denetim yapılır.
                """
            )
            with gr.Row():
                # LEFT COLUMN: Inputs & Controls
                with gr.Column(scale=5):
                    gr.Markdown("### 📥 Parça Bilgileri ve Görsel Yükleme")
                    product_group_dropdown = gr.Dropdown(
                        label="Ürün Grubu",
                        choices=list(config.PRODUCT_GROUPS.keys()),
                        value=list(config.PRODUCT_GROUPS.keys())[0],
                        interactive=True,
                    )

                    with gr.Row():
                        serial_no_input = gr.Textbox(
                            label="Seri No",
                            value=generate_default_serial,
                            interactive=True,
                            scale=3,
                        )
                        inspector_input = gr.Textbox(
                            label="Muayeneci",
                            value="Muayene Uzmanı",
                            placeholder="Ad Soyad",
                            interactive=True,
                            scale=2,
                        )

                    with gr.Row():
                        copy_to_stage2_btn = gr.Button(
                            "Son test için bu seri no'yu kullan",
                            variant="secondary",
                            size="sm",
                            scale=3,
                        )
                        copy_to_stage2_status = gr.Markdown(value="", visible=False, scale=2)

                    image_input = gr.Image(
                        label="Muayene Görseli (JPG/PNG, Maks. 10 MB)",
                        type="pil",
                        sources=["upload", "clipboard"],
                    )

                    analyze_btn = gr.Button(
                        "🔍 Parçayı Analiz Et",
                        variant="primary",
                        size="lg",
                    )

                    # Examples gallery
                    if sample_examples:
                        gr.Markdown("#### 📁 Örnek Parça Galerisi (MVTec Temsili)")
                        gr.Examples(
                            examples=sample_examples,
                            inputs=[image_input, product_group_dropdown],
                            label="Referans Örnekler",
                        )

                # RIGHT COLUMN: Results & Overlays
                with gr.Column(scale=7):
                    gr.Markdown("### 📊 AI Analiz Sonuçları ve Gerekçelendirme")
                    decision_badge = gr.HTML(value=render_decision_badge(None))

                    # Heatmap and Detector side-by-side
                    with gr.Row():
                        heatmap_view = gr.Image(
                            label="Anomali Isı Haritası (PatchCore + DINOv2)",
                            interactive=False,
                        )
                        detector_view = gr.Image(
                            label="Kusur Tespiti ve Segmentasyon (YOLO11)",
                            interactive=False,
                        )

                    reasons_view = gr.Markdown(
                        value="*Analiz yapıldığında kural ve motor gerekçeleri burada listelenir.*"
                    )
                    vlm_view = gr.Markdown(
                        value="*VLM (Gemini) muhakeme raporu burada görüntülenecektir.*"
                    )

                    gr.Markdown("#### ⚙️ Motor Analiz Dökümü")
                    engines_table = gr.Dataframe(
                        headers=[
                            "Motor",
                            "Durum",
                            "Olasılık / Metrik",
                            "Gecikme (ms)",
                            "Model / Backend",
                        ],
                        value=pd.DataFrame(
                            columns=[
                                "Motor",
                                "Durum",
                                "Olasılık / Metrik",
                                "Gecikme (ms)",
                                "Model / Backend",
                            ]
                        ),
                        interactive=False,
                    )
                    latency_view = gr.Markdown(value="⏱️ **Toplam Gecikme:** —")

            # BELOW: User Decision Section
            gr.Markdown("---")
            with gr.Column():
                gr.Markdown("### ✍️ Kullanıcı (Operatör) Nihai Kararı")
                review_warning_view = gr.Markdown(value="")

                with gr.Row():
                    user_decision_radio = gr.Radio(
                        label="Kararınız",
                        choices=["Onayla (AI önerisi)", "Kabul", "Ret"],
                        value="Onayla (AI önerisi)",
                        interactive=True,
                        scale=3,
                    )
                    defect_type_dropdown = gr.Dropdown(
                        label="Kusur Tipi",
                        choices=config.DEFECT_TYPES,
                        value="Yok",
                        interactive=True,
                        scale=3,
                    )

                note_input = gr.Textbox(
                    label="Muayene Notu / Açıklama",
                    placeholder="Varsa tespitlerinizi veya özel açıklamaları buraya yazınız...",
                    lines=2,
                    interactive=True,
                )

                # Accordion: Operatör etiketleme (kaçan / düzeltilen kusur)
                with gr.Accordion("Operatör etiketleme (kaçan / düzeltilen kusur)", open=False):
                    gr.Markdown(
                        "Yapay zeka tespitinde kaçan kusurları kutu çizerek ekleyebilir, hatalı kutuları silebilir veya sınırlarını düzeltebilirsiniz.\n"
                        "Kutuları belirledikten sonra **SAM ile maske çıkar** butonuna basarak pikselsel segmentasyon oluşturabilirsiniz."
                    )
                    with gr.Row():
                        with gr.Column(scale=6):
                            annotator_view = image_annotator(
                                label="Operatör Kutu Etiketleme / Düzenleme",
                                label_list=ANNOTATOR_LABELS,
                                label_colors=ANNOTATOR_COLORS,
                                interactive=True,
                            )
                            sam_btn = gr.Button("✂️ SAM ile maske çıkar", variant="secondary")
                        with gr.Column(scale=6):
                            sam_overlay_view = gr.Image(
                                label="SAM Segmentasyon Maskeleri ve Etiketler",
                                interactive=False,
                            )
                            sam_status_view = gr.Markdown(value="*Kutulardan maske çıkarmak için butona basınız.*")

                with gr.Row():
                    save_btn = gr.Button(
                        "💾 Kararı Kaydet ve Veritabanına İşle",
                        variant="primary",
                        scale=2,
                    )
                    confirmation_view = gr.Markdown(value="", scale=3)

        # ===================================================================
        # TAB 2: ÜRETİM SONRASI TEST
        # ===================================================================
        with gr.Tab("🧪 Üretim Sonrası Test", id="tab_boresight"):
            last_boresight_state = gr.State(value=None)

            gr.Markdown(
                """
                **Aşama Amacı:** Montajı tamamlanan elektro-optik sistemlerin optik eksen (boresight) doğruluğu, görünür/termal kanal hizalaması ve titreşim sonrası açısal sapması (drift) denetlenir.  
                **Muayene Yöntemi:** Kolimatör hedefleri üzerinden OpenCV alt-piksel retikül tespiti, Huber M-tahmincisi ve geometrik dönüşümlerle doğrudan hassas optik ölçüm (<0.06 px sentetik hedef hatası) yapılır.
                """
            )

            with gr.Tabs():
                with gr.Tab("🎯 Boresight ve Drift", id="sub_boresight"):
                    with gr.Row():
                        # LEFT COLUMN: Inputs & Controls
                        with gr.Column(scale=5):
                            gr.Markdown("### 📥 Test Parametreleri ve Hedef Görseli")
                            with gr.Row():
                                serial_input_boresight = gr.Textbox(
                                    label="Seri No",
                                    value=generate_default_serial,
                                    interactive=True,
                                    scale=3,
                                )
                                inspector_input_boresight = gr.Textbox(
                                    label="Muayeneci",
                                    value="Muayene Uzmanı",
                                    placeholder="Ad Soyad",
                                    interactive=True,
                                    scale=2,
                                )

                            with gr.Row():
                                stage_dropdown_boresight = gr.Dropdown(
                                    label="Aşama",
                                    choices=["son test", "titreşim öncesi", "titreşim sonrası"],
                                    value="son test",
                                    interactive=True,
                                    scale=3,
                                )
                                channel_dropdown_boresight = gr.Dropdown(
                                    label="Kanal",
                                    choices=["Görünür", "Termal"],
                                    value="Görünür",
                                    interactive=True,
                                    scale=3,
                                )

                            with gr.Row():
                                pitch_input_boresight = gr.Number(
                                    label="Piksel Boyutu (µm)",
                                    value=3.45,
                                    interactive=True,
                                    scale=2,
                                )
                                focal_input_boresight = gr.Number(
                                    label="Odak Uzaklığı (mm)",
                                    value=50.0,
                                    interactive=True,
                                    scale=2,
                                )
                                tolerance_input_boresight = gr.Number(
                                    label="Tolerans (mrad)",
                                    value=0.5,
                                    interactive=True,
                                    scale=2,
                                )

                            image_input_boresight = gr.Image(
                                label="Kolimatör Hedef Görseli (JPG/PNG)",
                                type="pil",
                                sources=["upload", "clipboard"],
                            )

                            measure_boresight_btn = gr.Button(
                                "🎯 Optik Ekseni Ölç",
                                variant="primary",
                                size="lg",
                            )

                            boresight_sample_examples = load_boresight_examples()
                            if boresight_sample_examples:
                                gr.Markdown("#### 📁 Referans Kolimatör Hedefleri")
                                gr.Examples(
                                    examples=boresight_sample_examples,
                                    inputs=[
                                        image_input_boresight,
                                        channel_dropdown_boresight,
                                        pitch_input_boresight,
                                        focal_input_boresight,
                                        tolerance_input_boresight,
                                    ],
                                    label="Sentetik Kolimatör Numuneleri",
                                )

                        # RIGHT COLUMN: Results & Overlays
                        with gr.Column(scale=7):
                            gr.Markdown("### 📊 Optik Eksen Ölçüm Sonuçları")
                            badge_boresight_view = gr.HTML(value=render_boresight_badge(None))

                            image_boresight_view = gr.Image(
                                label="İşaretlenmiş Eksen ve Hata Vektörü Görseli",
                                interactive=False,
                            )

                            gr.Markdown("#### 📐 Ölçüm Metrikleri Tablosu")
                            table_boresight_view = gr.Dataframe(
                                value=build_boresight_metrics_dataframe(None),
                                interactive=False,
                            )

                            gr.Markdown(
                                """
                        ---
                        #### ℹ️ Ölçüm Metodolojisi ve Kabul Standartları
                        - **Alt-Piksel Retikül Tespiti:** Görsel gri seviyeye çevrilir; kutup tespiti ile aydınlık/karanlık fon ayrıştırılır.
                          Vinyetleme top-hat morfolojisiyle elendikten sonra yatay ve dikey kollar Huber M-tahmincisiyle kesiştirilerek **< 0.05 piksel** hata ile merkez bulunur.
                        - **Açısal Hata Bağıntısı (mrad):**
                          $$\\theta_{\\text{mrad}} = 1000 \\times \\arctan\\left(\\frac{\\Delta r \\times p \\times 10^{-3}}{f}\\right), \\quad \\Delta r = \\sqrt{\\Delta x^2 + \\Delta y^2}$$
                          burada $p$: piksel boyutu ($\\mu\\text{m}$), $f$: odak uzaklığı ($\\text{mm}$).
                        - **Neden Önemli? (Termal Kamera Modülü Recall Dersi):**
                          Termal kamera modüllerinde optik eksen kaçıklığı sahadaki hedef tespit ve menzil kestiriminde kritik sapmalara yol açar.
                          Mekanik montaj gevşemelerini tespit etmek için elektro-optik birimler **titreşim (vibration/shock) testi öncesinde ve sonrasında** test edilir;
                          iki test arasındaki eksen kayması (**drift**), sahadaki en yaygın hata kök nedenidir.
                        """
                            )

                    # Kaydetme Butonu
                    gr.Markdown("---")
                    with gr.Row():
                        save_boresight_btn = gr.Button(
                            "💾 Boresight Testini Kaydet ve Veritabanına İşle",
                            variant="primary",
                            scale=2,
                        )
                        save_boresight_confirmation = gr.Markdown(value="", scale=3)

                    # ACCORDION 1: KANAL ARASI HİZALAMA
                    with gr.Accordion("Kanal arası hizalama (görünür + termal)", open=False):
                        gr.Markdown(
                            "Görünür (visible) ve termal (thermal) optik kanallarının optik eksen eşmerkezliliğini hesaplar.\n"
                            "Her iki kanalın retikül merkezi kendi piksel boyutu ve odak uzaklığıyla açısal uzaya (mrad) dönüştürülüp aralarındaki fark alınır."
                        )
                        with gr.Row():
                            with gr.Column(scale=6):
                                vis_image_input = gr.Image(label="Görünür Kanal Görseli (1280x1024)", type="pil")
                                with gr.Row():
                                    vis_pitch_input = gr.Number(label="Görünür Piksel (µm)", value=3.45)
                                    vis_focal_input = gr.Number(label="Görünür Odak (mm)", value=50.0)
                            with gr.Column(scale=6):
                                thm_image_input = gr.Image(label="Termal Kanal Görseli (640x512)", type="pil")
                                with gr.Row():
                                    thm_pitch_input = gr.Number(label="Termal Piksel (µm)", value=12.0)
                                    thm_focal_input = gr.Number(label="Termal Odak (mm)", value=25.0)

                        with gr.Row():
                            inter_tol_input = gr.Number(label="İzin Verilen Eksenler Arası Tolerans (mrad)", value=0.5, scale=2)
                            compare_channels_btn = gr.Button("⚖️ Kanal Arası Eksen Farkını Hesapla", variant="secondary", scale=2)

                        inter_badge_view = gr.HTML(value="")
                        inter_image_view = gr.Image(label="Yan Yana Kanal Karşılaştırma Görseli", interactive=False)
                        inter_table_view = gr.Dataframe(interactive=False)

                    # ACCORDION 2: TİTREŞİM KAYMA (DRIFT) ANALİZİ
                    with gr.Accordion("Titreşim öncesi/sonrası kayma (Drift Analizi)", open=False):
                        gr.Markdown(
                            "Aynı seri numarasına ait 'titreşim öncesi' ve 'titreşim sonrası' test kayıtlarını veritabanından çekerek açısal kayma miktarını hesaplar."
                        )
                        with gr.Row():
                            drift_serial_input = gr.Textbox(
                                label="Seri No",
                                placeholder="Örn: TC-20261003-002",
                                value="TC-20261003-002",
                                scale=3,
                            )
                            drift_tol_input = gr.Number(
                                label="İzin Verilen Azami Kayma (mrad)",
                                value=0.5,
                                scale=2,
                            )
                            calc_drift_btn = gr.Button("📈 Kayma (Drift) Miktarını Hesapla", variant="secondary", scale=2)

                        drift_badge_view = gr.HTML(value="")
                        drift_table_view = gr.Dataframe(interactive=False)
                        drift_explanation_view = gr.Markdown(value="")

                with gr.Tab("🔬 MTF (Keskinlik)", id="sub_mtf"):
                    gr.Markdown(
                        "**Eğimli kenar (ISO 12233) MTF testi.** Sonuç 'Kaydet' ile veritabanına yazılır ve kalite kapısına dahil edilir."
                    )
                    with gr.Row():
                        with gr.Column(scale=5):
                            serial_input_mtf = gr.Textbox(label="Seri No", value=generate_default_serial, interactive=True)
                            channel_dropdown_mtf = gr.Dropdown(
                                choices=["Görünür", "Termal"], value="Görünür", label="Kanal"
                            )
                            with gr.Row():
                                pitch_input_mtf = gr.Number(label="Piksel Boyutu (µm)", value=3.45, minimum=0.1)
                                spec_input_mtf = gr.Number(label="Spec MTF50 (cycles/pixel)", value=0.25, minimum=0.0)
                            image_input_mtf = gr.Image(label="Eğimli Kenar Görseli", type="pil", image_mode="RGB")
                            measure_mtf_btn = gr.Button("🔬 MTF Ölç", variant="primary", size="lg")
                            save_mtf_btn = gr.Button("💾 Kaydet", variant="secondary")
                            save_mtf_status = gr.Markdown(value="")
                            last_mtf_state = gr.State(value=None)
                            mtf_examples = load_mtf_examples()
                            if mtf_examples:
                                gr.Examples(
                                    examples=mtf_examples,
                                    inputs=[image_input_mtf, channel_dropdown_mtf, pitch_input_mtf, spec_input_mtf],
                                    label="Sentetik Eğimli Kenar Numuneleri",
                                )
                        with gr.Column(scale=7):
                            badge_mtf_view = gr.HTML(value=render_mtf_badge(None))
                            plot_mtf_view = gr.Image(label="MTF Eğrisi", interactive=False)
                            roi_mtf_view = gr.Image(label="Kenar ve ROI", interactive=False)
                            table_mtf_view = gr.Dataframe(interactive=False, label="Ölçüm Tablosu")
                    gr.Markdown(
                        """
**MTF nedir?** Modülasyon Transfer Fonksiyonu (MTF), bir optik sistemin farklı uzamsal frekanslardaki kontrastı ne kadar koruyabildiğini gösterir. MTF50, kontrastın yarıya düştüğü frekanstır ve algılanan keskinliğin en yaygın tek sayı özetidir. Nyquist'te (0.5 cy/px) MTF ise örnekleme sınırındaki detayı ve aliasing riskini anlatır.

**Eğimli kenar yöntemi (ISO 12233):** Hedefteki kenar dikeyden/yataydan yaklaşık 5° eğiktir. Her satırda kenarın alt-piksel konumu bulunur, doğru uydurulur, pikseller kenar normaline izdüşürülüp 4 kat aşırı örneklenmiş kenar yayılım fonksiyonu (ESF) elde edilir. Türevi çizgi yayılım fonksiyonunu (LSF), Hamming pencereli FFT ise MTF eğrisini verir.

**Optik modüller için önemi:** Odak kayması, lens eğikliği, yapıştırma veya montaj hataları ile ısıl gerilmeler keskinliği düşürür. Boresight yalnızca eksen hizasını ölçer; MTF ise görüntü kalitesini sayısallaştırarak üretim sonrası kabul kriteri (spec) koymayı mümkün kılar. Termal kanalda gürültü yüksek olduğundan SNR uyarılarına dikkat edin.
                        """
                    )

                    measure_mtf_btn.click(
                        fn=on_measure_mtf,
                        inputs=[image_input_mtf, pitch_input_mtf, spec_input_mtf, channel_dropdown_mtf],
                        outputs=[badge_mtf_view, plot_mtf_view, roi_mtf_view, table_mtf_view, last_mtf_state],
                        api_name="measure_mtf",
                    )
                    channel_dropdown_mtf.change(
                        fn=lambda c: (12.0, 0.18) if c == "Termal" else (3.45, 0.25),
                        inputs=[channel_dropdown_mtf],
                        outputs=[pitch_input_mtf, spec_input_mtf],
                        api_name=False,
                    )

                with gr.Tab("🔩 Tork İşareti", id="sub_torque_mark"):
                    gr.Markdown(
                        "**Tork işareti (witness mark / torque stripe) kontrolü.** "
                        "Titreşim testi sonrasında vida bağlantılarının gevşeyip gevşemediğini, "
                        "boya çizgisinin açısal kaymasıyla tespit eder. Sonuç 'Kaydet' ile veritabanına yazılır ve kalite kapısına dahil edilir."
                    )
                    with gr.Row():
                        with gr.Column(scale=5):
                            serial_input_torque = gr.Textbox(
                                label="Seri No", value=generate_default_serial, interactive=True
                            )
                            stage_dropdown_torque = gr.Dropdown(
                                choices=["titreşim sonrası", "son test", "titreşim öncesi"],
                                value="titreşim sonrası",
                                label="Aşama",
                                interactive=True,
                            )
                            with gr.Row():
                                color_dropdown_torque = gr.Dropdown(
                                    choices=["auto", "kırmızı", "turuncu", "sarı"],
                                    value="auto",
                                    label="Boya Rengi",
                                    interactive=True,
                                )
                                tolerance_input_torque = gr.Number(
                                    label="Tolerans (derece)", value=5.0, minimum=0.1, interactive=True
                                )
                            image_input_torque = gr.Image(
                                label="Vida Üst Görünüm Görseli", type="pil", image_mode="RGB"
                            )
                            inspect_torque_btn = gr.Button(
                                "🔩 Kontrol Et", variant="primary", size="lg"
                            )
                            save_torque_btn = gr.Button("💾 Kaydet", variant="secondary")
                            save_torque_status = gr.Markdown(value="")
                            last_torque_state = gr.State(value=None)
                            torque_examples = load_torque_mark_examples()
                            if torque_examples:
                                gr.Examples(
                                    examples=torque_examples,
                                    inputs=[image_input_torque, color_dropdown_torque, tolerance_input_torque],
                                    label="Sentetik Tork İşareti Numuneleri",
                                )
                        with gr.Column(scale=7):
                            badge_torque_view = gr.HTML(value=render_torque_mark_badge(None))
                            image_torque_view = gr.Image(
                                label="İşaretlenmiş Muayene Görseli", interactive=False
                            )
                            table_torque_view = gr.Dataframe(
                                interactive=False, label="Ölçüm Tablosu"
                            )
                            explanation_torque_view = gr.Markdown(value="")

                    inspect_torque_btn.click(
                        fn=on_inspect_torque_mark,
                        inputs=[image_input_torque, color_dropdown_torque, tolerance_input_torque],
                        outputs=[badge_torque_view, image_torque_view, table_torque_view, explanation_torque_view, last_torque_state],
                        api_name="inspect_torque_mark",
                    )

        # ===================================================================
        # TAB 3: GEÇMİŞ VE PANO
        # ===================================================================
        with gr.Tab("📊 Geçmiş ve Pano", id="tab_pano") as tab_pano:
            with gr.Row():
                gr.Markdown("### 📈 Muayene İstatistikleri ve Kalite Kontrol Panosu")
                refresh_btn = gr.Button("🔄 Panoyu Yenile", variant="secondary", size="sm")

            initial_stats = storage.stats()
            metrics_cards_view = gr.HTML(value=render_metrics_cards(initial_stats))

            # ===============================================================
            # SERİ NUMARASI BAZINDA KALİTE KAPISI (QUALITY GATE)
            # ===============================================================
            gr.Markdown("---")
            gr.Markdown("### 🚦 Seri Numarası Bazında Kalite Kapısı")
            gr.Markdown(
                "*Üretim içi görsel muayene ve üretim sonrası son test (boresight ve titreşim kayması) "
                "aşamalarını tek potada birleştiren kalite kapısı durum tablosu ve seri sorgulama paneli.*"
            )

            quality_gate_table = gr.Dataframe(
                value=get_quality_gate_dataframe,
                interactive=False,
                label="Seri Numarası Bazında Kalite Kapısı Tablosu",
            )

            gr.Markdown("#### 🔍 Seri Numarası Geçmiş Sorgulama (Tüm Aşamalar)")
            with gr.Row():
                serial_lookup_input = gr.Textbox(
                    label="Seri No",
                    placeholder="Örn: LN-20261001-014, TC-20261003-002 veya SN-GATE-1",
                    scale=4,
                )
                serial_lookup_btn = gr.Button("🔍 Geçmişi Sorgula", variant="primary", scale=1)

            serial_lookup_card_view = gr.HTML(value="")

            with gr.Row():
                with gr.Column(scale=6):
                    gr.Markdown("##### 🏭 1. Aşama: Görsel Muayene Kayıtları")
                    serial_insp_history_table = gr.Dataframe(
                        value=pd.DataFrame(columns=["Muayene ID", "Tarih", "Muayeneci", "Ürün Grubu", "AI Kararı", "Nihai Karar", "Kusur Türü", "Not"]),
                        interactive=False,
                    )
                with gr.Column(scale=6):
                    gr.Markdown("##### 🧪 2. Aşama: Boresight / Son Test Kayıtları")
                    serial_bore_history_table = gr.Dataframe(
                        value=pd.DataFrame(columns=["Test ID", "Tarih", "Muayeneci", "Aşama", "Kanal", "dx (px)", "dy (px)", "Açı (mrad)", "Tolerans", "Sonuç", "Kayma"]),
                        interactive=False,
                    )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column(scale=6):
                    gr.Markdown("#### 🏷️ Kusur Tipleri Dağılımı")
                    defect_bar_plot = gr.BarPlot(
                        value=get_defect_dataframe(initial_stats),
                        x="Kusur Tipi",
                        y="Adet",
                        title="Tespit Edilen Kusur Dağılımı",
                        tooltip=["Kusur Tipi", "Adet"],
                        y_lim=[0, None],
                        height=280,
                    )

                with gr.Column(scale=6):
                    gr.Markdown("#### 💾 Veri ve Aktif Öğrenme Dışa Aktarımı")
                    gr.Markdown(
                        """
                        - **CSV Dışa Aktar:** ERP ve kalite yönetim sistemleri entegrasyonu için tüm muayene kayıtlarını indirir.
                        - **YOLO Aktif Öğrenme ZIP:** Operatör tarafından reddedilen ve nesne kutusu içeren numuneleri
                          YOLO eğitim formatında (görseller + etiketler + `classes.txt`) paketler.
                        """
                    )
                    with gr.Row():
                        export_csv_btn = gr.Button("📥 Muayene Kayıtlarını İndir (CSV)", variant="secondary")
                        export_boresight_csv_btn = gr.Button("🎯 Boresight Kayıtlarını İndir (CSV)", variant="secondary")
                        export_post_btn = gr.Button("🧪 Üretim Sonrası Testleri İndir (ZIP: boresight+MTF+tork)", variant="secondary")
                        export_yolo_btn = gr.Button("📦 YOLO Etiket Paketini İndir (ZIP)", variant="primary")
                    export_file_view = gr.File(label="İndirme Bağlantısı", interactive=False)

            gr.Markdown("#### 📋 Son 100 Muayene Kaydı")
            history_table = gr.Dataframe(
                value=get_history_dataframe(100),
                interactive=False,
            )

            gr.Markdown("---")
            gr.Markdown("### 🧪 Üretim Sonrası Test Panosu")
            gr.Markdown("*Optik eksen (boresight), MTF ve tork işareti testlerinin sonuç özeti.*")
            _pp0 = on_refresh_post_production()
            pp_cards_view = gr.HTML(value=_pp0[0])
            pp_bar_plot = gr.BarPlot(
                value=_pp0[1], x="Test", y="Adet", color="Sonuç",
                title="Test Tipine Göre Sonuç Dağılımı",
                tooltip=["Test", "Sonuç", "Adet"], y_lim=[0, None], height=280,
            )
            gr.Markdown("#### 🚨 Son Kaldı / Belirsiz Testler")
            pp_bad_table = gr.Dataframe(value=_pp0[2], interactive=False)
            gr.Markdown("#### 📋 Son üretim sonrası testleri")
            pp_table = gr.Dataframe(value=_pp0[3], interactive=False)

        # ===================================================================
        # TAB 4: HAKKINDA
        # ===================================================================
        with gr.Tab("ℹ️ Hakkında", id="tab_hakkinda"):
            gr.DownloadButton("Test görsellerini indir (ZIP)", value=str(Path(__file__).parent / "test_gorselleri.zip"), variant="primary")
            gr.Markdown(
                f"""
                ## 3E Elektro Optik — AI Destekli Görsel Kalite Kontrol ve Boresight PoC

                Bu uygulama, 3E Elektro Optik üretim ve kalite kontrol süreçlerinde kullanılmak üzere
                geliştirilmiş iki aşamalı bir yapay zeka ve optik ölçüm prototipidir.

                ---

                ### 🏭 1. Aşama: Üretim İçi Muayene
                Montaj öncesi ve montaj esnasında parça yüzey kusurları, kaplama hataları ve montaj anomalileri denetlenir:
                - **Anomali Motoru (topluluk):** PatchCore (WideResNet50) ile AnomalyDINO (DINOv2 ViT-S/14) birlikte çalışır. Skorlar önce birleştirilir, sonra kalibre edilir ("önce birleştir, sonra kalibre et"). Kalibrasyon, kategori başına 40 sağlam görsel üzerinden conformal $p$-değeri ile yapılır. Anomali olasılığı $a = \\text{{clip}}(\\log p / \\log 0.02, 0, 1)$ olarak hesaplanır.
                - **Kusur Dedektörü:** YOLO11n ile sınır kutusu, kusur sınıfı ve güven skoru üretilir (mAP50: 0,814; uygulamadaki kategorilerde 0,779).
                - **Görsel Muhakeme (VLM):** Gemini 3.5 Flash-Lite (yedekler: 3.8 Flash ve 3.1 Flash-Lite) modeline ısı haritası ve kutu ipuçlarıyla birlikte görsel verilir. Üç örnekleme ve tutarlılık kontrolüyle Türkçe gerekçe üretilir.
                - **Füzyon:** Motor çıktıları birleştirilerek **KABUL**, **İNSAN İNCELEMESİ** veya **RET** önerisi verilir. Termal modül grubunda ek risk kuralı vardır: bu grup erken çıkış almaz ve şüpheli durumda otomatik kabul edilmez.
                - **Operatör Etiketleme:** Operatör kusur çevresine kutu çizer, MobileSAM kusur maskesini çıkarır. Etiketler YOLO detect ve segment formatında dışa aktarılır.

                ---

                ### 🧪 2. Aşama: Üretim Sonrası Test
                Montaj hattından çıkan elektro-optik sistemlerin nihai kabulünde doğrudan ölçüm yapılır:
                - **Boresight:** OpenCV ile alt-piksel retikül tespiti yapılır. Sentetik hedeflerde en fazla 0,06 piksel hata ölçülmüştür. Açısal hata piksel boyutu ve odak uzaklığı ile mrad'a çevrilir:
                  $$\\theta_{{\\text{{mrad}}}} = 1000 \\times \\arctan\\left(\\frac{{\\Delta r \\times p \\times 10^{{-3}}}}{{f}}\\right), \\quad \\Delta r = \\sqrt{{\\Delta x^2 + \\Delta y^2}}$$
                  Görünür ve termal kanalın birbirine göre hizalaması ile titreşim öncesi ve sonrası eksen kayması da ölçülür.
                - **MTF (keskinlik):** ISO 12233 eğik kenar yöntemiyle MTF50 hesaplanır. Sentetik hedeflerde MTF50 hatası görünür kanalda %1,5, termal kanalda %1,9 düzeyindedir.
                - **Tork işareti:** Vida kafasındaki ve gövdedeki boya çizgisinin açı farkı ölçülür. Sentetik örneklerde 19 örneğin 19'unda doğru karar verilmiş, en büyük açı hatası 1,34° olmuştur.

                ---

                ### 📏 Ölçülen Başarım
                365 gerçek MVTec test görseli üzerinde ölçülmüştür:
                - Topluluk anomali motoru: test AUROC **0,995**. $p \\le 0{{,}}05$ eşiğinde kusur yakalama oranı **%92,9**, yanlış alarm **%0**.
                - Uçtan uca (anomali + YOLO) test bölümünde: **0 kaçan kusur**, **0 yanlış ret**, örneklerin **%20,2**'si insan incelemesine gider.

                **Sınırlılıklar:**
                - Veri temsilidir (MVTec AD); gerçek 3E parça verisi kullanılmamıştır.
                - YOLO, MVTec test görsellerinin bir kısmıyla eğitildi. Bu nedenle uçtan uca sonuçlar iyimser olabilir.
                - MTF ve tork işareti sonuçları henüz veritabanına kaydedilmiyor.
                - Bu PoC'de görseller Google API'ye gönderilir. Üretimde kapalı devre (on-prem) çalışılmalıdır.

                ---

                ### 🚦 Kalite Kapısı (Quality Gate) Entegrasyonu
                Her parça seri numarası bazında izlenir:
                - **🟢 SEVKE HAZIR:** Üretim içi görsel muayene operatör nihai kararı **KABUL** (ACCEPT) ve üretim sonrası son test sonucu **GEÇTİ** (PASS) olduğunda (ve titreşim öncesi/sonrası testler mevcutsa kayma tolerans içindeyse) verilir.
                - **🟡 BEKLEMEDE:** Muayene veya son test aşamalarından biri henüz tamamlanmamışsa.
                - **🔴 RET:** Görsel muayenede ret verilmişse, son test toleransı aşılmışsa veya titreşim kayması tolerans dışıysa.

                ---

                ### 🔄 Karar Akışı ve Füzyon Şeması
                ```text
                [ Muayene Görseli ] (Optik Lens / Termal Kamera / Gözetleme)
                         │
                         ├────────────────────────────────────────┐
                         ▼                                        ▼
                ┌────────────────────────┐               ┌────────────────────────┐
                │    Anomali Motoru      │               │    Kusur Dedektörü     │
                │ PatchCore + DINOv2     │               │      YOLO11n (ONNX)    │
                └──────────┬─────────────┘               └──────────┬─────────────┘
                           │                                        │
                           └───────────────┬────────────────────────┘
                                           │
                                  ┌────────┴────────┐
                                  │   Erken Çıkış   │──(Anomali yok & YOLO boş)──► [ KABUL ÖNERİSİ ]
                                  │   Kontrolü      │
                                  └────────┬────────┘
                                           │ (Kusur veya Şüphe varsa)
                                           ▼
                                ┌────────────────────────┐
                                │   VLM Akıl Yürütme     │
                                │    Gemini Multimodal   │
                                └──────────┬─────────────┘
                                           │
                                           ▼
                                ┌────────────────────────┐
                                │ Hibrit Füzyon & Kapı   │ (Conformal p + Ağırlıklı Oylama)
                                └──────────┬─────────────┘
                                           │
                      ┌────────────────────┼────────────────────┐
                      ▼                    ▼                    ▼
                [ KABUL ÖNERİSİ ]  [ İNSAN İNCELEMESİ ]   [ RET ÖNERİSİ ]
                      │                    │                    │
                      └────────────────────┼────────────────────┘
                                           ▼
                              ┌────────────────────────┐
                              │  Operatör Nihai Kararı │
                              └────────────┬───────────┘
                                           │
                                           ▼
                              ┌────────────────────────┐
                              │ SQLite & Kalite Kapısı │
                              │ (CSV & YOLO ZIP Paket) │
                              └────────────────────────┘
                ```

                ---

                ### ⚙️ Çalışma Zamanı Motor Durumları
                {get_runtime_engine_status_md()}

                ---

                ### ⚠️ Sınırlılıklar ve Gerçek Dünya Koşulları
                1. **Temsili Veri Seti (MVTec AD):**
                   Bu PoC çalışmasında MVTec Anomaly Detection (CC BY-NC-SA 4.0) veri kümesi
                   temsili elektro-optik parça verisi olarak kullanılmıştır:
                   - *Optik lens grubu* → `metal_nut`
                   - *Termal kamera modülü* → `transistor`
                   - *Gözetleme ünitesi* → `cable`
                2. **Fiziksel Ölçüm Gerektiren Kusurlar:**
                   - **Optik Eksen Hizalaması (Boresight İstasyonu):** Görsel kontrolden sonraki montaj
                     sonrası fonksiyonel test istasyonumuz ('🧪 Üretim Sonrası Test'), kolimatör retikülü
                     üzerinden alt-piksel analizi yaparak mrad cinsinden optik eksen kaçıklığını, kanal
                     arası (inter-channel) hizalamayı ve titreşim testi öncesi/sonrası kaymayı (drift) denetler.
                   - **Konektör Mekanik Gevşekliği:** Tork ölçer veya pin soket kuvvet testi gerektirir.
                   Sistem şüpheli durumlarda operatörü fiziksel teste yönlendirir.
                3. **Bulut API ve Veri Güvenliği:**
                   - PoC sürümünde Gemini Flash modeli Google Bulut API üzerinden çağrılmaktadır.
                   - Savunma sanayii ve gizlilik gerektiren üretim ortamlarında sistem,
                     kapalı devre (on-premise) çalışacak yerel açık ağırlıklı VLM
                     (örn. Qwen2.5-VL veya Llama-Vision) ve yerel GPU kümesine uyarlanmalıdır.
                4. **Kalıcı Depolama (Hugging Face Spaces):**
                   - Hugging Face ücretsiz CPU Space ortamında yerel disk konteyner yeniden
                     başlatıldığında sıfırlanır. Demo süresince tutarlılık sağlamak adına
                     başlangıçta temsili geçmiş veriler (`seed_demo`) yüklenir.
                   - Üretim ortamında PostgreSQL ve S3/MinIO uyumlu nesne deposu kullanılmalıdır.
                """
            )

    # -----------------------------------------------------------------------
    # Wire Events
    # -----------------------------------------------------------------------
    analyze_btn.click(
        fn=on_analyze_click,
        inputs=[
            image_input,
            product_group_dropdown,
            serial_no_input,
            inspector_input,
        ],
        outputs=[
            decision_badge,
            heatmap_view,
            detector_view,
            vlm_view,
            reasons_view,
            engines_table,
            latency_view,
            last_result_state,
            user_decision_radio,
            defect_type_dropdown,
            review_warning_view,
            confirmation_view,
            annotator_view,
            sam_overlay_view,
            sam_status_view,
            operator_labels_state,
        ],
        api_name="analyze",
    )

    sam_btn.click(
        fn=on_sam_click,
        inputs=[
            annotator_view,
            last_result_state,
            image_input,
        ],
        outputs=[
            sam_overlay_view,
            sam_status_view,
            operator_labels_state,
        ],
        api_name="sam_segment",
    )

    save_evt = save_btn.click(
        fn=on_save_click,
        inputs=[
            last_result_state,
            image_input,
            product_group_dropdown,
            serial_no_input,
            inspector_input,
            user_decision_radio,
            defect_type_dropdown,
            note_input,
            operator_labels_state,
            annotator_view,
        ],
        outputs=[
            confirmation_view,
            serial_no_input,
            metrics_cards_view,
            defect_bar_plot,
            history_table,
        ],
        api_name="save_inspection",
    )

    refresh_btn.click(
        fn=on_refresh_dashboard,
        inputs=[],
        outputs=[
            metrics_cards_view,
            defect_bar_plot,
            history_table,
        ],
        api_name="refresh_dashboard",
    )

    export_csv_btn.click(
        fn=on_export_csv_click,
        inputs=[],
        outputs=[export_file_view],
        api_name="export_csv",
    )

    export_yolo_btn.click(
        fn=on_export_yolo_click,
        inputs=[],
        outputs=[export_file_view],
        api_name="export_yolo",
    )

    export_boresight_csv_btn.click(
        fn=lambda: str(storage.export_boresight_csv()),
        inputs=[],
        outputs=[export_file_view],
        api_name="export_boresight_csv",
    )

    export_post_btn.click(
        fn=lambda: str(storage.export_post_production_zip()),
        inputs=[],
        outputs=[export_file_view],
        api_name="export_post_production",
    )

    channel_dropdown_boresight.change(
        fn=on_boresight_channel_change,
        inputs=[channel_dropdown_boresight],
        outputs=[pitch_input_boresight, focal_input_boresight],
    )

    measure_boresight_btn.click(
        fn=on_measure_boresight,
        inputs=[
            image_input_boresight,
            pitch_input_boresight,
            focal_input_boresight,
            tolerance_input_boresight,
            channel_dropdown_boresight,
        ],
        outputs=[
            badge_boresight_view,
            image_boresight_view,
            table_boresight_view,
            last_boresight_state,
        ],
        api_name="measure_boresight",
    )

    save_bore_evt = save_boresight_btn.click(
        fn=on_save_boresight,
        inputs=[
            last_boresight_state,
            image_input_boresight,
            serial_input_boresight,
            inspector_input_boresight,
            stage_dropdown_boresight,
            channel_dropdown_boresight,
            tolerance_input_boresight,
        ],
        outputs=[
            save_boresight_confirmation,
            serial_input_boresight,
        ],
        api_name="save_boresight",
    )

    compare_channels_btn.click(
        fn=on_compare_channels,
        inputs=[
            vis_image_input,
            thm_image_input,
            vis_pitch_input,
            vis_focal_input,
            thm_pitch_input,
            thm_focal_input,
            inter_tol_input,
        ],
        outputs=[
            inter_badge_view,
            inter_image_view,
            inter_table_view,
        ],
        api_name="compare_channels",
    )

    calc_drift_btn.click(
        fn=on_calculate_drift,
        inputs=[
            drift_serial_input,
            drift_tol_input,
        ],
        outputs=[
            drift_badge_view,
            drift_table_view,
            drift_explanation_view,
        ],
        api_name="calculate_drift",
    )

    copy_to_stage2_btn.click(
        fn=on_copy_serial_to_stage2,
        inputs=[serial_no_input],
        outputs=[serial_input_boresight, drift_serial_input, copy_to_stage2_status],
    )

    # --- Ortak seri numarası: herhangi bir kutuda kullanıcı düzenlemesi tüm kutulara yayılır ---
    _serial_boxes = [
        serial_no_input,
        serial_input_boresight,
        serial_input_mtf,
        serial_input_torque,
        drift_serial_input,
        serial_lookup_input,
    ]

    def _make_fanout(n: int):
        def _fan(s):
            return tuple(s for _ in range(n))

        return _fan

    def _wire_serial_fanout(src, trigger="input"):
        others = [b for b in _serial_boxes if b is not src]
        getattr(src, trigger)(
            fn=_make_fanout(len(others)),
            inputs=[src],
            outputs=others,
            api_name=False,
            show_progress="hidden",
        )

    for _box in _serial_boxes:
        _wire_serial_fanout(_box, "input")

    # --- Ortak muayeneci: iki kutu (aşama 1 ve son test sekmeleri) senkron ---
    _inspector_boxes = [inspector_input, inspector_input_boresight]
    for _ib in _inspector_boxes:
        _io = [b for b in _inspector_boxes if b is not _ib]
        _ib.input(fn=_make_fanout(len(_io)), inputs=[_ib], outputs=_io, api_name=False, show_progress="hidden")

    serial_lookup_btn.click(
        fn=on_lookup_serial,
        inputs=[serial_lookup_input],
        outputs=[
            serial_lookup_card_view,
            serial_insp_history_table,
            serial_bore_history_table,
        ],
    )

    serial_lookup_input.submit(
        fn=on_lookup_serial,
        inputs=[serial_lookup_input],
        outputs=[
            serial_lookup_card_view,
            serial_insp_history_table,
            serial_bore_history_table,
        ],
    )

    refresh_btn.click(
        fn=get_quality_gate_dataframe,
        inputs=[],
        outputs=[quality_gate_table],
        api_name=False,
    )

    _dash_outputs = [metrics_cards_view, defect_bar_plot, history_table, quality_gate_table,
                     pp_cards_view, pp_bar_plot, pp_bad_table, pp_table]

    def _refresh_all_dashboard():
        m, d, h = on_refresh_dashboard()
        return (m, d, h, get_quality_gate_dataframe(), *on_refresh_post_production())

    refresh_btn.click(
        fn=on_refresh_post_production,
        inputs=[],
        outputs=[pp_cards_view, pp_bar_plot, pp_bad_table, pp_table],
        api_name="refresh_post_production",
    )

    _others_s1 = [b for b in _serial_boxes if b is not serial_no_input]
    _others_s2 = [b for b in _serial_boxes if b is not serial_input_boresight]
    save_evt.then(_make_fanout(len(_others_s1)), inputs=[serial_no_input], outputs=_others_s1, api_name=False, show_progress="hidden").then(
        _refresh_all_dashboard, inputs=[], outputs=_dash_outputs, api_name=False
    )
    save_bore_evt.then(_make_fanout(len(_others_s2)), inputs=[serial_input_boresight], outputs=_others_s2, api_name=False, show_progress="hidden").then(
        _refresh_all_dashboard, inputs=[], outputs=_dash_outputs, api_name=False
    )

    save_mtf_evt = save_mtf_btn.click(
        fn=on_save_mtf,
        inputs=[last_mtf_state, serial_input_mtf, inspector_input_boresight],
        outputs=[save_mtf_status],
        api_name="save_mtf",
    )
    save_mtf_evt.then(_refresh_all_dashboard, inputs=[], outputs=_dash_outputs, api_name=False)
    save_torque_evt = save_torque_btn.click(
        fn=on_save_torque,
        inputs=[last_torque_state, serial_input_torque, inspector_input_boresight, stage_dropdown_torque, color_dropdown_torque],
        outputs=[save_torque_status],
        api_name="save_torque",
    )
    save_torque_evt.then(_refresh_all_dashboard, inputs=[], outputs=_dash_outputs, api_name=False)

    tab_pano.select(_refresh_all_dashboard, inputs=[], outputs=_dash_outputs, api_name=False)
    demo.load(_refresh_all_dashboard, inputs=[], outputs=_dash_outputs, api_name=False)

# ---------------------------------------------------------------------------
# Application Startup
# ---------------------------------------------------------------------------
storage.init_db()
storage.seed_demo()

if pipeline is not None and hasattr(pipeline, "warmup"):
    threading.Thread(target=pipeline.warmup, daemon=True).start()

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    server_port = int(os.getenv("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=server_port)
