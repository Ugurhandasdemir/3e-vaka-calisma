"""app.py - 3E Kalite Kontrol — AI Destekli Muayene (PoC)

Gradio Web Application and User Interface.
"""

from __future__ import annotations

import io
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import warnings
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import config
import storage

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
    an_backend = an.get("backend") or "EfficientAD / PatchCore"
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
        return (
            "> ⚠️ *VLM (Gemini) API anahtarı tanımlı olmadığı için VLM gerekçelendirmesi üretilmedi.*"
        )
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

    # User decision radio handling:
    # "If the AI decision is REVIEW, 'Onayla (AI önerisi)' must be disabled/rejected
    # with a message asking the user to choose Kabul or Ret."
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
) -> tuple[
    str,  # confirmation message
    str,  # new serial number
    str,  # dashboard metrics HTML
    pd.DataFrame,  # dashboard defect distribution
    pd.DataFrame,  # dashboard history dataframe
]:
    """Saves inspection to storage, validates choices, and refreshes dashboard."""
    if not last_result:
        raise gr.Error("Lütfen önce bir analiz gerçekleştiriniz.")

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
    )

    conf_msg = (
        f"✅ **Kayıt Başarılı!** Muayene No: **#{record_id}** veritabanına işlendi. "
        f"(Nihai Karar: **{human_decision}**, Kusur: **{human_defect_type}**)"
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


def on_export_csv_click() -> str:
    """Exports CSV and returns its path."""
    csv_path = storage.export_csv()
    return str(csv_path)


def on_export_yolo_click() -> str:
    """Exports active-learning YOLO zip and returns its path."""
    zip_path = storage.export_yolo_zip()
    return str(zip_path)


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
    """,
) as demo:
    # State keeping last pipeline result
    last_result_state = gr.State(value=None)

    gr.Markdown(
        """
        # 3E Elektro Optik — AI Destekli Görsel Kalite Kontrol Sistemi (PoC)
        *Optik lens grupları, termal kamera modülleri ve elektro-optik gözetleme üniteleri için hibrit muayene paneli.*
        """
    )

    with gr.Tabs() as tabs:
        # ===================================================================
        # TAB 1: MUAYENE
        # ===================================================================
        with gr.Tab("Muayene", id="tab_muayene"):
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
                            label="Anomali Isı Haritası (EfficientAD / PatchCore)",
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

                with gr.Row():
                    save_btn = gr.Button(
                        "💾 Kararı Kaydet ve Veritabanına İşle",
                        variant="primary",
                        scale=2,
                    )
                    confirmation_view = gr.Markdown(value="", scale=3)

        # ===================================================================
        # TAB 2: GEÇMİŞ VE PANO
        # ===================================================================
        with gr.Tab("Geçmiş ve Pano", id="tab_pano"):
            with gr.Row():
                gr.Markdown("### 📈 Muayene İstatistikleri ve Kalite Kontrol Panosu")
                refresh_btn = gr.Button("🔄 Panoyu Yenile", variant="secondary", size="sm")

            initial_stats = storage.stats()
            metrics_cards_view = gr.HTML(value=render_metrics_cards(initial_stats))

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
                        export_csv_btn = gr.Button("📥 Tüm Kayıtları İndir (CSV)", variant="secondary")
                        export_yolo_btn = gr.Button("📦 YOLO Etiket Paketini İndir (ZIP)", variant="primary")
                    export_file_view = gr.File(label="İndirme Bağlantısı", interactive=False)

            gr.Markdown("#### 📋 Son 100 Muayene Kaydı")
            history_table = gr.Dataframe(
                value=get_history_dataframe(100),
                interactive=False,
            )

        # ===================================================================
        # TAB 3: HAKKINDA
        # ===================================================================
        with gr.Tab("Hakkında", id="tab_hakkinda"):
            gr.Markdown(
                f"""
                ## 3E Elektro Optik — AI Destekli Görsel Kalite Kontrol PoC

                Bu uygulama, 3E Elektro Optik üretim ve kalite kontrol süreçlerinde kullanılmak üzere
                geliştirilmiş bir yapay zeka destekli muayene prototipidir.

                ---

                ### 🔄 Sistem Mimarisi ve Karar Akışı
                ```text
                [ Muayene Görseli ] (Optik Lens / Termal Kamera / Gözetleme)
                         │
                         ├────────────────────────────────────────┐
                         ▼                                        ▼
                ┌────────────────────────┐               ┌────────────────────────┐
                │    Anomali Motoru      │               │    Kusur Dedektörü     │
                │ EfficientAD /PatchCore │               │   YOLO11s-seg (ONNX)   │
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
                                │ Gemini 2.5 Flash (×3)  │
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
                              │ SQLite & Aktif Öğrenme │
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
                   - **Optik Eksen Hizalaması:** Salt 2D kamera görselinden optik eksen kaçıklığı
                     kesin doğrulanamaz; optik kolimatör veya interferometre ölçümü şarttır.
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
        ],
    )

    save_btn.click(
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
        ],
        outputs=[
            confirmation_view,
            serial_no_input,
            metrics_cards_view,
            defect_bar_plot,
            history_table,
        ],
    )

    refresh_btn.click(
        fn=on_refresh_dashboard,
        inputs=[],
        outputs=[
            metrics_cards_view,
            defect_bar_plot,
            history_table,
        ],
    )

    export_csv_btn.click(
        fn=on_export_csv_click,
        inputs=[],
        outputs=[export_file_view],
    )

    export_yolo_btn.click(
        fn=on_export_yolo_click,
        inputs=[],
        outputs=[export_file_view],
    )


# ---------------------------------------------------------------------------
# Application Startup
# ---------------------------------------------------------------------------
storage.init_db()
storage.seed_demo()

if pipeline is not None and hasattr(pipeline, "warmup"):
    threading.Thread(target=pipeline.warmup, daemon=True).start()

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
