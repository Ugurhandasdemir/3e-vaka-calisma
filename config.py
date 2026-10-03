"""Shared configuration and contracts for the 3E QC PoC.

Every module imports constants from here; do not duplicate them elsewhere.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))  # SQLite + saved images
DATA_DIR.mkdir(parents=True, exist_ok=True)

APP_VERSION = "0.1.0"

# Product group (UI, Turkish) -> MVTec AD category used as representative data.
PRODUCT_GROUPS = {
    "Optik lens grubu": "metal_nut",
    "Termal kamera modülü": "transistor",
    "Gözetleme ünitesi": "cable",
}
HIGH_RISK_GROUPS = {"Termal kamera modülü"}  # recall history -> never auto-accept a flagged item

# Our defect taxonomy (Turkish labels shown in UI and stored in DB).
DEFECT_TYPES = [
    "Yüzey çiziği",
    "Kaplama kusuru",
    "Konektör/montaj kusuru",
    "Optik eksen şüphesi",
    "Diğer",
    "Yok",
]

# MVTec defect folder / YOLO class name -> our taxonomy. Matching is by substring, lowercase.
CLASS_MAP = {
    "scratch": "Yüzey çiziği",
    "damaged_case": "Yüzey çiziği",
    "color": "Kaplama kusuru",
    "contamination": "Kaplama kusuru",
    "bent": "Konektör/montaj kusuru",
    "cut_lead": "Konektör/montaj kusuru",
    "misplaced": "Konektör/montaj kusuru",
    "flip": "Konektör/montaj kusuru",
    "manipulated": "Konektör/montaj kusuru",
    "thread": "Konektör/montaj kusuru",
    "cable_swap": "Konektör/montaj kusuru",
    "missing": "Konektör/montaj kusuru",
    "insulation": "Kaplama kusuru",
    "coating": "Kaplama kusuru",
    "connector": "Konektör/montaj kusuru",
}


def map_class(name: str) -> str:
    n = (name or "").lower()
    for key, label in CLASS_MAP.items():
        if key in n:
            return label
    return "Diğer"


# Model files (optional; engines fall back gracefully if missing).
def anomaly_onnx(category: str) -> Path:
    return MODELS_DIR / f"anomaly_{category}.onnx"


def calib_file(category: str) -> Path:
    return MODELS_DIR / f"calib_{category}.npy"


# Anomaly ensemble: PatchCore-WRN50 + AnomalyDINO (DINOv2 ViT-S/14), combine-then-calibrate
# (eval_results/BIRLESTIRME.md section 4). Set ANOMALY_ENSEMBLE=false for WRN50-only.
ANOMALY_ENSEMBLE = os.getenv("ANOMALY_ENSEMBLE", "true").strip().lower() not in ("0", "false", "no", "off")


def dino_bank_file(category: str) -> Path:
    return MODELS_DIR / f"bank_{category}_dinov2_vits14.npy"


def ensemble_calib_file(category: str) -> Path:
    return MODELS_DIR / f"calib_{category}_ensemble.npy"


def ensemble_meta_file(category: str) -> Path:
    return MODELS_DIR / f"calib_{category}_ensemble.json"


YOLO_MODEL = MODELS_DIR / os.getenv("YOLO_MODEL_FILE", "yolo.onnx")  # .onnx or .pt, det or seg

# VLM
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODELS = [m for m in os.getenv(
    "GEMINI_FALLBACK_MODELS", "gemini-3.8-flash,gemini-3.1-flash-lite").split(",") if m]
VLM_SAMPLES = int(os.getenv("VLM_SAMPLES", "3"))
VLM_TIMEOUT_S = 25

# Fusion (see fusion.py docstring for semantics)
# Tuned on the TUNE split with the ensemble anomaly engine (eval_results/e3b_best_params.json).
WEIGHTS = {"anomaly": 0.45, "detector": 0.25, "vlm": 0.30}
VOTE_THRESHOLDS = {"anomaly": 0.485,  # a-scale; a >= 0.485  <=>  p_value <= 0.15 (ensemble p, n=40)
                   "detector": 0.30, "vlm": 0.50}
ACCEPT_BELOW = 0.20
REJECT_ABOVE = 0.55
EARLY_EXIT_P = 0.50  # p_value above this and no detections -> skip VLM, suggest accept
DETECTOR_CONF = 0.25

DECISIONS = {"ACCEPT": "KABUL önerisi", "REVIEW": "İNSAN İNCELEMESİ", "REJECT": "RET önerisi"}
