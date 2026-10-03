"""Inspection pipeline: anomaly -> detector -> (early exit | VLM) -> fusion. See docs/CONTRACT.md."""
import os
import time

import cv2
import numpy as np
from PIL import Image

import config
from calib import Calibrator
from engines.anomaly import AnomalyEngine, warm_bank
from engines.detector import Detector
from engines.vlm import VLMEngine, _empty
from fusion import fuse

_anom, _calib = {}, {}
_detector = None
_vlm = None


def _get_anom(cat):
    if cat not in _anom:
        _anom[cat] = AnomalyEngine(cat)
        _calib[cat] = Calibrator(cat, _anom[cat])
    return _anom[cat], _calib[cat]


def _get_detector():
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


def _get_vlm():
    global _vlm
    if _vlm is None:
        _vlm = VLMEngine()
    return _vlm


def warmup():
    """Build PatchCore banks + calibration for all categories (call at app start)."""
    for cat in set(config.PRODUCT_GROUPS.values()):
        try:
            eng, cal = _get_anom(cat)
            if eng.backend == "patchcore":
                warm_bank(cat)
            _ = cal.scores
        except Exception:
            pass
    _get_detector()
    _get_vlm()


def _overlay(img, hm):
    base = np.asarray(img.convert("RGB"))
    col = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
    return Image.fromarray(cv2.addWeighted(base, 0.55, col, 0.45, 0))


def run(image, product_group):
    t0 = time.time()
    image = image.convert("RGB")
    cat = config.PRODUCT_GROUPS[product_group]

    # anomaly
    an = {"available": False, "backend": None, "score": 0.0, "p_value": 1.0, "prob": 0.0, "latency_ms": 0, "error": None}
    overlay = None
    try:
        eng, cal = _get_anom(cat)
        r = eng.predict(image)
        if r["backend"] is None:
            an["error"] = r["error"]
        else:
            p = cal.p_value(r["score"]) if len(cal.scores) else 1.0
            an.update(available=True, backend=r["backend"], score=r["score"], p_value=p, prob=1 - p,
                      latency_ms=r["latency_ms"], error=r["error"])
            overlay = _overlay(image, r["heatmap"])
        anomaly_version = eng.version
    except Exception as e:
        an["error"] = str(e)
        anomaly_version = None

    # detector
    det = _get_detector()
    d = det.predict(image)
    dets = d["detections"]
    dengine = {"available": d["available"], "detections": dets,
               "prob": max((x["conf"] for x in dets), default=0.0),
               "latency_ms": d["latency_ms"], "error": d["error"]}

    # VLM / early exit
    vlm_engine = _get_vlm()
    # High-risk groups (thermal module, recall history) never take the early exit: all engines run.
    early = (an["available"] and an["p_value"] > config.EARLY_EXIT_P and not dets
             and product_group not in config.HIGH_RISK_GROUPS)
    if early:
        v = _empty(None if vlm_engine.has_key else "GEMINI_API_KEY yok", called=False)
        v["available"] = vlm_engine.has_key
        v_for_fusion = dict(v, available=False)
    else:
        v = vlm_engine.analyze(image, overlay, product_group, an, dengine)
        v_for_fusion = v

    fz = fuse(an, dengine, v_for_fusion, product_group, early_exit=early)

    return {
        "decision": fz["decision"], "decision_label": fz["decision_label"],
        "confidence": fz["confidence"], "defect_score": fz["defect_score"],
        "defect_type": fz["defect_type"], "severity": fz["severity"], "reasons": fz["reasons"],
        "heatmap_overlay": overlay, "detection_image": d["annotated"],
        "engines": {"anomaly": an, "detector": dengine, "vlm": v},
        "model_versions": {"anomaly": anomaly_version, "detector": det.version,
                           "vlm": config.GEMINI_MODEL, "app": config.APP_VERSION},
        "latency_ms": int((time.time() - t0) * 1000),
    }
