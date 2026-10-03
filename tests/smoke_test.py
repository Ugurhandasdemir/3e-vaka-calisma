"""Smoke test: python tests/smoke_test.py (no Gemini key needed)."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.pop("GEMINI_API_KEY", None)
os.environ.pop("GOOGLE_API_KEY", None)

import numpy as np
from PIL import Image

import config
import pipeline

KEYS = {"decision", "decision_label", "confidence", "defect_score", "defect_type", "severity", "reasons",
        "heatmap_overlay", "detection_image", "engines", "model_versions", "latency_ms"}
ENG = {"anomaly": {"available", "backend", "score", "p_value", "prob", "latency_ms", "error"},
       "detector": {"available", "detections", "prob", "latency_ms", "error"},
       "vlm": {"available", "called", "model", "results", "majority_type", "consistency", "prob",
               "reasoning", "location", "latency_ms", "error"}}

# wait for samples
for _ in range(60):
    if all(len(list((config.SAMPLES_DIR / c / s).glob("*.png"))) >= 2
           for c in set(config.PRODUCT_GROUPS.values()) for s in ("good", "calib", "defect", "test_good")):
        break
    time.sleep(10)

state = {"defect": False}


def fake_analyze(image, overlay, product_group, anomaly, detector):
    from engines.vlm import aggregate
    if state["defect"]:
        r = {"defect_present": True, "defect_type": "Yüzey çiziği", "severity": "Orta", "location": "orta",
             "reasoning": "Sahte VLM.", "self_confidence": 0.85}
    else:
        r = {"defect_present": False, "defect_type": "Yok", "severity": "Düşük", "location": "-",
             "reasoning": "Sahte VLM.", "self_confidence": 0.9}
    return aggregate([dict(r) for _ in range(3)], 5)


def check(res):
    assert KEYS <= set(res), KEYS - set(res)
    for k, ks in ENG.items():
        assert ks <= set(res["engines"][k]), (k, ks - set(res["engines"][k]))
    assert res["decision"] in config.DECISIONS and res["defect_type"] in config.DEFECT_TYPES


t0 = time.time()
pipeline.warmup()
print(f"warmup {time.time()-t0:.1f}s")
rows = []
for fake in (False, True):
    if fake:
        os.environ["GEMINI_API_KEY"] = "fake"
        pipeline._vlm = None
        pipeline._get_vlm().analyze = fake_analyze
    for group, cat in config.PRODUCT_GROUPS.items():
        ps = {"good": [], "defect": []}
        for kind, folder in (("good", "test_good"), ("defect", "defect")):
            files = sorted((config.SAMPLES_DIR / cat / folder).glob("*.png"))[:2]
            for f in files:
                state["defect"] = kind == "defect"
                res = pipeline.run(Image.open(f), group)
                check(res)
                a = res["engines"]["anomaly"]
                assert a["available"], a
                ps[kind].append(a["p_value"])
                rows.append((("fakeVLM" if fake else "noVLM"), cat, kind, res["decision"], res["confidence"],
                             res["defect_score"], a["p_value"], a["latency_ms"], res["latency_ms"]))
        if not fake or True:
            gm, dm = np.mean(ps["good"]), np.mean(ps["defect"])
            assert dm < gm, f"{cat}: defect p {dm:.3f} !< good p {gm:.3f}"

print(f"{'mode':8}{'cat':12}{'kind':8}{'decision':9}{'conf':>6}{'score':>7}{'p':>8}{'anom_ms':>8}{'tot_ms':>8}")
for r in rows:
    print(f"{r[0]:8}{r[1]:12}{r[2]:8}{r[3]:9}{r[4]:6.2f}{r[5]:7.2f}{r[6]:8.3f}{r[7]:8d}{r[8]:8d}")
print("OK")
