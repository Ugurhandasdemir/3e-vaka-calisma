"""E3b: fusion search with the production ensemble anomaly engine (WRN50 + AnomalyDINO)."""
import csv, itertools, json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import numpy as np
from PIL import Image
import config
from calib import Calibrator
from engines.anomaly import AnomalyEngine
from eval.e3_fusion import evaluate_split, load_or_extract_engine_outputs
from eval.metrics import compute_auroc

R = config.ROOT / "eval_results"
out = load_or_extract_engine_outputs()
engs = {}
for r in out:
    c = r["category"]
    if c not in engs:
        e = AnomalyEngine(c); assert e.backend == "patchcore+dinov2", e.backend
        engs[c] = (e, Calibrator(c, e))
    e, cal = engs[c]
    s = e.predict(Image.open(config.ROOT / r["path"]).convert("RGB"))["score"]
    r["ens_score"], r["ens_p"] = s, cal.p_value(s)
    r["old_p"] = r["anomaly_p_value"]

with open(R / "e3b_ensemble.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["path", "category", "label", "split", "ens_score", "ens_p", "wrn_p", "detector_conf", "detector_label"])
    for r in out:
        w.writerow([r["path"], r["category"], r["label"], r["split"], r["ens_score"], r["ens_p"], r["old_p"], r["detector_conf"], r["detector_label"] or ""])

def quick(items, key, ps):
    y = np.array([i["label"] for i in items]); sc = np.array([i[key] for i in items])
    p = np.array([i[ps] for i in items])
    return compute_auroc(y, sc) if key.endswith("score") else None, float((p[y == 1] <= .05).mean()), float((p[y == 0] <= .05).mean())

test = [r for r in out if r["split"] == "test"]; tune = [r for r in out if r["split"] == "tune"]
print("TEST AUROC/recall@.05/FPR@.05 ens:", quick(test, "ens_score", "ens_p"))
print("TEST wrn recall/FPR:", quick(test, "anomaly_score", "old_p"))
for c in engs:
    print(c, "ens", quick([r for r in test if r["category"] == c], "ens_score", "ens_p"))

for r in out: r["anomaly_p_value"] = r["ens_p"]
grid = {"w_anomaly": [0.35, 0.45, 0.55, 0.65], "w_detector": [0.20, 0.25, 0.35],
        "accept_below": [0.20, 0.25, 0.30, 0.35, 0.40], "reject_above": [0.55, 0.60, 0.65, 0.70, 0.75],
        "anomaly_vote_p": [0.0244, 0.0488, 0.0732, 0.0976, 0.12, 0.15],
        "detector_vote_conf": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]}
ks = list(grid)
cands = []
for v in itertools.product(*grid.values()):
    p = dict(zip(ks, v)); m = evaluate_split(tune, **p)
    if m["escape_rate"] == 0 and m["false_reject_rate"] <= 0.05:
        cands.append((m["review_rate"], -m["auto_accuracy"], m["false_reject_rate"], -p["anomaly_vote_p"], p, m))
cands.sort(key=lambda c: c[:4])
print(len(cands), "valid candidates")
best, mt = cands[0][4], cands[0][5]
res = {"params": best, "tune_metrics": mt, "test_metrics": evaluate_split(test, **best),
       "previous_test_metrics": json.load(open(R / "e3_best_params.json"))["current_test_metrics"]}
json.dump(res, open(R / "e3b_best_params.json", "w"), indent=2)
print(json.dumps(res, indent=1))
