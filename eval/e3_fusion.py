"""Evaluation script for E3: End-to-end decision fusion and tuning."""
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PIL import Image

import config
from calib import Calibrator
from engines.anomaly import AnomalyEngine, warm_bank
from engines.detector import Detector
from eval.dataset import CATEGORIES, load_split
from eval.fusion_sim import simulate_fusion

RAW_DIR = config.ROOT / "eval_results" / "raw"
RESULTS_DIR = config.ROOT / "eval_results"


def extract_engine_outputs() -> List[Dict]:
    """Run WRN50 anomaly and YOLO detector on all test images."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    outputs_csv = RAW_DIR / "e3_engine_outputs.csv"

    split_info = load_split()
    path_to_item = {}
    for item in split_info["tune"] + split_info["test"]:
        path_to_item[item["path"]] = item

    all_paths = sorted(path_to_item.keys())
    print(f"Extracting engine outputs for {len(all_paths)} images...")

    # Load pre-computed anomaly scores from e1_scores.csv if available
    e1_csv = RAW_DIR / "e1_scores.csv"
    cached_anom = {}
    if e1_csv.exists():
        with open(e1_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["method"] == "patchcore_wide_resnet50_2":
                    cached_anom[r["path"]] = (float(r["score"]), float(r["p_value"]))

    detector = Detector()
    anomaly_engines = {}
    calibrators = {}
    if not cached_anom:
        for cat in CATEGORIES:
            warm_bank(cat)
            eng = AnomalyEngine(cat)
            cal = Calibrator(cat, eng)
            anomaly_engines[cat] = eng
            calibrators[cat] = cal

    outputs = []
    t0 = time.time()
    for idx, path in enumerate(all_paths):
        item = path_to_item[path]
        cat = item["category"]
        abs_path = config.ROOT / path
        img = Image.open(abs_path).convert("RGB")

        # Anomaly
        if path in cached_anom:
            anom_score, anom_p = cached_anom[path]
        else:
            eng = anomaly_engines[cat]
            cal = calibrators[cat]
            anom_res = eng.predict(img)
            anom_score = float(anom_res["score"])
            anom_p = float(cal.p_value(anom_score)) if len(cal.scores) else 1.0

        # Detector
        det_res = detector.predict(img)
        dets = det_res.get("detections", [])
        if dets:
            top_det = max(dets, key=lambda d: d["conf"])
            det_conf = float(top_det["conf"])
            det_label = top_det["label_tr"]
        else:
            det_conf = 0.0
            det_label = None

        row = {
            "path": path,
            "category": cat,
            "label": item["label"],
            "defect_folder": item["defect_folder"],
            "defect_type": item["defect_type"],
            "split": item["split"],
            "anomaly_score": anom_score,
            "anomaly_p_value": anom_p,
            "detector_conf": det_conf,
            "detector_label": det_label or "",
        }
        outputs.append(row)
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(all_paths)} images (elapsed: {time.time() - t0:.1f}s)...")

    # Save to CSV
    fieldnames = [
        "path", "category", "label", "defect_folder", "defect_type", "split",
        "anomaly_score", "anomaly_p_value", "detector_conf", "detector_label"
    ]
    with open(outputs_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in outputs:
            writer.writerow(r)
    print(f"Saved e3 engine outputs to {outputs_csv} ({len(outputs)} rows)")
    return outputs


def load_or_extract_engine_outputs() -> List[Dict]:
    """Load engine outputs from raw CSV if exists, otherwise extract."""
    outputs_csv = RAW_DIR / "e3_engine_outputs.csv"
    if outputs_csv.exists():
        outputs = []
        with open(outputs_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                outputs.append({
                    "path": r["path"],
                    "category": r["category"],
                    "label": int(r["label"]),
                    "defect_folder": r["defect_folder"],
                    "defect_type": r["defect_type"],
                    "split": r["split"],
                    "anomaly_score": float(r["anomaly_score"]),
                    "anomaly_p_value": float(r["anomaly_p_value"]),
                    "detector_conf": float(r["detector_conf"]),
                    "detector_label": r["detector_label"] or None,
                })
        return outputs
    return extract_engine_outputs()


def evaluate_split(
    items: List[Dict],
    w_anomaly: float,
    w_detector: float,
    accept_below: float,
    reject_above: float,
    anomaly_vote_p: float,
    detector_vote_conf: float,
) -> Dict[str, float]:
    """Evaluate fusion decisions on a set of items and compute QC metrics."""
    decisions = []
    pred_types = []

    for item in items:
        res = simulate_fusion(
            anomaly_p_value=item["anomaly_p_value"],
            detector_conf=item["detector_conf"],
            detector_label=item["detector_label"],
            category=item["category"],
            w_anomaly=w_anomaly,
            w_detector=w_detector,
            accept_below=accept_below,
            reject_above=reject_above,
            anomaly_vote_p=anomaly_vote_p,
            detector_vote_conf=detector_vote_conf,
        )
        decisions.append(res["decision"])
        pred_types.append(res["defect_type"])

    n_total = len(items)
    defects = [i for i, item in enumerate(items) if item["label"] == 1]
    goods = [i for i, item in enumerate(items) if item["label"] == 0]

    # 1. Escape rate: defect -> ACCEPT
    escapes = sum(1 for i in defects if decisions[i] == "ACCEPT")
    escape_rate = escapes / len(defects) if defects else 0.0

    # 2. False reject rate: good -> REJECT
    false_rejects = sum(1 for i in goods if decisions[i] == "REJECT")
    false_reject_rate = false_rejects / len(goods) if goods else 0.0

    # 3. Human review rate
    reviews = sum(1 for d in decisions if d == "REVIEW")
    review_rate = reviews / n_total if n_total else 0.0

    # 4. Auto-decision accuracy (on ACCEPT and REJECT)
    auto_indices = [i for i, d in enumerate(decisions) if d in ("ACCEPT", "REJECT")]
    if auto_indices:
        correct_auto = sum(
            1 for i in auto_indices
            if (decisions[i] == "ACCEPT" and items[i]["label"] == 0) or
               (decisions[i] == "REJECT" and items[i]["label"] == 1)
        )
        auto_accuracy = correct_auto / len(auto_indices)
    else:
        auto_accuracy = 0.0

    # 5. Defect-type accuracy on REJECT
    reject_indices = [i for i, d in enumerate(decisions) if d == "REJECT"]
    if reject_indices:
        correct_dtype = sum(
            1 for i in reject_indices
            if pred_types[i] == items[i]["defect_type"]
        )
        defect_type_acc_on_reject = correct_dtype / len(reject_indices)
    else:
        defect_type_acc_on_reject = 0.0

    return {
        "escape_rate": round(escape_rate, 4),
        "false_reject_rate": round(false_reject_rate, 4),
        "review_rate": round(review_rate, 4),
        "auto_accuracy": round(auto_accuracy, 4),
        "defect_type_acc_on_reject": round(defect_type_acc_on_reject, 4),
        "num_escapes": escapes,
        "num_false_rejects": false_rejects,
        "num_reviews": reviews,
        "num_auto": len(auto_indices),
        "num_rejects": len(reject_indices),
        "num_total": n_total,
    }


def grid_search_tune(tune_items: List[Dict]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """Grid-search fusion parameters on TUNE split.
    
    Objective: minimize human review rate
    Subject to:
      1) escape_rate == 0.0 (fallback <= 0.02)
      2) false_reject_rate <= 0.05
    """
    param_grid = {
        "w_anomaly": [0.35, 0.45, 0.55, 0.65],
        "w_detector": [0.20, 0.25, 0.35],
        "accept_below": [0.20, 0.25, 0.30, 0.35, 0.40],
        "reject_above": [0.55, 0.60, 0.65, 0.70, 0.75],
        "anomaly_vote_p": [0.02, 0.03, 0.05, 0.053, 0.08, 0.10],
        "detector_vote_conf": [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    }

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Evaluating {len(combinations)} parameter combinations on TUNE split ({len(tune_items)} items)...")

    valid_candidates_zero_escape = []
    valid_candidates_fallback = []

    for params in combinations:
        metrics = evaluate_split(tune_items, **params)
        esc = metrics["escape_rate"]
        fr = metrics["false_reject_rate"]
        rev = metrics["review_rate"]
        acc = metrics["auto_accuracy"]

        candidate = (params, metrics)
        if esc == 0.0 and fr <= 0.05:
            valid_candidates_zero_escape.append(candidate)
        elif esc <= 0.02 and fr <= 0.05:
            valid_candidates_fallback.append(candidate)

    if valid_candidates_zero_escape:
        print(f"Found {len(valid_candidates_zero_escape)} candidates satisfying escape_rate == 0.0 and false_reject <= 0.05")
        # Sort by: 1) review_rate ascending, 2) auto_accuracy descending, 3) false_reject ascending
        valid_candidates_zero_escape.sort(
            key=lambda c: (c[1]["review_rate"], -c[1]["auto_accuracy"], c[1]["false_reject_rate"])
        )
        best_params, best_tune_metrics = valid_candidates_zero_escape[0]
    elif valid_candidates_fallback:
        print(f"Warning: No candidate with escape_rate == 0. Using fallback (escape <= 0.02). Found {len(valid_candidates_fallback)}")
        valid_candidates_fallback.sort(
            key=lambda c: (c[1]["review_rate"], c[1]["escape_rate"], -c[1]["auto_accuracy"])
        )
        best_params, best_tune_metrics = valid_candidates_fallback[0]
    else:
        print("Warning: No candidate met strict constraints. Selecting minimum escape rate then lowest review rate.")
        all_evals = [(p, evaluate_split(tune_items, **p)) for p in combinations]
        all_evals.sort(key=lambda c: (c[1]["escape_rate"], c[1]["review_rate"]))
        best_params, best_tune_metrics = all_evals[0]

    return best_params, best_tune_metrics


def run_e3():
    """Run E3 end-to-end evaluation, current vs tuned comparison, and save results."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    outputs = load_or_extract_engine_outputs()
    tune_items = [x for x in outputs if x["split"] == "tune"]
    test_items = [x for x in outputs if x["split"] == "test"]

    print(f"Loaded {len(outputs)} images: {len(tune_items)} tune, {len(test_items)} test")

    # Current config parameters
    current_params = {
        "w_anomaly": 0.45,
        "w_detector": 0.25,
        "accept_below": 0.30,
        "reject_above": 0.70,
        "anomaly_vote_p": 0.053,  # p <= 0.02^0.75
        "detector_vote_conf": 0.40,
    }

    # Evaluate current on TEST and TUNE
    current_test_metrics = evaluate_split(test_items, **current_params)
    current_tune_metrics = evaluate_split(tune_items, **current_params)

    print("\n--- CURRENT CONFIG (TEST SPLIT) ---")
    for k, v in current_test_metrics.items():
        print(f"  {k}: {v}")

    # Grid search on TUNE
    print("\n--- GRID SEARCH ON TUNE SPLIT ---")
    best_params, best_tune_metrics = grid_search_tune(tune_items)
    print("Best params found:", json.dumps(best_params, indent=2))
    print("Tuned performance on TUNE split:")
    for k, v in best_tune_metrics.items():
        print(f"  {k}: {v}")

    # Evaluate chosen best params on TEST
    tuned_test_metrics = evaluate_split(test_items, **best_params)
    print("\n--- TUNED CONFIG (TEST SPLIT) ---")
    for k, v in tuned_test_metrics.items():
        print(f"  {k}: {v}")

    # Save e3_best_params.json
    best_params_file = RESULTS_DIR / "e3_best_params.json"
    best_params_out = {
        "params": best_params,
        "tune_metrics": best_tune_metrics,
        "test_metrics": tuned_test_metrics,
        "current_params": current_params,
        "current_test_metrics": current_test_metrics,
    }
    best_params_file.write_text(json.dumps(best_params_out, indent=2))
    print(f"Saved best params to {best_params_file}")

    # Save e3_fusion.csv
    fusion_csv = RESULTS_DIR / "e3_fusion.csv"
    csv_rows = [
        {
            "config": "current",
            "split": "test",
            "escape_rate": current_test_metrics["escape_rate"],
            "false_reject_rate": current_test_metrics["false_reject_rate"],
            "human_review_rate": current_test_metrics["review_rate"],
            "auto_accuracy": current_test_metrics["auto_accuracy"],
            "defect_type_acc_on_reject": current_test_metrics["defect_type_acc_on_reject"],
            "num_escapes": current_test_metrics["num_escapes"],
            "num_false_rejects": current_test_metrics["num_false_rejects"],
            "num_reviews": current_test_metrics["num_reviews"],
            "num_auto": current_test_metrics["num_auto"],
            "num_rejects": current_test_metrics["num_rejects"],
            "w_anomaly": current_params["w_anomaly"],
            "w_detector": current_params["w_detector"],
            "accept_below": current_params["accept_below"],
            "reject_above": current_params["reject_above"],
            "anomaly_vote_p": current_params["anomaly_vote_p"],
            "detector_vote_conf": current_params["detector_vote_conf"],
        },
        {
            "config": "tuned",
            "split": "test",
            "escape_rate": tuned_test_metrics["escape_rate"],
            "false_reject_rate": tuned_test_metrics["false_reject_rate"],
            "human_review_rate": tuned_test_metrics["review_rate"],
            "auto_accuracy": tuned_test_metrics["auto_accuracy"],
            "defect_type_acc_on_reject": tuned_test_metrics["defect_type_acc_on_reject"],
            "num_escapes": tuned_test_metrics["num_escapes"],
            "num_false_rejects": tuned_test_metrics["num_false_rejects"],
            "num_reviews": tuned_test_metrics["num_reviews"],
            "num_auto": tuned_test_metrics["num_auto"],
            "num_rejects": tuned_test_metrics["num_rejects"],
            "w_anomaly": best_params["w_anomaly"],
            "w_detector": best_params["w_detector"],
            "accept_below": best_params["accept_below"],
            "reject_above": best_params["reject_above"],
            "anomaly_vote_p": best_params["anomaly_vote_p"],
            "detector_vote_conf": best_params["detector_vote_conf"],
        },
        {
            "config": "current",
            "split": "tune",
            "escape_rate": current_tune_metrics["escape_rate"],
            "false_reject_rate": current_tune_metrics["false_reject_rate"],
            "human_review_rate": current_tune_metrics["review_rate"],
            "auto_accuracy": current_tune_metrics["auto_accuracy"],
            "defect_type_acc_on_reject": current_tune_metrics["defect_type_acc_on_reject"],
            "num_escapes": current_tune_metrics["num_escapes"],
            "num_false_rejects": current_tune_metrics["num_false_rejects"],
            "num_reviews": current_tune_metrics["num_reviews"],
            "num_auto": current_tune_metrics["num_auto"],
            "num_rejects": current_tune_metrics["num_rejects"],
            "w_anomaly": current_params["w_anomaly"],
            "w_detector": current_params["w_detector"],
            "accept_below": current_params["accept_below"],
            "reject_above": current_params["reject_above"],
            "anomaly_vote_p": current_params["anomaly_vote_p"],
            "detector_vote_conf": current_params["detector_vote_conf"],
        },
        {
            "config": "tuned",
            "split": "tune",
            "escape_rate": best_tune_metrics["escape_rate"],
            "false_reject_rate": best_tune_metrics["false_reject_rate"],
            "human_review_rate": best_tune_metrics["review_rate"],
            "auto_accuracy": best_tune_metrics["auto_accuracy"],
            "defect_type_acc_on_reject": best_tune_metrics["defect_type_acc_on_reject"],
            "num_escapes": best_tune_metrics["num_escapes"],
            "num_false_rejects": best_tune_metrics["num_false_rejects"],
            "num_reviews": best_tune_metrics["num_reviews"],
            "num_auto": best_tune_metrics["num_auto"],
            "num_rejects": best_tune_metrics["num_rejects"],
            "w_anomaly": best_params["w_anomaly"],
            "w_detector": best_params["w_detector"],
            "accept_below": best_params["accept_below"],
            "reject_above": best_params["reject_above"],
            "anomaly_vote_p": best_params["anomaly_vote_p"],
            "detector_vote_conf": best_params["detector_vote_conf"],
        },
    ]

    fieldnames = list(csv_rows[0].keys())
    with open(fusion_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)
    print(f"Saved e3 fusion metrics to {fusion_csv}")


if __name__ == "__main__":
    run_e3()
