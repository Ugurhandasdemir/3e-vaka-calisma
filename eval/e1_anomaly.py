"""Evaluation script for E1: Anomaly detection engines benchmark."""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from PIL import Image

import config
from eval.dataset import CATEGORIES, load_split
from eval.metrics import compute_auroc, compute_conformal_p_values, evaluate_binary_predictions

RAW_DIR = config.ROOT / "eval_results" / "raw"
RESULTS_DIR = config.ROOT / "eval_results"


def run_worker_patchcore(backbone: str) -> List[Dict]:
    """Run PatchCore benchmark with specified backbone (wide_resnet50_2 or resnet18)."""
    tmp_out = RAW_DIR / f".tmp_{backbone}.json"
    worker_script = f"""
import os, sys, time, json
from pathlib import Path

os.environ['PATCHCORE_BACKBONE'] = '{backbone}'
_ROOT = Path('{config.ROOT}')
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import numpy as np
from PIL import Image
import config
from engines.anomaly import AnomalyEngine, warm_bank
from calib import Calibrator
from eval.dataset import list_category_images, CATEGORIES

torch.set_num_threads(4)

results = []
for cat in CATEGORIES:
    warm_bank(cat)
    eng = AnomalyEngine(cat)
    cal = Calibrator(cat, eng)
    cal_scores = list(cal.scores)
    
    test_items = list_category_images(cat)
    for item in test_items:
        img = Image.open(item['abs_path']).convert('RGB')
        t0 = time.time()
        res = eng.predict(img)
        lat = int((time.time() - t0) * 1000)
        score = float(res['score'])
        p = cal.p_value(score) if len(cal.scores) else 1.0
        
        results.append({{
            'path': item['path'],
            'category': cat,
            'label': item['label'],
            'defect_folder': item['defect_folder'],
            'defect_type': item['defect_type'],
            'method': f'patchcore_{backbone}',
            'score': score,
            'p_value': p,
            'pred_defect': int(p <= 0.05),
            'latency_ms': lat,
        }})

with open('{tmp_out.as_posix()}', 'w', encoding='utf-8') as f:
    json.dump(results, f)
"""
    cmd = [str(config.ROOT / ".venv" / "bin" / "python"), "-c", worker_script]
    env = os.environ.copy()
    env["PATCHCORE_BACKBONE"] = backbone
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    if not tmp_out.exists():
        raise RuntimeError(f"Worker did not create {tmp_out}. Stderr: {proc.stderr}")
    data = json.loads(tmp_out.read_text(encoding="utf-8"))
    tmp_out.unlink(missing_ok=True)
    return data


def run_worker_dino() -> List[Dict]:
    """Run AnomalyDINO benchmark."""
    from eval.anomaly_dino import AnomalyDINOEngine
    from eval.dataset import list_category_images

    torch.set_num_threads(4)
    results = []
    for cat in CATEGORIES:
        eng = AnomalyDINOEngine(cat)
        test_items = list_category_images(cat)
        for item in test_items:
            img = Image.open(item["abs_path"]).convert("RGB")
            t0 = time.time()
            res = eng.predict(img)
            lat = int((time.time() - t0) * 1000)
            score = float(res["score"])
            p = eng.p_value(score)

            results.append({
                "path": item["path"],
                "category": cat,
                "label": item["label"],
                "defect_folder": item["defect_folder"],
                "defect_type": item["defect_type"],
                "method": "anomaly_dino_vits14",
                "score": score,
                "p_value": p,
                "pred_defect": int(p <= 0.05),
                "latency_ms": lat,
            })
    return results


def run_e1():
    """Execute all anomaly engine benchmarks, save raw scores and summary metrics."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load split to attach split tag ('tune' or 'test') to scores
    split_info = load_split()
    path_to_split = {x["path"]: "tune" for x in split_info["tune"]}
    path_to_split.update({x["path"]: "test" for x in split_info["test"]})

    all_scores: List[Dict] = []
    summary_rows: List[Dict] = []

    methods_to_run = [
        ("PatchCore (wide_resnet50_2)", "wide_resnet50_2"),
        ("PatchCore (resnet18)", "resnet18"),
        ("AnomalyDINO (ViT-S/14)", "dino"),
    ]

    for method_name, method_key in methods_to_run:
        print(f"=== Running {method_name} ===")
        t0 = time.time()
        if method_key in ("wide_resnet50_2", "resnet18"):
            records = run_worker_patchcore(method_key)
        elif method_key == "dino":
            records = run_worker_dino()
        else:
            records = []
        print(f"  Finished in {time.time() - t0:.2f} s. Processed {len(records)} image scores.")

        for r in records:
            r["split"] = path_to_split.get(r["path"], "unknown")
            all_scores.append(r)

        # Compute summary metrics per category
        for cat in CATEGORIES:
            cat_records = [r for r in records if r["category"] == cat]
            y_true = [r["label"] for r in cat_records]
            y_scores = [r["score"] for r in cat_records]
            p_vals = [r["p_value"] for r in cat_records]
            latencies = [r["latency_ms"] for r in cat_records]

            auroc = compute_auroc(y_true, y_scores)
            preds_p005 = [1 if p <= 0.05 else 0 for p in p_vals]
            metrics = evaluate_binary_predictions(y_true, preds_p005)

            summary_rows.append({
                "method": method_name,
                "category": cat,
                "auroc": round(auroc, 4),
                "recall_at_p005": round(metrics["tpr"], 4),
                "fpr_at_p005": round(metrics["fpr"], 4),
                "mean_latency_ms": round(float(np.mean(latencies)), 1),
                "num_images": len(cat_records),
                "num_defect": sum(y_true),
                "num_good": len(y_true) - sum(y_true),
            })

    # Check EfficientAD ONNX
    print("=== Checking EfficientAD ONNX ===")
    for cat in CATEGORIES:
        onnx_file = config.anomaly_onnx(cat)
        if not onnx_file.exists():
            print(f"  EfficientAD ONNX model for {cat} not found ({onnx_file.name}); skipping as instructed.")
            summary_rows.append({
                "method": "EfficientAD (ONNX)",
                "category": cat,
                "auroc": "N/A (model yok)",
                "recall_at_p005": "N/A",
                "fpr_at_p005": "N/A",
                "mean_latency_ms": "N/A",
                "num_images": 0,
                "num_defect": 0,
                "num_good": 0,
            })

    # Save e1_scores.csv
    scores_csv = RAW_DIR / "e1_scores.csv"
    if all_scores:
        fieldnames = ["path", "category", "label", "defect_folder", "defect_type", "split",
                      "method", "score", "p_value", "pred_defect", "latency_ms"]
        with open(scores_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_scores:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"Saved per-image scores to {scores_csv} ({len(all_scores)} rows)")

    # Save e1_anomaly.csv
    summary_csv = RESULTS_DIR / "e1_anomaly.csv"
    if summary_rows:
        sum_fields = ["method", "category", "auroc", "recall_at_p005", "fpr_at_p005", "mean_latency_ms",
                      "num_images", "num_defect", "num_good"]
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sum_fields)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        print(f"Saved anomaly engine benchmark to {summary_csv}")

    return summary_rows, all_scores


if __name__ == "__main__":
    run_e1()
