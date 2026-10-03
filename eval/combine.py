"""Evaluation script for E1b: Score and p-value combination benchmark.

Combines PatchCore (wide_resnet50_2) and AnomalyDINO (ViT-S/14) scores and p-values
across 11 combiners and evaluates performance on both 'test' split and all data.
"""

import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from eval.metrics import compute_auroc

# Paths
SCORES_FILE = _ROOT / "eval_results" / "raw" / "e1_scores.csv"
OUTPUT_CSV = _ROOT / "eval_results" / "e1b_combine.csv"
CACHE_DIR = _ROOT / "eval_results" / "cache"
MODELS_DIR = _ROOT / "models"

CATEGORIES = ["metal_nut", "transistor", "cable"]
WRN_METHOD = "patchcore_wide_resnet50_2"
DINO_METHOD = "anomaly_dino_vits14"


def get_calib_paths(category: str) -> Tuple[Path, Path]:
    """Get calibration file paths for WRN50 and DINO."""
    p_wrn = MODELS_DIR / f"calib_{category}_patchcore_wide_resnet50_2.npy"
    p_dino = CACHE_DIR / f"calib_{category}_dinov2_vits14.npy"
    return p_wrn, p_dino


def check_calibration_files() -> Dict[str, Dict[str, bool]]:
    """Check existence of all required calibration files."""
    status = {}
    for cat in CATEGORIES:
        p_wrn, p_dino = get_calib_paths(cat)
        status[cat] = {
            "wrn_path": str(p_wrn),
            "wrn_exists": p_wrn.exists(),
            "dino_path": str(p_dino),
            "dino_exists": p_dino.exists(),
        }
    return status


def load_calibration_data(category: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load calibration numpy arrays for a given category."""
    p_wrn, p_dino = get_calib_paths(category)
    if not p_wrn.exists():
        raise FileNotFoundError(f"Missing WRN calibration file: {p_wrn}. Recomputing nothing per instructions.")
    if not p_dino.exists():
        raise FileNotFoundError(f"Missing DINO calibration file: {p_dino}. Recomputing nothing per instructions.")

    c_wrn = np.load(p_wrn).astype(np.float64).ravel()
    c_dino = np.load(p_dino).astype(np.float64).ravel()
    return c_wrn, c_dino


# ---------------------------------------------------------
# Mathematical and Statistical Helpers
# ---------------------------------------------------------

def normal_survival_p(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Survival function 1 - Phi(z) = 0.5 * erfc(z / sqrt(2))."""
    z_arr = np.asarray(z, dtype=np.float64)
    # Using math.erfc via vectorization for guaranteed numerical stability
    v_erfc = np.vectorize(math.erfc)
    return 0.5 * v_erfc(z_arr / math.sqrt(2.0))


def chi2_4_survival_p(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Survival function P(chi2(4) >= x) = (1 + x/2) * exp(-x/2) for x >= 0."""
    x_arr = np.maximum(0.0, np.asarray(x, dtype=np.float64))
    return (1.0 + x_arr / 2.0) * np.exp(-x_arr / 2.0)


def roc_max_recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05) -> float:
    """Calculate maximum achievable recall at FPR <= target_fpr using score thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0:
        return 0.0
    if n_neg == 0:
        return 1.0

    desc_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_sorted = y_score[desc_indices]
    y_true_sorted = y_true[desc_indices]

    distinct_indices = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, y_true_sorted.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tpr = tps / n_pos
    fpr = fps / n_neg

    valid = fpr <= target_fpr
    if not np.any(valid):
        return 0.0
    return float(np.max(tpr[valid]))


def get_fractional_ranks(v: np.ndarray) -> np.ndarray:
    """Compute tied ranks normalized to [0, 1]."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return np.array([])
    order = np.argsort(v)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)

    unique_vals, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    if len(unique_vals) < n:
        tie_ranks = np.zeros(len(unique_vals), dtype=float)
        np.add.at(tie_ranks, inverse, ranks)
        tie_ranks /= counts
        ranks = tie_ranks[inverse]
    return ranks / float(n)


# ---------------------------------------------------------
# Combiner Implementations (a) through (k)
# ---------------------------------------------------------

def combine_wrn50_alone(sw: np.ndarray, pw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(a) WRN50 alone."""
    return sw.copy(), pw.copy()


def combine_dino_alone(sd: np.ndarray, pd_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(b) DINO alone."""
    return sd.copy(), pd_val.copy()


def combine_zscore_mean(
    sw: np.ndarray, sd: np.ndarray, cw: np.ndarray, cd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """(c) z-score mean using calibration mean/std (parametric Gaussian p)."""
    mu_w, std_w = float(np.mean(cw)), float(np.std(cw))
    mu_d, std_d = float(np.mean(cd)), float(np.std(cd))
    zw = (sw - mu_w) / (std_w if std_w > 0 else 1.0)
    zd = (sd - mu_d) / (std_d if std_d > 0 else 1.0)
    s_comb = (zw + zd) / 2.0
    p_comb = normal_survival_p(s_comb)
    return s_comb, p_comb


def combine_minmax_mean(
    sw: np.ndarray, sd: np.ndarray, cw: np.ndarray, cd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """(d) min-max(calibration) mean (Kriegel et al. 2011 outlier probability)."""
    min_w, max_w = float(np.min(cw)), float(np.max(cw))
    min_d, max_d = float(np.min(cd)), float(np.max(cd))
    range_w = (max_w - min_w) if max_w > min_w else 1.0
    range_d = (max_d - min_d) if max_d > min_d else 1.0
    nw = (sw - min_w) / range_w
    nd = (sd - min_d) / range_d
    s_comb = (nw + nd) / 2.0
    # Following Kriegel 2011, score in [0, 1] interpreted as P(outlier) => p = 1 - P(outlier)
    p_comb = np.clip(1.0 - s_comb, 0.0, 1.0)
    return s_comb, p_comb


def combine_max_zscore(
    sw: np.ndarray, sd: np.ndarray, cw: np.ndarray, cd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """(e) max of z-scores (parametric Gaussian p)."""
    mu_w, std_w = float(np.mean(cw)), float(np.std(cw))
    mu_d, std_d = float(np.mean(cd)), float(np.std(cd))
    zw = (sw - mu_w) / (std_w if std_w > 0 else 1.0)
    zd = (sd - mu_d) / (std_d if std_d > 0 else 1.0)
    s_comb = np.maximum(zw, zd)
    p_comb = normal_survival_p(s_comb)
    return s_comb, p_comb


def combine_then_calibrate(
    sw: np.ndarray, sd: np.ndarray, cw: np.ndarray, cd: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """(f) 'combine-then-calibrate': nonconformity = mean of z-scores, conformal p computed

    from the SAME combination applied to calibration images (leave-one-out on calibration).
    """
    mu_w, std_w = float(np.mean(cw)), float(np.std(cw))
    mu_d, std_d = float(np.mean(cd)), float(np.std(cd))
    zw_test = (sw - mu_w) / (std_w if std_w > 0 else 1.0)
    zd_test = (sd - mu_d) / (std_d if std_d > 0 else 1.0)
    s_test = (zw_test + zd_test) / 2.0

    # Calibration images nonconformity scores
    zw_cal = (cw - mu_w) / (std_w if std_w > 0 else 1.0)
    zd_cal = (cd - mu_d) / (std_d if std_d > 0 else 1.0)
    s_cal = (zw_cal + zd_cal) / 2.0

    # Conformal p-value for test items against calibration distribution:
    # p = (1 + #{cal >= test}) / (n_cal + 1)
    counts = np.sum(s_cal[None, :] >= s_test[:, None], axis=1)
    p_comb = (1.0 + counts) / (len(s_cal) + 1.0)
    return s_test, p_comb


def combine_bonferroni(pw: np.ndarray, pd_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(g) Bonferroni min-p * 2."""
    p_comb = np.minimum(1.0, 2.0 * np.minimum(pw, pd_val))
    s_comb = 1.0 - p_comb
    return s_comb, p_comb


def combine_vovk_wang(pw: np.ndarray, pd_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(h) Vovk-Wang 2*arithmetic-mean p: 2 * (p1 + p2)/2 = p1 + p2."""
    p_comb = np.minimum(1.0, pw + pd_val)
    s_comb = 1.0 - p_comb
    return s_comb, p_comb


def combine_cauchy_acat(pw: np.ndarray, pd_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(i) Cauchy combination ACAT (Liu & Xie 2020 JASA)."""
    # Clip p slightly away from 0 and 1 to prevent tan(pi/2) blowup
    p1 = np.clip(pw, 1e-15, 1.0 - 1e-15)
    p2 = np.clip(pd_val, 1e-15, 1.0 - 1e-15)
    t1 = np.tan((0.5 - p1) * np.pi)
    t2 = np.tan((0.5 - p2) * np.pi)
    T = 0.5 * t1 + 0.5 * t2
    p_comb = np.clip(0.5 - np.arctan(T) / np.pi, 0.0, 1.0)
    s_comb = 1.0 - p_comb
    return s_comb, p_comb


def combine_fisher(pw: np.ndarray, pd_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(j) Fisher combination (assumes independence)."""
    p1 = np.clip(pw, 1e-15, 1.0)
    p2 = np.clip(pd_val, 1e-15, 1.0)
    X2 = -2.0 * (np.log(p1) + np.log(p2))
    p_comb = chi2_4_survival_p(X2)
    s_comb = 1.0 - p_comb
    return s_comb, p_comb


def combine_rank_average(sw: np.ndarray, sd: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """(k) Rank average (only as score, no p)."""
    rw = get_fractional_ranks(sw)
    rd = get_fractional_ranks(sd)
    s_comb = (rw + rd) / 2.0
    return s_comb, None


# ---------------------------------------------------------
# Evaluation Runner
# ---------------------------------------------------------

COMBINER_CONFIGS = [
    ("a_wrn50_alone", "WRN50 alone"),
    ("b_dino_alone", "DINO alone"),
    ("c_zscore_mean", "z-score mean"),
    ("d_minmax_mean", "min-max mean"),
    ("e_max_zscore", "max of z-scores"),
    ("f_combine_then_calibrate", "combine-then-calibrate"),
    ("g_bonferroni_min_p", "Bonferroni min-p*2"),
    ("h_vovk_wang_mean_p", "Vovk-Wang 2*mean p"),
    ("i_cauchy_acat", "Cauchy ACAT"),
    ("j_fisher", "Fisher"),
    ("k_rank_average", "Rank average"),
]


def evaluate_predictions(
    y_true: np.ndarray, s_comb: np.ndarray, p_comb: Optional[np.ndarray]
) -> Dict[str, float]:
    """Compute AUROC, recall/FPR at p<=0.05, recall/FPR at p<=0.10, and max recall at FPR<=5%."""
    auroc = compute_auroc(y_true, s_comb)
    max_rec_at_fpr05 = roc_max_recall_at_fpr(y_true, s_comb, 0.05)

    if p_comb is not None and not np.all(np.isnan(p_comb)):
        pred_05 = (p_comb <= 0.05).astype(int)
        tp_05 = np.sum((pred_05 == 1) & (y_true == 1))
        fp_05 = np.sum((pred_05 == 1) & (y_true == 0))
        rec_05 = float(tp_05 / np.sum(y_true == 1)) if np.sum(y_true == 1) > 0 else 0.0
        fpr_05 = float(fp_05 / np.sum(y_true == 0)) if np.sum(y_true == 0) > 0 else 0.0

        pred_10 = (p_comb <= 0.10).astype(int)
        tp_10 = np.sum((pred_10 == 1) & (y_true == 1))
        fp_10 = np.sum((pred_10 == 1) & (y_true == 0))
        rec_10 = float(tp_10 / np.sum(y_true == 1)) if np.sum(y_true == 1) > 0 else 0.0
        fpr_10 = float(fp_10 / np.sum(y_true == 0)) if np.sum(y_true == 0) > 0 else 0.0
    else:
        rec_05, fpr_05, rec_10, fpr_10 = np.nan, np.nan, np.nan, np.nan

    return {
        "auroc": float(auroc),
        "recall_at_p005": rec_05,
        "fpr_at_p005": fpr_05,
        "recall_at_p010": rec_10,
        "fpr_at_p010": fpr_10,
        "max_recall_at_fpr05": float(max_rec_at_fpr05),
    }


def run_benchmark():
    """Main benchmark execution function."""
    print("=" * 70)
    print("E1b: Anomaly Detector Score & p-value Combination Benchmark")
    print("=" * 70)

    # 1. Check calibration files
    cal_status = check_calibration_files()
    print("\nCalibration files status:")
    all_ok = True
    for cat, stat in cal_status.items():
        w_ok = stat["wrn_exists"]
        d_ok = stat["dino_exists"]
        print(f"  [{cat}] WRN50: {'OK' if w_ok else 'MISSING'} ({stat['wrn_path']})")
        print(f"  [{cat}] DINO:  {'OK' if d_ok else 'MISSING'} ({stat['dino_path']})")
        if not (w_ok and d_ok):
            all_ok = False

    if not all_ok:
        print("\nERROR: Required calibration file(s) missing! Recomputing nothing per instructions.")
        sys.exit(1)

    # 2. Load raw scores
    if not SCORES_FILE.exists():
        raise FileNotFoundError(f"Missing raw scores file: {SCORES_FILE}")
    print(f"\nLoading raw scores from {SCORES_FILE}...")
    df = pd.read_csv(SCORES_FILE)
    wrn_df = df[df["method"] == WRN_METHOD].copy()
    dino_df = df[df["method"] == DINO_METHOD].copy()

    merge_cols = ["path", "category", "label", "defect_folder", "defect_type", "split"]
    merged = pd.merge(wrn_df, dino_df, on=merge_cols, suffixes=("_wrn", "_dino"))
    print(f"Merged {len(merged)} test items across {merged['category'].nunique()} categories.")

    # 3. Iterate over categories, splits, and combiners
    results_rows = []

    splits = ["test", "all"]

    for split in splits:
        print(f"\n---------------- SPLIT: {split.upper()} ----------------")
        split_data = merged if split == "all" else merged[merged["split"] == "test"]

        # Track per-split summary across categories to compute macro-average
        combiner_metrics = {m_id: [] for m_id, _ in COMBINER_CONFIGS}

        for cat in CATEGORIES:
            cat_df = split_data[split_data["category"] == cat].copy()
            y_true = cat_df["label"].values
            sw = cat_df["score_wrn"].values
            pw = cat_df["p_value_wrn"].values
            sd = cat_df["score_dino"].values
            pd_val = cat_df["p_value_dino"].values

            cw, cd = load_calibration_data(cat)

            for m_id, m_label in COMBINER_CONFIGS:
                if m_id == "a_wrn50_alone":
                    s_comb, p_comb = combine_wrn50_alone(sw, pw)
                elif m_id == "b_dino_alone":
                    s_comb, p_comb = combine_dino_alone(sd, pd_val)
                elif m_id == "c_zscore_mean":
                    s_comb, p_comb = combine_zscore_mean(sw, sd, cw, cd)
                elif m_id == "d_minmax_mean":
                    s_comb, p_comb = combine_minmax_mean(sw, sd, cw, cd)
                elif m_id == "e_max_zscore":
                    s_comb, p_comb = combine_max_zscore(sw, sd, cw, cd)
                elif m_id == "f_combine_then_calibrate":
                    s_comb, p_comb = combine_then_calibrate(sw, sd, cw, cd)
                elif m_id == "g_bonferroni_min_p":
                    s_comb, p_comb = combine_bonferroni(pw, pd_val)
                elif m_id == "h_vovk_wang_mean_p":
                    s_comb, p_comb = combine_vovk_wang(pw, pd_val)
                elif m_id == "i_cauchy_acat":
                    s_comb, p_comb = combine_cauchy_acat(pw, pd_val)
                elif m_id == "j_fisher":
                    s_comb, p_comb = combine_fisher(pw, pd_val)
                elif m_id == "k_rank_average":
                    s_comb, p_comb = combine_rank_average(sw, sd)
                else:
                    raise ValueError(f"Unknown combiner: {m_id}")

                metrics = evaluate_predictions(y_true, s_comb, p_comb)
                combiner_metrics[m_id].append(metrics)

                row = {
                    "split": split,
                    "category": cat,
                    "method": m_id,
                    "method_label": m_label,
                    **metrics,
                }
                results_rows.append(row)

        # Macro-average across categories for this split
        for m_id, m_label in COMBINER_CONFIGS:
            cat_list = combiner_metrics[m_id]
            def safe_nanmean(lst):
                arr = np.array(lst, dtype=float)
                return float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else np.nan

            mean_auroc = float(np.mean([m["auroc"] for m in cat_list]))
            mean_rec05 = safe_nanmean([m["recall_at_p005"] for m in cat_list])
            mean_fpr05 = safe_nanmean([m["fpr_at_p005"] for m in cat_list])
            mean_rec10 = safe_nanmean([m["recall_at_p010"] for m in cat_list])
            mean_fpr10 = safe_nanmean([m["fpr_at_p010"] for m in cat_list])
            mean_maxrec = float(np.mean([m["max_recall_at_fpr05"] for m in cat_list]))

            results_rows.append({
                "split": split,
                "category": "MEAN",
                "method": m_id,
                "method_label": m_label,
                "auroc": mean_auroc,
                "recall_at_p005": mean_rec05,
                "fpr_at_p005": mean_fpr05,
                "recall_at_p010": mean_rec10,
                "fpr_at_p010": mean_fpr10,
                "max_recall_at_fpr05": mean_maxrec,
            })

            rec05_str = f"{mean_rec05*100:.2f}%" if not np.isnan(mean_rec05) else "N/A"
            fpr05_str = f"{mean_fpr05*100:.2f}%" if not np.isnan(mean_fpr05) else "N/A"
            print(f"[{split.upper()}] {m_label:23s} | AUROC: {mean_auroc:.4f} | Rec@05: {rec05_str:7s} | FPR@05: {fpr05_str:7s} | MaxRec@5%FPR: {mean_maxrec*100:.2f}%")

    # 4. Save to CSV
    out_df = pd.DataFrame(results_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved benchmark results to {OUTPUT_CSV} ({len(out_df)} rows).")


if __name__ == "__main__":
    run_benchmark()
