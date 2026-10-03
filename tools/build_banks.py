#!/usr/bin/env python3
"""Build PatchCore memory banks and heatmap reference scales for all categories in config.PRODUCT_GROUPS."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

import config
from engines.anomaly import BACKBONE, bank_file, bank_meta_file, _features, _nn_dist
from calib import calib_patchcore_file, Calibrator
from engines.anomaly import AnomalyEngine


def build_bank(category: str):
    print(f"[{category}] Building bank with backbone {BACKBONE}...")
    files = sorted((config.SAMPLES_DIR / category / "good").glob("*.png"))
    if not files:
        raise RuntimeError(f"samples/{category}/good is empty")

    feats = []
    for i in range(0, len(files), 8):
        imgs = [Image.open(p).convert("RGB") for p in files[i:i + 8]]
        f, _ = _features(imgs)
        feats.append(f.reshape(-1, f.shape[-1]))
    allf = torch.cat(feats)
    g = torch.Generator().manual_seed(0)
    k = min(allf.shape[0], max(2000, int(allf.shape[0] * 0.10)))
    idx = torch.randperm(allf.shape[0], generator=g)[:k]
    bank = allf[idx].contiguous()

    # Heatmap reference scale: image-max NN distance of (a few) good images against the coreset
    maxes = []
    for f in feats[:3]:
        n = f.shape[0] // 1024
        d = _nn_dist(f, bank).reshape(n, -1)
        maxes.extend(d.max(dim=1).values.tolist())
    ref = float(np.mean(maxes)) if maxes else 1.0

    bf = bank_file(category)
    mf = bank_meta_file(category)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(bf, bank.cpu().numpy().astype(np.float16))
    mf.write_text(json.dumps({"ref": float(ref), "category": category, "backbone": BACKBONE}, indent=2))
    print(f"[{category}] Saved bank to {bf} (shape: {bank.shape}, float16)")
    print(f"[{category}] Saved heatmap ref scale {ref:.6f} to {mf}")

    # Ensure calibration file exists
    cf = calib_patchcore_file(category)
    if not cf.exists():
        print(f"[{category}] Computing calibration scores...")
        eng = AnomalyEngine(category)
        cal = Calibrator(category, eng)
        print(f"[{category}] Saved calibration scores to {cf} ({len(cal.scores)} samples)")
    else:
        print(f"[{category}] Existing calibration file found: {cf}")


def main():
    categories = sorted(set(config.PRODUCT_GROUPS.values()))
    print(f"Building PatchCore memory banks for categories: {categories}")
    for cat in categories:
        build_bank(cat)
    print("All memory banks built successfully.")


if __name__ == "__main__":
    main()
