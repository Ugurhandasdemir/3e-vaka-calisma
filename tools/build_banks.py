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


def build_ensemble(category: str):
    """DINOv2 bank (float16) + combine-then-calibrate statistics (BIRLESTIRME.md section 4)."""
    import engines.anomaly as A
    print(f"[{category}] Building DINOv2 ViT-S/14 bank...")
    files = sorted((config.SAMPLES_DIR / category / "good").glob("*.png"))
    toks = [A.dino_tokens(Image.open(p).convert("RGB")) for p in files]
    bank16 = torch.cat(toks).numpy().astype(np.float16)
    np.save(config.dino_bank_file(category), bank16)
    A._dino_banks.pop(category, None)
    bank = torch.from_numpy(bank16).float()  # use exactly what is stored
    print(f"[{category}] dino bank {bank16.shape} float16")

    cal_files = sorted((config.SAMPLES_DIR / category / "calib").glob("*.png"))
    A._build_bank(category)
    cw, cd = [], []
    for p in cal_files:
        img = Image.open(p).convert("RGB")
        f, _ = A._features([img])
        cw.append(float(A._nn_dist(f[0], A._banks[category]["bank"]).max()))
        cd.append(float(A.dino_nn_dist(A.dino_tokens(img), bank).max()))
    cw, cd = np.array(cw), np.array(cd)
    mu_w, sd_w = float(cw.mean()), float(cw.std()) or 1.0
    mu_d, sd_d = float(cd.mean()), float(cd.std()) or 1.0
    s_cal = 0.5 * ((cw - mu_w) / sd_w + (cd - mu_d) / sd_d)
    np.save(config.ensemble_calib_file(category), s_cal)
    meta = {"category": category, "n": len(cw), "method": "combine-then-calibrate, S=mean(z_wrn,z_dino)",
            "mu_wrn": mu_w, "sigma_wrn": sd_w, "mu_dino": mu_d, "sigma_dino": sd_d,
            "dino_ref": mu_d, "wrn_scores": cw.tolist(), "dino_scores": cd.tolist()}
    config.ensemble_meta_file(category).write_text(json.dumps(meta, indent=2))
    print(f"[{category}] ensemble calib n={len(cw)} mu_w={mu_w:.3f} sd_w={sd_w:.3f} mu_d={mu_d:.4f} sd_d={sd_d:.4f}")


def main():
    categories = sorted(set(config.PRODUCT_GROUPS.values()))
    force = "--force-wrn" in sys.argv
    print(f"Building banks for categories: {categories}")
    for cat in categories:
        if force or not bank_file(cat).exists():
            build_bank(cat)
        build_ensemble(cat)
    print("All banks built successfully.")


if __name__ == "__main__":
    main()
