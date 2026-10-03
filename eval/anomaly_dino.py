"""AnomalyDINO-style engine using DINOv2 ViT-S/14 patch features and cosine distance."""
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import config

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_SIZE = 448

_model = None
_model_lock = False


def _get_dinov2_model():
    global _model
    if _model is None:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _model = m
    return _model


def _prep_dino(img: Image.Image) -> torch.Tensor:
    a = np.asarray(img.convert("RGB").resize((_SIZE, _SIZE), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (a - _MEAN) / _STD
    return torch.from_numpy(x.transpose(2, 0, 1)[None].astype(np.float32))


def extract_patch_tokens(img: Image.Image) -> torch.Tensor:
    """Extract L2-normalized patch tokens of shape (1024, 384)."""
    m = _get_dinov2_model()
    x = _prep_dino(img)
    with torch.no_grad():
        out = m.forward_features(x)
        tokens = out["x_norm_patchtokens"][0]  # (1024, 384)
        tokens = F.normalize(tokens, p=2, dim=-1)
    return tokens


CACHE_DIR = config.ROOT / "eval_results" / "cache"


def dino_bank_file(category: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"bank_{category}_dinov2_vits14.npy"


def dino_calib_file(category: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"calib_{category}_dinov2_vits14.npy"


def compute_patch_cosine_dists(q: torch.Tensor, bank: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Compute nearest neighbor cosine distance for each patch in q against bank.
    
    Cosine distance = 1.0 - max(cosine_similarity).
    """
    dists = []
    for i in range(0, q.shape[0], chunk):
        qc = q[i : i + chunk]
        sim = torch.mm(qc, bank.t())  # (chunk, N_bank)
        max_sim = sim.max(dim=1).values
        dists.append(1.0 - max_sim)
    return torch.cat(dists)


class AnomalyDINOEngine:
    def __init__(self, category: str):
        self.category = category
        self.backend = "anomaly-dinov2"
        self.version = "dinov2-vits14"
        self.bank = self._get_or_build_bank()
        self.calib_scores = self._get_or_build_calib()

    def _get_or_build_bank(self) -> torch.Tensor:
        bf = dino_bank_file(self.category)
        if bf.exists():
            arr = np.load(bf)
            return torch.from_numpy(arr).float()

        good_files = sorted((config.SAMPLES_DIR / self.category / "good").glob("*.png"))
        if not good_files:
            raise RuntimeError(f"samples/{self.category}/good is empty")

        tokens = []
        for f in good_files:
            img = Image.open(f).convert("RGB")
            t = extract_patch_tokens(img)
            tokens.append(t)
        bank = torch.cat(tokens, dim=0).contiguous()

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            np.save(bf, bank.cpu().numpy().astype(np.float32))
        except Exception:
            pass
        return bank

    def _get_or_build_calib(self) -> np.ndarray:
        cf = dino_calib_file(self.category)
        if cf.exists():
            return np.load(cf).astype(np.float64).ravel()

        calib_files = sorted((config.SAMPLES_DIR / self.category / "calib").glob("*.png"))
        scores = []
        for f in calib_files:
            img = Image.open(f).convert("RGB")
            res = self.predict(img)
            scores.append(res["score"])
        scores_arr = np.array(scores, dtype=np.float64)

        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            np.save(cf, scores_arr)
        except Exception:
            pass
        return scores_arr

    def predict(self, img: Image.Image) -> Dict:
        t0 = time.time()
        q = extract_patch_tokens(img)
        dists = compute_patch_cosine_dists(q, self.bank)
        score = float(dists.max().item())
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "score": score,
            "backend": self.backend,
            "version": self.version,
            "latency_ms": latency_ms,
        }

    def p_value(self, score: float) -> float:
        n = len(self.calib_scores)
        if n == 0:
            return 1.0
        return float((1 + np.sum(self.calib_scores >= score)) / (n + 1))
