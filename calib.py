"""Conformal-style calibrator: p = (1 + #{cal >= s}) / (n + 1)."""
import numpy as np

from engines.anomaly import BACKBONE
from PIL import Image

import config

_cache = {}


def calib_patchcore_file(category: str):
    return config.MODELS_DIR / f"calib_{category}_patchcore_{BACKBONE}.npy"


class Calibrator:
    def __init__(self, category, engine):
        self.category = category
        self.engine = engine
        self.scores = self._load()

    def _load(self):
        key = (self.category, self.engine.backend, BACKBONE)
        if key in _cache:
            return _cache[key]
        scores = None
        if self.engine.backend == "efficientad-onnx" and config.calib_file(self.category).exists():
            scores = np.load(config.calib_file(self.category)).astype(np.float64).ravel()
        else:
            f = calib_patchcore_file(self.category)
            if f.exists():
                scores = np.load(f).astype(np.float64).ravel()
            else:
                files = sorted((config.SAMPLES_DIR / self.category / "calib").glob("*.png"))
                vals = [self.engine.predict(Image.open(p).convert("RGB"))["score"] for p in files]
                scores = np.array(vals, dtype=np.float64)
                if len(scores):
                    try:
                        config.MODELS_DIR.mkdir(exist_ok=True)
                        np.save(f, scores)
                    except Exception:
                        pass
        _cache[key] = scores
        return scores

    def p_value(self, score: float) -> float:
        n = len(self.scores)
        return float((1 + np.sum(self.scores >= score)) / (n + 1))
