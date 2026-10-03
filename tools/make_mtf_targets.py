"""Sentetik egimli kenar MTF hedefleri (analitik gercekli).

Kenar erf ile analitik uretilir (aliasing yok); Gaussian PSF icin
MTF(f) = exp(-2 pi^2 sigma^2 f^2) gercegi gecerlidir.
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
erf = np.vectorize(math.erf)

OUT = Path(__file__).resolve().parent.parent / "samples" / "mtf"
SIGMAS = [0.6, 1.0, 1.5, 2.5]
ANGLE = 5.0


def edge(w, h, sigma, angle_deg, invert, lo, hi, noise, seed):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    t = math.tan(math.radians(angle_deg))
    cx, cy = w / 2 + 0.31, h / 2
    d = ((xx + 0.5) - (cx + t * (yy + 0.5 - cy))) / math.sqrt(1 + t * t)
    prof = 0.5 * (1 + erf(d / (sigma * math.sqrt(2))))
    if invert:
        prof = 1 - prof
    img = lo + (hi - lo) * prof + rng.normal(0, noise, prof.shape)
    return Image.fromarray(np.clip(np.round(img), 0, 255).astype(np.uint8))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt = {}
    cfg = {
        "vis": (1024, 1024, False, 30, 220, 0.8, 1, "Görünür", 3.45),
        "thm": (640, 512, True, 60, 200, 4.0, 2, "Termal", 12.0),
    }
    for s in SIGMAS:
        m50 = math.sqrt(math.log(2) / (2 * math.pi ** 2)) / s
        nyq = math.exp(-2 * math.pi ** 2 * s * s * 0.25)
        for name, (w, h, inv, lo, hi, nz, seed, ch, pitch) in cfg.items():
            fn = f"{name}_sigma_{s}.png"
            edge(w, h, s, ANGLE, inv, lo, hi, nz, seed + int(s * 10)).save(OUT / fn)
            gt[fn] = {"sigma_px": s, "angle_deg": ANGLE, "channel": ch, "pixel_pitch_um": pitch,
                      "mtf50_true_cy_px": m50, "mtf_nyquist_true": nyq, "inverted": inv}
    (OUT / "ground_truth.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False))
    print(f"{len(gt)} hedef -> {OUT}")


if __name__ == "__main__":
    main()
