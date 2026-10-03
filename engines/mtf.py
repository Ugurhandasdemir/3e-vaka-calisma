"""ISO 12233 egimli kenar (slanted-edge) MTF olcumu.

Akis: kenar tespiti -> ROI -> satir basi alt-piksel kenar konumu (turev centroidi)
-> dogru uydurma -> 4x asiri ornekli ESF -> LSF (Hamming) -> |FFT| = MTF.
"""
from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

OVERSAMPLE = 4


def _to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def _detect_roi(gray: np.ndarray) -> tuple[int, int, int, int]:
    """Baskin egimli kenari bulup etrafinda (x0,y0,x1,y1) ROI dondurur."""
    h, w = gray.shape
    g8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    sm = cv2.GaussianBlur(g8, (0, 0), 2.0)
    edges = cv2.Canny(sm, 20, 60)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=40,
                            minLineLength=max(30, min(h, w) // 8), maxLineGap=6)
    if lines is None:
        raise ValueError("Eğimli kenar bulunamadı")
    best = max(np.asarray(lines).reshape(-1, 4), key=lambda l: np.hypot(l[2] - l[0], l[3] - l[1]))
    x1, y1, x2, y2 = [float(v) for v in best]
    ys, xs = np.nonzero(edges)
    dx, dy = x2 - x1, y2 - y1
    n = np.hypot(dx, dy)
    d = np.abs((xs - x1) * dy - (ys - y1) * dx) / n
    m = d < 4
    if m.sum() < 20:
        xs, ys = np.array([x1, x2]), np.array([y1, y2])
    else:
        xs, ys = xs[m], ys[m]
    pad = 24
    x0 = int(max(0, xs.min() - pad)); x1b = int(min(w, xs.max() + pad + 1))
    y0 = int(max(0, ys.min() - pad)); y1b = int(min(h, ys.max() + pad + 1))
    if x1b - x0 < 48:
        c = (x0 + x1b) // 2; x0, x1b = max(0, c - 32), min(w, c + 32)
    if y1b - y0 < 48:
        c = (y0 + y1b) // 2; y0, y1b = max(0, c - 32), min(h, c + 32)
    return x0, y0, x1b, y1b


def _edge_profile(roi: np.ndarray):
    """Kenar konumu x = slope*y + intercept (dikey kabul; yatayda transpoze)."""
    gx = np.abs(cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)).sum()
    gy = np.abs(cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)).sum()
    transposed = gy > gx
    r = roi.T.copy() if transposed else roi
    h, w = r.shape
    sm_img = cv2.GaussianBlur(r, (0, 0), 0.8)
    locs, rows = [], []
    for y in range(h):
        d = np.diff(sm_img[y])
        a = np.abs(d)
        if a.max() < 1e-6:
            continue
        k = int(a.argmax())
        lo, hi = max(0, k - 6), min(len(d), k + 7)
        seg = np.clip(d[lo:hi] * np.sign(d[k]), 0, None)
        if seg.sum() <= 0:
            continue
        locs.append((seg * (np.arange(lo, hi) + 0.5)).sum() / seg.sum())
        rows.append(y + 0.5)
    locs, rows = np.array(locs), np.array(rows, dtype=np.float64)
    if len(locs) < 10:
        raise ValueError("Yetersiz kenar satırı")
    keep = np.ones(len(locs), bool)
    s = b = 0.0
    for _ in range(3):
        A = np.vstack([rows[keep], np.ones(keep.sum())]).T
        s, b = np.linalg.lstsq(A, locs[keep], rcond=None)[0]
        res = locs - (s * rows + b)
        sd = max(res[keep].std(), 0.05)
        keep = np.abs(res) < 3 * sd
        if keep.sum() < 10:
            break
    return r, transposed, float(s), float(b)


def _mtf_from_roi(r: np.ndarray, slope: float, intercept: float):
    h, w = r.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = ((xx + 0.5) - (slope * (yy + 0.5) + intercept)) / np.sqrt(1 + slope ** 2)
    dist = dist.ravel(); val = r.ravel()
    m = np.abs(dist) < 32
    dist, val = dist[m], val[m]
    bins = np.round(dist * OVERSAMPLE).astype(int)
    bins -= bins.min()
    nb = bins.max() + 1
    cnt = np.bincount(bins, minlength=nb).astype(np.float64)
    sm = np.bincount(bins, weights=val, minlength=nb)
    ok = cnt > 0
    idx = np.arange(nb)
    esf = np.interp(idx, idx[ok], sm[ok] / cnt[ok])
    inverted = esf[:nb // 4].mean() > esf[-nb // 4:].mean()
    if inverted:
        esf = esf[::-1]
    lsf = np.diff(esf)
    if lsf.sum() <= 0:
        raise ValueError("Geçersiz ESF")
    pk = int(np.argmax(lsf))
    half = min(pk, len(lsf) - 1 - pk, 30 * OVERSAMPLE)
    lsf = lsf[pk - half: pk + half + 1]
    lsf_w = lsf * np.hamming(len(lsf))
    nfft = 4096
    spec = np.abs(np.fft.rfft(lsf_w, nfft))
    mtf = spec / spec[0]
    f = np.fft.rfftfreq(nfft, d=1.0 / OVERSAMPLE)  # cycles/pixel
    return f, mtf, esf, lsf, inverted


def _cross(f, mtf, level):
    below = np.nonzero(mtf < level)[0]
    if len(below) == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return 0.0
    f0, f1, m0, m1 = f[i - 1], f[i], mtf[i - 1], mtf[i]
    return float(f0 + (m0 - level) * (f1 - f0) / (m0 - m1))


def _plot_mtf(f, mtf, mtf50, nyq, spec) -> Image.Image:
    W, H, L, R, T, B = 640, 420, 70, 20, 40, 60
    img = np.full((H, W, 3), 255, np.uint8)
    pw, ph = W - L - R, H - T - B
    X = lambda v: int(L + v * pw)
    Y = lambda v: int(T + (1 - min(max(v, 0), 1.05) / 1.05) * ph)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(0, 11, 2):
        v = i / 10
        cv2.line(img, (L, Y(v)), (W - R, Y(v)), (225, 225, 225), 1)
        cv2.putText(img, f"{v:.1f}", (L - 36, Y(v) + 5), font, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    for fv in (0, 0.25, 0.5, 0.75, 1.0):
        cv2.line(img, (X(fv), T), (X(fv), H - B), (235, 235, 235), 1)
        cv2.putText(img, f"{fv:.2f}", (X(fv) - 14, H - B + 18), font, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.rectangle(img, (L, T), (W - R, H - B), (120, 120, 120), 1)
    cv2.line(img, (X(0.5), T), (X(0.5), H - B), (200, 120, 0), 1, cv2.LINE_AA)
    cv2.putText(img, "Nyquist", (X(0.5) + 4, T + 14), font, 0.42, (200, 120, 0), 1, cv2.LINE_AA)
    for lv in (0.5, 0.1):
        cv2.line(img, (L, Y(lv)), (W - R, Y(lv)), (170, 170, 220), 1, cv2.LINE_AA)
    m = f <= 1.0
    pts = np.array([[X(a), Y(b)] for a, b in zip(f[m], mtf[m])], np.int32)
    cv2.polylines(img, [pts], False, (200, 40, 30), 2, cv2.LINE_AA)
    if spec is not None:
        c = (0, 150, 60)
        cv2.line(img, (X(spec), T), (X(spec), H - B), c, 1, cv2.LINE_AA)
        cv2.putText(img, f"spec {spec:.2f}", (X(spec) + 4, H - B - 8), font, 0.42, c, 1, cv2.LINE_AA)
    if not np.isnan(mtf50):
        cv2.circle(img, (X(mtf50), Y(0.5)), 5, (20, 20, 200), -1, cv2.LINE_AA)
    cv2.putText(img, f"MTF50 = {mtf50:.3f} cy/px  |  MTF@Nyq = {nyq:.3f}", (L, 24), font, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(img, "Frekans (cycles/pixel)", (L + pw // 2 - 80, H - 14), font, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.putText(img, "MTF", (12, T + ph // 2), font, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    return Image.fromarray(img)


def measure_mtf(image: Image.Image, pixel_pitch_um: float, channel: str = "Görünür",
                spec_mtf50: float | None = None,
                roi: tuple[int, int, int, int] | None = None) -> dict[str, Any]:
    t0 = time.perf_counter()
    gray = _to_gray(image)
    H, W = gray.shape
    if roi is None:
        roi = _detect_roi(gray)
    x0, y0, x1, y1 = [int(v) for v in roi]
    crop = gray[y0:y1, x0:x1]
    r, transposed, slope, intercept = _edge_profile(crop)
    f, mtf, esf, lsf, inverted = _mtf_from_roi(r, slope, intercept)

    angle = float(np.degrees(np.arctan(slope)))
    warnings: list[str] = []
    if not (2.0 <= abs(angle) <= 10.0):
        warnings.append(f"Kenar açısı {abs(angle):.1f}° — ISO 12233 için 2–10° aralığı önerilir; sonuç güvenilir olmayabilir.")
    n = len(esf)
    contrast = abs(esf[-(n // 8):].mean() - esf[: n // 8].mean())
    cols = r.shape[1]
    k = max(2, cols // 10)
    raw_noise = float(np.mean([r[:, :k].std(), r[:, -k:].std()]))
    snr = float(contrast / max(raw_noise, 1e-6))
    if snr < 20:
        warnings.append(f"Düşük SNR ({snr:.0f}) — gürültü MTF'i bozabilir; kontrastı artırın veya kareleri ortalayın.")
    if contrast < 20:
        warnings.append("Kenar kontrastı düşük (<20 gri seviye).")

    mtf50 = _cross(f, mtf, 0.5)
    mtf10 = _cross(f, mtf, 0.1)
    nyq = float(np.interp(0.5, f, mtf))
    lp = lambda cy: float(cy * 1000.0 / pixel_pitch_um) if pixel_pitch_um else float("nan")
    passed = None if spec_mtf50 is None else bool(mtf50 >= spec_mtf50)

    plot = _plot_mtf(f, mtf, mtf50, nyq, spec_mtf50)

    anno = cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    cv2.rectangle(anno, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 0), max(2, W // 300))
    ys = np.array([0.0, r.shape[0]])
    xs = slope * ys + intercept
    if transposed:
        p = [(int(round(x0 + y)), int(round(y0 + x))) for x, y in zip(xs, ys)]
    else:
        p = [(int(round(x0 + x)), int(round(y0 + y))) for x, y in zip(xs, ys)]
    cv2.line(anno, p[0], p[1], (255, 0, 0), max(1, W // 600), cv2.LINE_AA)
    roi_img = Image.fromarray(anno)

    keep = f <= 1.0
    return {
        "channel": channel,
        "pixel_pitch_um": pixel_pitch_um,
        "roi": (x0, y0, x1, y1),
        "edge_angle_deg": angle,
        "edge_orientation": "yatay" if transposed else "dikey",
        "polarity": "açık→koyu" if inverted else "koyu→açık",
        "mtf50_cy_px": mtf50,
        "mtf50_lp_mm": lp(mtf50),
        "mtf10_cy_px": mtf10,
        "mtf10_lp_mm": lp(mtf10),
        "mtf_nyquist": nyq,
        "snr": snr,
        "spec_mtf50": spec_mtf50,
        "passed": passed,
        "warnings": warnings,
        "freq_cy_px": f[keep],
        "mtf_curve": mtf[keep],
        "plot_image": plot,
        "roi_image": roi_img,
        "latency_ms": int(round((time.perf_counter() - t0) * 1000)),
    }
