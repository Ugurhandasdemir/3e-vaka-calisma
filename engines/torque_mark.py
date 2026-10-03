"""engines/torque_mark.py – Tork İşareti (witness-mark / torque-stripe) muayene motoru.

Titreşim testi sonrası vidanın gevşeyip gevşemediğini, vida kafası ile gövde
üzerindeki boya çizgisinin açısal kaymasını ölçerek tespit eder.

Tamamen OpenCV + NumPy; ek model dosyasına ihtiyaç duymaz.
"""
from __future__ import annotations

import math
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── HSV Renk Aralıkları ──────────────────────────────────────────────────
_COLOR_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "kırmızı": [
        (np.array([0, 80, 80]),   np.array([10, 255, 255])),
        (np.array([165, 80, 80]), np.array([180, 255, 255])),
    ],
    "turuncu": [
        (np.array([10, 80, 80]),  np.array([25, 255, 255])),
    ],
    "sarı": [
        (np.array([20, 80, 80]),  np.array([40, 255, 255])),
    ],
}


def _detect_paint_mask(hsv: np.ndarray, paint_color: str) -> np.ndarray:
    """Returns binary mask of paint pixels for the given colour."""
    if paint_color != "auto":
        ranges = _COLOR_RANGES.get(paint_color)
        if not ranges:
            raise ValueError(f"Bilinmeyen boya rengi: {paint_color}")
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)
        return mask

    # auto: try each colour, pick the one with most saturated pixels
    best_mask = None
    best_count = 0
    for name, ranges in _COLOR_RANGES.items():
        m = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, lo, hi)
        cnt = int(np.count_nonzero(m))
        if cnt > best_count:
            best_count = cnt
            best_mask = m
    if best_mask is None or best_count < 10:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)
    return best_mask


def _detect_circle(gray: np.ndarray) -> tuple[int, int, int] | None:
    """Detect the screw-head circle via HoughCircles with fallback."""
    h, w = gray.shape[:2]
    min_r = int(min(h, w) * 0.08)
    max_r = int(min(h, w) * 0.45)

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(h, w) // 3,
        param1=80, param2=40,
        minRadius=min_r, maxRadius=max_r,
    )
    if circles is not None:
        c = circles[0, 0]
        return int(round(c[0])), int(round(c[1])), int(round(c[2]))

    # Fallback: largest roughly circular contour
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity > 0.5 and area > best_area:
            best = cnt
            best_area = area
    if best is not None:
        (cx, cy), radius = cv2.minEnclosingCircle(best)
        return int(round(cx)), int(round(cy)), int(round(radius))
    return None


def _fit_line_pca(pts: np.ndarray) -> tuple[float, float, float, float] | None:
    """Fit a line through 2-D points using PCA.  Returns (vx, vy, x0, y0)."""
    if len(pts) < 5:
        return None
    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    if cov.ndim < 2:
        return None
    eigvals, eigvecs = np.linalg.eigh(cov)
    # principal component = eigenvector with largest eigenvalue
    idx = np.argmax(eigvals)
    vx, vy = eigvecs[:, idx]
    return float(vx), float(vy), float(mean[0]), float(mean[1])


def _angle_between_lines(
    l1: tuple[float, float, float, float],
    l2: tuple[float, float, float, float],
) -> float:
    """Returns absolute angle difference in degrees [0, 90]."""
    v1 = np.array([l1[0], l1[1]], dtype=np.float64)
    v2 = np.array([l2[0], l2[1]], dtype=np.float64)
    cos_a = abs(np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    cos_a = min(cos_a, 1.0)
    return math.degrees(math.acos(cos_a))


def _perpendicular_offset(
    l1: tuple[float, float, float, float],
    l2: tuple[float, float, float, float],
) -> float:
    """Perpendicular distance between the two fitted-line centres (px)."""
    # Project the midpoint of l2 onto the direction perpendicular to l1
    vx, vy = l1[0], l1[1]
    norm = math.hypot(vx, vy) + 1e-12
    # perpendicular direction
    nx, ny = -vy / norm, vx / norm
    dx = l2[2] - l1[2]
    dy = l2[3] - l1[3]
    return abs(nx * dx + ny * dy)


def _draw_annotation(
    pil_img: Image.Image,
    circle: tuple[int, int, int] | None,
    line_head: tuple[float, float, float, float] | None,
    line_housing: tuple[float, float, float, float] | None,
    angle_deg: float | None,
    decision: str,
) -> Image.Image:
    """Draw circle, fitted lines, angle text, and decision on the image."""
    img = pil_img.copy().convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]

    if circle:
        cx, cy, r = circle
        cv2.circle(arr, (cx, cy), r, (0, 255, 255), 2)
        cv2.circle(arr, (cx, cy), 3, (0, 255, 255), -1)

    def _draw_line(line, color, thickness=2):
        if line is None:
            return
        vx, vy, x0, y0 = line
        length = max(w, h)
        x1 = int(x0 - vx * length)
        y1 = int(y0 - vy * length)
        x2 = int(x0 + vx * length)
        y2 = int(y0 + vy * length)
        cv2.line(arr, (x1, y1), (x2, y2), color, thickness)

    _draw_line(line_head, (0, 200, 0), 2)     # green for head
    _draw_line(line_housing, (255, 100, 0), 2) # orange for housing

    # Text
    if angle_deg is not None:
        label = f"Angle: {angle_deg:.1f} deg  [{decision}]"
        cv2.putText(arr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return Image.fromarray(arr)


# ── Public API ────────────────────────────────────────────────────────────
def inspect(
    image: Image.Image,
    paint_color: str = "auto",
    tolerance_deg: float = 5.0,
    tolerance_px: float | None = None,
) -> dict[str, Any]:
    """Inspect a torque stripe (witness mark) on a screw.

    Parameters
    ----------
    image : PIL.Image
        Top-view photograph of the screw and surrounding housing.
    paint_color : str
        'auto' | 'kırmızı' | 'turuncu' | 'sarı'
    tolerance_deg : float
        Maximum allowed angular difference (degrees) between head and housing
        paint segments for a PASS decision.
    tolerance_px : float or None
        Maximum perpendicular offset (pixels).  If None, uses 20 % of the
        screw-head radius (or 15 px fallback).

    Returns
    -------
    dict with keys: decision, angle_deg, offset_px, head_pixels, housing_pixels,
         circle, annotated_image, latency_ms, explanation
    """
    t0 = time.perf_counter()
    rgb = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h_img, w_img = gray.shape[:2]

    # 1. Paint mask
    paint_mask = _detect_paint_mask(hsv, paint_color)
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_OPEN, kernel)
    paint_mask = cv2.morphologyEx(paint_mask, cv2.MORPH_CLOSE, kernel)

    total_paint = int(np.count_nonzero(paint_mask))

    # 2. Circle detection
    circle = _detect_circle(gray)

    # 3. Split paint into head / housing
    if circle:
        cx, cy, r = circle
        yy, xx = np.mgrid[0:h_img, 0:w_img]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        head_region = dist <= r
        # Housing = near circle but outside (within 1.8× radius)
        housing_region = (dist > r) & (dist < r * 1.8)
    else:
        # Fallback: split at image center
        cx, cy = w_img // 2, h_img // 2
        fallback_r = min(h_img, w_img) // 4
        yy, xx = np.mgrid[0:h_img, 0:w_img]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        head_region = dist <= fallback_r
        housing_region = dist > fallback_r

    head_paint = paint_mask & head_region.astype(np.uint8) * 255
    housing_paint = paint_mask & housing_region.astype(np.uint8) * 255

    head_pts = np.column_stack(np.where(head_paint > 0))[:, ::-1]  # (x, y)
    housing_pts = np.column_stack(np.where(housing_paint > 0))[:, ::-1]

    n_head = len(head_pts)
    n_housing = len(housing_pts)

    # Determine default tolerance_px
    if tolerance_px is None:
        if circle:
            tolerance_px = circle[2] * 0.20
        else:
            tolerance_px = 15.0

    # 4. Fit lines
    line_head = _fit_line_pca(head_pts) if n_head >= 5 else None
    line_housing = _fit_line_pca(housing_pts) if n_housing >= 5 else None

    # 5. Compute metrics
    angle_deg: float | None = None
    offset_px: float | None = None

    if line_head and line_housing:
        angle_deg = _angle_between_lines(line_head, line_housing)
        offset_px = _perpendicular_offset(line_head, line_housing)

    # 6. Decision
    MIN_PIXELS = 30  # minimum paint pixels required per side

    if total_paint < MIN_PIXELS:
        decision = "BELİRSİZ"
        explanation = "Yeterli boya pikseli bulunamadı. Görüntü kalitesini kontrol ediniz."
    elif circle is None and total_paint < MIN_PIXELS * 3:
        decision = "BELİRSİZ"
        explanation = "Vida kafası dairesi tespit edilemedi ve boya pikseli yetersiz."
    elif n_head < MIN_PIXELS and n_housing >= MIN_PIXELS:
        decision = "KALDI"
        explanation = "İşaret eksik: Vida kafası üzerinde boya çizgisi tespit edilemedi (kafa tarafı silinmiş veya dönmüş olabilir)."
    elif n_housing < MIN_PIXELS and n_head >= MIN_PIXELS:
        decision = "KALDI"
        explanation = "İşaret eksik: Gövde (housing) tarafında boya çizgisi bulunamadı."
    elif n_head < MIN_PIXELS and n_housing < MIN_PIXELS:
        decision = "BELİRSİZ"
        explanation = "Her iki tarafta da yeterli boya pikseli yok."
    elif angle_deg is not None and offset_px is not None:
        angle_ok = angle_deg <= tolerance_deg
        offset_ok = offset_px <= tolerance_px
        if angle_ok and offset_ok:
            decision = "GEÇTİ"
            explanation = (
                f"Boya çizgisi sürekliliği doğrulandı. Açısal fark {angle_deg:.1f}° "
                f"(tolerans ≤ {tolerance_deg:.1f}°), ofset {offset_px:.1f} px."
            )
        else:
            decision = "KALDI"
            parts = []
            if not angle_ok:
                parts.append(f"açısal fark {angle_deg:.1f}° > tolerans {tolerance_deg:.1f}°")
            if not offset_ok:
                parts.append(f"ofset {offset_px:.1f} px > tolerans {tolerance_px:.1f} px")
            explanation = "Gevşeme şüphesi: " + "; ".join(parts) + "."
    else:
        decision = "BELİRSİZ"
        explanation = "Çizgi uydurma başarısız."

    # Annotated image
    annotated = _draw_annotation(
        image, circle, line_head, line_housing, angle_deg, decision,
    )

    latency_ms = int(round((time.perf_counter() - t0) * 1000))

    return {
        "decision": decision,
        "angle_deg": angle_deg,
        "offset_px": offset_px,
        "head_pixels": n_head,
        "housing_pixels": n_housing,
        "total_paint_pixels": total_paint,
        "circle": circle,
        "tolerance_deg": tolerance_deg,
        "tolerance_px": tolerance_px,
        "annotated_image": annotated,
        "latency_ms": latency_ms,
        "explanation": explanation,
    }
