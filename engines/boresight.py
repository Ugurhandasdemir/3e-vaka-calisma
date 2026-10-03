"""engines/boresight.py - Optical axis (boresight) measurement engine.

Pure OpenCV/numpy implementation (no AI/neural networks).
Calculates reticle center with sub-pixel accuracy, angular deviation in mrad,
pass/fail status vs tolerance, inter-channel alignment, and vibration drift.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _extract_subpixel_center(
    gray: np.ndarray,
) -> tuple[float, float, float, str]:
    """Finds crosshair center with sub-pixel precision.

    Returns:
        (x_center, y_center, quality_score, method_used)
    """
    h, w = gray.shape

    # 1. Determine polarity (bright-on-dark vs dark-on-bright)
    # Collimator targets usually have lines occupying < 25% of image area
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frac_above = float(np.mean(gray > otsu_val))

    # If bright pixels are majority (> 50%), background is bright (inverted polarity)
    if frac_above >= 0.5:
        norm_gray = (255 - gray).astype(np.uint8)
        polarity_str = "dark-on-bright"
    else:
        norm_gray = gray.copy()
        polarity_str = "bright-on-dark"

    # 2. Suppress vignetting and uneven lighting via White Top-Hat
    # Reticle line thickness is typically 1-5 pixels; kernel of 25x25 preserves lines while flattening background
    tophat_ksize = min(31, max(15, (min(h, w) // 30) | 1))
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_ksize, tophat_ksize))
    tophat = cv2.morphologyEx(norm_gray, cv2.MORPH_TOPHAT, tophat_kernel)

    # 3. Thresholding and morphological cleanup
    _, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )

    # 4. Isolate horizontal and vertical crosshair arms
    # Linear kernel length proportional to image dimensions
    kernel_len = max(25, int(min(w, h) * 0.05))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    h_feat = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, h_kernel)
    v_feat = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, v_kernel)

    # 5. Extract sub-pixel points along horizontal arm
    pts_h: list[list[float]] = []
    cols = np.where(np.any(h_feat > 0, axis=0))[0]
    for x in cols:
        rows = np.where(h_feat[:, x] > 0)[0]
        if len(rows) > 0:
            y_min = max(0, int(rows[0]) - 3)
            y_max = min(h, int(rows[-1]) + 4)
            weights = tophat[y_min:y_max, x].astype(np.float32)
            total_w = float(np.sum(weights))
            if total_w > 0:
                y_coords = np.arange(y_min, y_max, dtype=np.float32)
                y_sub = float(np.sum(y_coords * weights) / total_w)
                pts_h.append([float(x), y_sub])

    # 6. Extract sub-pixel points along vertical arm
    pts_v: list[list[float]] = []
    rows_y = np.where(np.any(v_feat > 0, axis=1))[0]
    for y in rows_y:
        cols_y = np.where(v_feat[y, :] > 0)[0]
        if len(cols_y) > 0:
            x_min = max(0, int(cols_y[0]) - 3)
            x_max = min(w, int(cols_y[-1]) + 4)
            weights = tophat[y, x_min:x_max].astype(np.float32)
            total_w = float(np.sum(weights))
            if total_w > 0:
                x_coords = np.arange(x_min, x_max, dtype=np.float32)
                x_sub = float(np.sum(x_coords * weights) / total_w)
                pts_v.append([x_sub, float(y)])

    # Outlier rejection to remove circle arc points
    arr_h = np.array(pts_h, dtype=np.float32) if len(pts_h) >= 10 else None
    arr_v = np.array(pts_v, dtype=np.float32) if len(pts_v) >= 10 else None

    if arr_h is not None and len(arr_h) >= 10:
        med_y = float(np.median(arr_h[:, 1]))
        inliers_h = arr_h[np.abs(arr_h[:, 1] - med_y) < 5.0]
        if len(inliers_h) >= 8:
            arr_h = inliers_h

    if arr_v is not None and len(arr_v) >= 10:
        med_x = float(np.median(arr_v[:, 0]))
        inliers_v = arr_v[np.abs(arr_v[:, 0] - med_x) < 5.0]
        if len(inliers_v) >= 8:
            arr_v = inliers_v

    # 7. Check if line fitting is possible
    if arr_h is not None and len(arr_h) >= 10 and arr_v is not None and len(arr_v) >= 10:
        # Fit lines with Huber M-estimator
        line_h = cv2.fitLine(arr_h, cv2.DIST_HUBER, 0, 0.01, 0.01)
        line_v = cv2.fitLine(arr_v, cv2.DIST_HUBER, 0, 0.01, 0.01)

        vx_h, vy_h, x0_h, y0_h = [float(v) for v in line_h.flatten()]
        vx_v, vy_v, x0_v, y0_v = [float(v) for v in line_v.flatten()]

        det = -vx_h * vy_v + vy_h * vx_v
        if abs(det) > 1e-4:
            t = (-(x0_v - x0_h) * vy_v + (y0_v - y0_h) * vx_v) / det
            det_x = x0_h + t * vx_h
            det_y = y0_h + t * vy_h

            # Line fit residuals for quality
            # Distance from inlier points to fitted line
            d_h = np.abs(-vy_h * (arr_h[:, 0] - x0_h) + vx_h * (arr_h[:, 1] - y0_h))
            d_v = np.abs(-vy_v * (arr_v[:, 0] - x0_v) + vx_v * (arr_v[:, 1] - y0_v))
            mean_res = float((np.mean(d_h) + np.mean(d_v)) / 2.0)

            # Contrast ratio
            bg_level = float(np.percentile(norm_gray, 20))
            fg_level = float(np.percentile(tophat[cleaned > 0], 80)) if np.any(cleaned > 0) else 100.0
            contrast = max(0.1, min(1.0, (fg_level - bg_level) / 255.0))

            res_score = max(0.0, 1.0 - min(1.0, mean_res / 1.5))
            coverage_score = min(1.0, (len(arr_h) + len(arr_v)) / float(w + h * 0.4))
            quality = round(float(0.5 * res_score + 0.3 * contrast + 0.2 * coverage_score), 3)

            return (float(det_x), float(det_y), quality, f"crosshair_lines ({polarity_str})")

    # 8. Fallback: intensity-weighted centroid of largest blob
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_cnt = max(contours, key=cv2.contourArea)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [largest_cnt], -1, 255, -1)
        weights = tophat.astype(np.float32) * (mask > 0)
        total_w = float(np.sum(weights))
        if total_w > 0:
            y_indices, x_indices = np.indices((h, w), dtype=np.float32)
            cx_sub = float(np.sum(x_indices * weights) / total_w)
            cy_sub = float(np.sum(y_indices * weights) / total_w)
            return (cx_sub, cy_sub, 0.65, f"intensity_centroid_fallback ({polarity_str})")

    # Final fallback: image center
    return (w / 2.0, h / 2.0, 0.2, "center_default_fallback")


def _annotate_measurement(
    image: Image.Image,
    center_img: tuple[float, float],
    center_ret: tuple[float, float],
    dx_px: float,
    dy_px: float,
    offset_px: float,
    az_mrad: float,
    el_mrad: float,
    err_mrad: float,
    tolerance_mrad: float,
    passed: bool,
    quality_score: float,
    channel: str,
) -> Image.Image:
    """Creates a professional annotated visualization with crosshairs and error vector."""
    annotated = image.convert("RGB").copy()
    w, h = annotated.size
    draw = ImageDraw.Draw(annotated)

    cx, cy = center_img
    rx, ry = center_ret

    # Colors
    target_color = (0, 190, 255)       # Cyan for optical reference (sensor center)
    status_color = (16, 185, 129) if passed else (239, 68, 68)  # Green / Red
    vector_color = (245, 158, 11)      # Amber for error vector

    # 1. Draw sensor center reference crosshair (cyan)
    cross_arm = max(20, int(min(w, h) * 0.04))
    draw.line([(cx - cross_arm, cy), (cx + cross_arm, cy)], fill=target_color, width=2)
    draw.line([(cx, cy - cross_arm), (cx, cy + cross_arm)], fill=target_color, width=2)
    draw.ellipse([(cx - 4, cy - 4), (cx + 4, cy + 4)], outline=target_color, width=2)

    # 2. Draw detected reticle center crosshair (status color)
    draw.line([(rx - cross_arm, ry), (rx + cross_arm, ry)], fill=status_color, width=2)
    draw.line([(rx, ry - cross_arm), (rx, ry + cross_arm)], fill=status_color, width=2)
    draw.ellipse([(rx - 8, ry - 8), (rx + 8, ry + 8)], outline=status_color, width=2)

    # 3. Draw error vector arrow from reference center to detected center
    if offset_px > 0.5:
        draw.line([(cx, cy), (rx, ry)], fill=vector_color, width=3)
        # Arrowhead at detected reticle
        angle = np.arctan2(ry - cy, rx - cx)
        arrow_len = 10.0
        p1 = (rx - arrow_len * np.cos(angle - np.pi / 6), ry - arrow_len * np.sin(angle - np.pi / 6))
        p2 = (rx - arrow_len * np.cos(angle + np.pi / 6), ry - arrow_len * np.sin(angle + np.pi / 6))
        draw.polygon([(rx, ry), p1, p2], fill=vector_color)

    # 4. Text Overlay Card in Top-Left
    banner_w = min(420, int(w * 0.45))
    banner_h = 135
    overlay = Image.new("RGBA", (banner_w, banner_h), (17, 24, 39, 215))
    annotated.paste(overlay, (14, 14), overlay)

    badge_text = "GEÇTİ (PASS)" if passed else "KALDI (FAIL)"
    b_color = (16, 185, 129) if passed else (239, 68, 68)

    # Draw text information
    draw.rectangle([14, 14, 14 + banner_w, 14 + 28], fill=(31, 41, 55, 230))
    draw.text((22, 20), f"OPTİK EKSEN (BORESIGHT) - {channel.upper()}", fill=(209, 213, 219))
    draw.text((22, 48), badge_text, fill=b_color)
    draw.text((160, 48), f"Hata: {err_mrad:.3f} mrad / Tol: {tolerance_mrad:.2f} mrad", fill=(243, 244, 246))

    draw.text((22, 74), f"Ofset: dx = {dx_px:+.2f} px, dy = {dy_px:+.2f} px (R = {offset_px:.2f} px)", fill=(209, 213, 219))
    draw.text((22, 94), f"Açısal: Az = {az_mrad:+.3f} mrad, El = {el_mrad:+.3f} mrad", fill=(209, 213, 219))
    draw.text((22, 114), f"Kalite Skoru: %{int(quality_score * 100)}", fill=(156, 163, 175))

    return annotated


def measure(
    image: Image.Image | str | Path,
    pixel_pitch_um: float = 3.45,
    focal_length_mm: float = 50.0,
    tolerance_mrad: float = 0.5,
    channel: str = "Görünür",
) -> dict[str, Any]:
    """Measures collimator reticle boresight error with sub-pixel accuracy.

    Args:
        image: Input PIL Image or path to image file.
        pixel_pitch_um: Sensor pixel pitch in micrometers (default visible: 3.45, thermal: 12.0).
        focal_length_mm: Collimator/lens focal length in millimeters (default visible: 50.0, thermal: 25.0).
        tolerance_mrad: Pass/fail limit in milliradians (default: 0.5 mrad).
        channel: Channel label ('Görünür' or 'Termal').

    Returns:
        dict containing reticle center, image center, offsets in pixels and mrad,
        PASS/FAIL status, quality score, annotated PIL image, and latency.
    """
    t0 = time.perf_counter()

    if isinstance(image, (str, Path)):
        pil_img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_img = image.convert("RGB")
    else:
        pil_img = Image.fromarray(np.uint8(image)).convert("RGB")

    w, h = pil_img.size
    cx = w / 2.0
    cy = h / 2.0

    gray = np.array(pil_img.convert("L"))

    # Extract sub-pixel reticle center
    rx, ry, quality, method = _extract_subpixel_center(gray)

    # Pixel offsets from optical reference (sensor center)
    dx_px = float(rx - cx)
    dy_px = float(ry - cy)
    offset_px = float(np.sqrt(dx_px**2 + dy_px**2))

    # Angular error in milliradians:
    # angle_rad = atan(offset_px * (pixel_pitch_um * 1e-3) / focal_length_mm)
    # angle_mrad = angle_rad * 1000.0
    pitch_mm = float(pixel_pitch_um) * 1e-3
    focal_mm = float(focal_length_mm) if focal_length_mm > 0 else 50.0

    az_mrad = float(np.arctan((dx_px * pitch_mm) / focal_mm) * 1000.0)
    el_mrad = float(np.arctan((dy_px * pitch_mm) / focal_mm) * 1000.0)
    total_offset_mm = offset_px * pitch_mm
    total_mrad = float(np.arctan(total_offset_mm / focal_mm) * 1000.0)

    passed = bool(total_mrad <= float(tolerance_mrad))
    result_str = "PASS" if passed else "FAIL"

    annotated = _annotate_measurement(
        image=pil_img,
        center_img=(cx, cy),
        center_ret=(rx, ry),
        dx_px=dx_px,
        dy_px=dy_px,
        offset_px=offset_px,
        az_mrad=az_mrad,
        el_mrad=el_mrad,
        err_mrad=total_mrad,
        tolerance_mrad=tolerance_mrad,
        passed=passed,
        quality_score=quality,
        channel=channel,
    )

    latency_ms = int(round((time.perf_counter() - t0) * 1000))

    return {
        "reticle_center": (round(rx, 4), round(ry, 4)),
        "image_center": (cx, cy),
        "dx_px": round(dx_px, 4),
        "dy_px": round(dy_px, 4),
        "offset_px": round(offset_px, 4),
        "az_mrad": round(az_mrad, 4),
        "el_mrad": round(el_mrad, 4),
        "err_mrad": round(total_mrad, 4),
        "total_mrad": round(total_mrad, 4),
        "tolerance_mrad": float(tolerance_mrad),
        "result": result_str,
        "passed": passed,
        "quality_score": quality,
        "annotated_image": annotated,
        "method": method,
        "latency_ms": latency_ms,
        "channel": channel,
        "pixel_pitch_um": float(pixel_pitch_um),
        "focal_length_mm": float(focal_length_mm),
    }


def compare_channels(
    visible_img: Image.Image | str | Path | dict[str, Any],
    thermal_img: Image.Image | str | Path | dict[str, Any],
    visible_pitch_um: float = 3.45,
    visible_focal_mm: float = 50.0,
    thermal_pitch_um: float = 12.0,
    thermal_focal_mm: float = 25.0,
    tolerance_mrad: float = 0.5,
) -> dict[str, Any]:
    """Calculates inter-channel boresight alignment between visible and thermal channels.

    Args:
        visible_img: Visible image or prior measure result dict.
        thermal_img: Thermal image or prior measure result dict.
        visible_pitch_um: Visible pixel pitch (default: 3.45 um).
        visible_focal_mm: Visible focal length (default: 50.0 mm).
        thermal_pitch_um: Thermal pixel pitch (default: 12.0 um).
        thermal_focal_mm: Thermal focal length (default: 25.0 mm).
        tolerance_mrad: Inter-channel tolerance in mrad (default: 0.5 mrad).

    Returns:
        dict with inter-channel angular error in mrad, az/el deltas, pass/fail status,
        and side-by-side comparison visualization.
    """
    if isinstance(visible_img, dict) and "az_mrad" in visible_img:
        vis_res = visible_img
    else:
        vis_res = measure(
            visible_img,
            pixel_pitch_um=visible_pitch_um,
            focal_length_mm=visible_focal_mm,
            tolerance_mrad=tolerance_mrad,
            channel="Görünür",
        )

    if isinstance(thermal_img, dict) and "az_mrad" in thermal_img:
        thm_res = thermal_img
    else:
        thm_res = measure(
            thermal_img,
            pixel_pitch_um=thermal_pitch_um,
            focal_length_mm=thermal_focal_mm,
            tolerance_mrad=tolerance_mrad,
            channel="Termal",
        )

    d_az = float(vis_res["az_mrad"] - thm_res["az_mrad"])
    d_el = float(vis_res["el_mrad"] - thm_res["el_mrad"])
    inter_mrad = float(np.sqrt(d_az**2 + d_el**2))

    passed = bool(inter_mrad <= float(tolerance_mrad))

    # Side-by-side comparison image
    vis_anno = vis_res["annotated_image"]
    thm_anno = thm_res["annotated_image"]

    target_h = 480
    v_w = int(vis_anno.width * (target_h / vis_anno.height))
    t_w = int(thm_anno.width * (target_h / thm_anno.height))
    v_scaled = vis_anno.resize((v_w, target_h), Image.Resampling.LANCZOS)
    t_scaled = thm_anno.resize((t_w, target_h), Image.Resampling.LANCZOS)

    combined = Image.new("RGB", (v_w + t_w + 10, target_h + 60), color=(17, 24, 39))
    combined.paste(v_scaled, (0, 60))
    combined.paste(t_scaled, (v_w + 10, 60))

    draw = ImageDraw.Draw(combined)
    status_txt = "KANAL HİZALAMASI: GEÇTİ (PASS)" if passed else "KANAL HİZALAMASI: KALDI (FAIL)"
    status_col = (16, 185, 129) if passed else (239, 68, 68)
    draw.text((20, 16), status_txt, fill=status_col)
    draw.text((360, 16), f"Eksenler Arası Fark: {inter_mrad:.3f} mrad (ΔAz: {d_az:+.3f}, ΔEl: {d_el:+.3f}) | Tol: {tolerance_mrad:.2f} mrad", fill=(243, 244, 246))

    return {
        "inter_channel_mrad": round(inter_mrad, 4),
        "d_az_mrad": round(d_az, 4),
        "d_el_mrad": round(d_el, 4),
        "tolerance_mrad": float(tolerance_mrad),
        "result": "PASS" if passed else "FAIL",
        "passed": passed,
        "visible_result": vis_res,
        "thermal_result": thm_res,
        "comparison_image": combined,
    }


def drift(
    before: dict[str, Any] | float,
    after: dict[str, Any] | float,
    tolerance_mrad: float = 0.5,
) -> dict[str, Any]:
    """Calculates optical axis drift between two stages (e.g. pre/post vibration test).

    Args:
        before: Measurement dict or err_mrad float before stress test.
        after: Measurement dict or err_mrad float after stress test.
        tolerance_mrad: Allowable drift limit in mrad (default: 0.5 mrad).

    Returns:
        dict with total drift in mrad, az/el drift deltas, and pass/fail status.
    """
    if isinstance(before, dict) and isinstance(after, dict):
        d_az = float(after.get("az_mrad", 0.0) - before.get("az_mrad", 0.0))
        d_el = float(after.get("el_mrad", 0.0) - before.get("el_mrad", 0.0))
        drift_val = float(np.sqrt(d_az**2 + d_el**2))
    elif isinstance(before, (int, float)) and isinstance(after, (int, float)):
        drift_val = abs(float(after) - float(before))
        d_az = drift_val
        d_el = 0.0
    else:
        d_az = 0.0
        d_el = 0.0
        drift_val = 0.0

    passed = bool(drift_val <= float(tolerance_mrad))

    return {
        "drift_mrad": round(drift_val, 4),
        "drift_az_mrad": round(d_az, 4),
        "drift_el_mrad": round(d_el, 4),
        "tolerance_mrad": float(tolerance_mrad),
        "result": "PASS" if passed else "FAIL",
        "passed": passed,
    }
