"""tools/make_boresight_targets.py - Synthetic collimator reticle target generator.

Generates realistic collimator reticle targets for visible and thermal channels
with sub-pixel ground truth, optical blur, sensor noise, and vignetting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import config

OUTPUT_DIR = config.SAMPLES_DIR / "boresight"
GROUND_TRUTH_FILE = OUTPUT_DIR / "ground_truth.json"


def render_collimator_target(
    width: int,
    height: int,
    true_x: float,
    true_y: float,
    circle_radius: float = 120.0,
    line_sigma: float = 1.0,
    circle_sigma: float = 1.0,
    is_inverted: bool = False,
    bg_level: float = 25.0,
    fg_amplitude: float = 200.0,
    vignetting_factor: float = 0.20,
    noise_sigma: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Renders a synthetic collimator target image with exact analytic sub-pixel reticle center."""
    rng = np.random.default_rng(seed)

    y_grid, x_grid = np.mgrid[0:height, 0:width].astype(np.float32)

    # 1. Analytic Gaussian profiles for lines and circle
    dist_h = np.abs(y_grid - float(true_y))
    dist_v = np.abs(x_grid - float(true_x))
    dist_c = np.abs(np.sqrt((x_grid - float(true_x))**2 + (y_grid - float(true_y))**2) - float(circle_radius))

    I_h = np.exp(-0.5 * (dist_h / float(line_sigma))**2)
    I_v = np.exp(-0.5 * (dist_v / float(line_sigma))**2)
    I_c = np.exp(-0.5 * (dist_c / float(circle_sigma))**2)

    reticle = np.maximum(np.maximum(I_h, I_v), I_c)

    # 2. Vignetting (radial intensity falloff from sensor center)
    cx, cy = width / 2.0, height / 2.0
    r_norm_sq = ((x_grid - cx) / cx)**2 + ((y_grid - cy) / cy)**2
    vignette = 1.0 - vignetting_factor * (r_norm_sq / 2.0)

    # 3. Base intensity synthesis with polarity
    if is_inverted:
        # Thermal: warm bright background with cooler dark reticle
        base = (bg_level * vignette) - (fg_amplitude * reticle)
    else:
        # Visible: dark collimator housing with illuminated reticle
        base = bg_level + (fg_amplitude * reticle * vignette)

    # 4. Add sensor noise
    noise = rng.normal(0.0, noise_sigma, (height, width))
    final_img = np.clip(base + noise, 0.0, 255.0).astype(np.uint8)

    return final_img


def generate_all_targets() -> list[dict[str, Any]]:
    """Generates visible and thermal target suites and writes ground truth JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    targets_spec = [
        # --- VISIBLE (1280x1024, pitch: 3.45 um, focal: 50.0 mm) ---
        {
            "filename": "vis_offset_0_0.png",
            "channel": "Görünür",
            "width": 1280,
            "height": 1024,
            "true_dx": 0.0,
            "true_dy": 0.0,
            "circle_radius": 140.0,
            "line_sigma": 1.1,
            "is_inverted": False,
            "bg_level": 28.0,
            "fg_amplitude": 210.0,
            "vignetting": 0.20,
            "noise_sigma": 3.0,
            "pixel_pitch_um": 3.45,
            "focal_length_mm": 50.0,
            "tolerance_mrad": 0.5,
            "seed": 101,
        },
        {
            "filename": "vis_offset_3_2_az.png",
            "channel": "Görünür",
            "width": 1280,
            "height": 1024,
            "true_dx": 3.2,
            "true_dy": 0.0,
            "circle_radius": 140.0,
            "line_sigma": 1.1,
            "is_inverted": False,
            "bg_level": 26.0,
            "fg_amplitude": 205.0,
            "vignetting": 0.22,
            "noise_sigma": 3.2,
            "pixel_pitch_um": 3.45,
            "focal_length_mm": 50.0,
            "tolerance_mrad": 0.5,
            "seed": 102,
        },
        {
            "filename": "vis_offset_15_0_el.png",
            "channel": "Görünür",
            "width": 1280,
            "height": 1024,
            "true_dx": 0.0,
            "true_dy": 15.0,
            "circle_radius": 140.0,
            "line_sigma": 1.1,
            "is_inverted": False,
            "bg_level": 25.0,
            "fg_amplitude": 210.0,
            "vignetting": 0.20,
            "noise_sigma": 3.0,
            "pixel_pitch_um": 3.45,
            "focal_length_mm": 50.0,
            "tolerance_mrad": 0.5,
            "seed": 103,
        },
        {
            "filename": "vis_offset_40_0_diag.png",
            "channel": "Görünür",
            "width": 1280,
            "height": 1024,
            "true_dx": 28.284,
            "true_dy": 28.284,
            "circle_radius": 140.0,
            "line_sigma": 1.1,
            "is_inverted": False,
            "bg_level": 25.0,
            "fg_amplitude": 200.0,
            "vignetting": 0.25,
            "noise_sigma": 3.5,
            "pixel_pitch_um": 3.45,
            "focal_length_mm": 50.0,
            "tolerance_mrad": 0.5,
            "seed": 104,
        },
        {
            "filename": "vis_offset_diag_neg.png",
            "channel": "Görünür",
            "width": 1280,
            "height": 1024,
            "true_dx": -5.4,
            "true_dy": 3.8,
            "circle_radius": 140.0,
            "line_sigma": 1.1,
            "is_inverted": False,
            "bg_level": 27.0,
            "fg_amplitude": 205.0,
            "vignetting": 0.20,
            "noise_sigma": 3.0,
            "pixel_pitch_um": 3.45,
            "focal_length_mm": 50.0,
            "tolerance_mrad": 0.5,
            "seed": 105,
        },
        {
            "filename": "vis_offset_subpix.png",
            "channel": "Görünür",
            "width": 1280,
            "height": 1024,
            "true_dx": 1.7,
            "true_dy": -2.4,
            "circle_radius": 140.0,
            "line_sigma": 1.1,
            "is_inverted": False,
            "bg_level": 25.0,
            "fg_amplitude": 205.0,
            "vignetting": 0.20,
            "noise_sigma": 3.0,
            "pixel_pitch_um": 3.45,
            "focal_length_mm": 50.0,
            "tolerance_mrad": 0.5,
            "seed": 106,
        },

        # --- THERMAL (640x512, pitch: 12.0 um, focal: 25.0 mm, inverted polarity) ---
        {
            "filename": "thm_offset_0_0.png",
            "channel": "Termal",
            "width": 640,
            "height": 512,
            "true_dx": 0.0,
            "true_dy": 0.0,
            "circle_radius": 75.0,
            "line_sigma": 1.7,
            "is_inverted": True,
            "bg_level": 215.0,
            "fg_amplitude": 155.0,
            "vignetting": 0.16,
            "noise_sigma": 5.5,
            "pixel_pitch_um": 12.0,
            "focal_length_mm": 25.0,
            "tolerance_mrad": 0.5,
            "seed": 201,
        },
        {
            "filename": "thm_offset_3_2_az.png",
            "channel": "Termal",
            "width": 640,
            "height": 512,
            "true_dx": 3.2,
            "true_dy": 0.0,
            "circle_radius": 75.0,
            "line_sigma": 1.8,
            "is_inverted": True,
            "bg_level": 210.0,
            "fg_amplitude": 150.0,
            "vignetting": 0.18,
            "noise_sigma": 5.8,
            "pixel_pitch_um": 12.0,
            "focal_length_mm": 25.0,
            "tolerance_mrad": 0.5,
            "seed": 202,
        },
        {
            "filename": "thm_offset_subpix_pass.png",
            "channel": "Termal",
            "width": 640,
            "height": 512,
            "true_dx": 0.6,
            "true_dy": -0.5,
            "circle_radius": 75.0,
            "line_sigma": 1.6,
            "is_inverted": True,
            "bg_level": 215.0,
            "fg_amplitude": 155.0,
            "vignetting": 0.15,
            "noise_sigma": 5.0,
            "pixel_pitch_um": 12.0,
            "focal_length_mm": 25.0,
            "tolerance_mrad": 0.5,
            "seed": 203,
        },
        {
            "filename": "thm_offset_15_0_el.png",
            "channel": "Termal",
            "width": 640,
            "height": 512,
            "true_dx": 0.0,
            "true_dy": -15.0,
            "circle_radius": 75.0,
            "line_sigma": 1.8,
            "is_inverted": True,
            "bg_level": 212.0,
            "fg_amplitude": 150.0,
            "vignetting": 0.18,
            "noise_sigma": 6.0,
            "pixel_pitch_um": 12.0,
            "focal_length_mm": 25.0,
            "tolerance_mrad": 0.5,
            "seed": 204,
        },
        {
            "filename": "thm_offset_40_0_diag.png",
            "channel": "Termal",
            "width": 640,
            "height": 512,
            "true_dx": 25.0,
            "true_dy": -30.0,
            "circle_radius": 75.0,
            "line_sigma": 1.8,
            "is_inverted": True,
            "bg_level": 208.0,
            "fg_amplitude": 145.0,
            "vignetting": 0.20,
            "noise_sigma": 6.2,
            "pixel_pitch_um": 12.0,
            "focal_length_mm": 25.0,
            "tolerance_mrad": 0.5,
            "seed": 205,
        },
        {
            "filename": "thm_offset_diag_neg.png",
            "channel": "Termal",
            "width": 640,
            "height": 512,
            "true_dx": -8.5,
            "true_dy": -6.3,
            "circle_radius": 75.0,
            "line_sigma": 1.7,
            "is_inverted": True,
            "bg_level": 210.0,
            "fg_amplitude": 150.0,
            "vignetting": 0.16,
            "noise_sigma": 5.5,
            "pixel_pitch_um": 12.0,
            "focal_length_mm": 25.0,
            "tolerance_mrad": 0.5,
            "seed": 206,
        },
    ]

    records: list[dict[str, Any]] = []

    for spec in targets_spec:
        w = spec["width"]
        h = spec["height"]
        cx = w / 2.0
        cy = h / 2.0
        tx = cx + spec["true_dx"]
        ty = cy + spec["true_dy"]

        img_arr = render_collimator_target(
            width=w,
            height=h,
            true_x=tx,
            true_y=ty,
            circle_radius=spec["circle_radius"],
            line_sigma=spec["line_sigma"],
            is_inverted=spec["is_inverted"],
            bg_level=spec["bg_level"],
            fg_amplitude=spec["fg_amplitude"],
            vignetting_factor=spec["vignetting"],
            noise_sigma=spec["noise_sigma"],
            seed=spec["seed"],
        )

        out_path = OUTPUT_DIR / spec["filename"]
        cv2.imwrite(str(out_path), img_arr)

        true_offset_px = float(np.sqrt(spec["true_dx"]**2 + spec["true_dy"]**2))
        pitch_mm = spec["pixel_pitch_um"] * 1e-3
        focal_mm = spec["focal_length_mm"]
        true_mrad = float(np.arctan((true_offset_px * pitch_mm) / focal_mm) * 1000.0)
        true_pass = bool(true_mrad <= spec["tolerance_mrad"])

        records.append({
            "filename": spec["filename"],
            "path": str(out_path),
            "channel": spec["channel"],
            "width": w,
            "height": h,
            "image_center": [cx, cy],
            "true_center": [round(tx, 4), round(ty, 4)],
            "true_dx": round(spec["true_dx"], 4),
            "true_dy": round(spec["true_dy"], 4),
            "true_offset_px": round(true_offset_px, 4),
            "true_err_mrad": round(true_mrad, 4),
            "true_pass": true_pass,
            "pixel_pitch_um": spec["pixel_pitch_um"],
            "focal_length_mm": spec["focal_length_mm"],
            "tolerance_mrad": spec["tolerance_mrad"],
            "is_inverted": spec["is_inverted"],
        })

    with open(GROUND_TRUTH_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(records)} targets into {OUTPUT_DIR}")
    print(f"Ground truth written to {GROUND_TRUTH_FILE}")
    return records


if __name__ == "__main__":
    generate_all_targets()
