"""tools/make_torque_mark_targets.py – Sentetik tork işareti (witness mark) hedefleri.

Üst-görünüm vida/gövde görüntüleri üretir: metalik doku, vida kafası dairesi,
boya çizgisi; gevşeme senaryoları için kafa bölgesi döndürülür.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent.parent / "samples" / "torque_mark"

# Paint colour BGR ranges used for drawing
PAINT_COLORS = {
    "kırmızı": (0, 0, 220),     # red in BGR
    "turuncu": (0, 120, 255),    # orange in BGR
    "sarı":    (0, 220, 255),    # yellow in BGR
}

# Rotation scenarios: (rotation_deg, expected_decision)
ROTATIONS = [
    (0,  "GEÇTİ"),
    (2,  "GEÇTİ"),
    (4,  "GEÇTİ"),
    (8,  "KALDI"),
    (15, "KALDI"),
    (30, "KALDI"),
]

W, H = 640, 480
CX, CY = W // 2, H // 2
RADIUS = 80  # screw head radius


def _metallic_background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    """Create metallic-looking housing background."""
    # Base grey with slight gradient
    base = np.full((h, w, 3), 160, dtype=np.uint8)
    grad = np.linspace(140, 180, h).reshape(-1, 1).astype(np.uint8)
    for c in range(3):
        base[:, :, c] = grad
    # Add noise for metallic texture
    noise = rng.normal(0, 8, (h, w)).astype(np.int16)
    for c in range(3):
        base[:, :, c] = np.clip(base[:, :, c].astype(np.int16) + noise, 80, 220).astype(np.uint8)
    return base


def _draw_screw_head(img: np.ndarray, cx: int, cy: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a shaded screw head circle with hex socket."""
    # Darker circle for screw head
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    # Shaded fill
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    shade = np.clip(120 - (dist - r * 0.3) * 0.3, 90, 140).astype(np.uint8)

    for c in range(3):
        img[:, :, c] = np.where(mask > 0, shade, img[:, :, c])

    # Add edge highlight
    cv2.circle(img, (cx, cy), r, (130, 130, 130), 2)

    # Hex socket (small hexagonal hole in center)
    hex_r = r // 3
    pts = []
    for i in range(6):
        angle = math.radians(60 * i + 30)
        px = int(cx + hex_r * math.cos(angle))
        py = int(cy + hex_r * math.sin(angle))
        pts.append([px, py])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(img, [pts], (70, 70, 70))

    return img


def _draw_paint_stripe(img: np.ndarray, cx: int, cy: int, r: int,
                       color_bgr: tuple, stripe_angle_deg: float = 0.0,
                       thickness: int = 8, length_factor: float = 1.8) -> np.ndarray:
    """Draw a straight paint stripe across the screw head and housing."""
    angle_rad = math.radians(stripe_angle_deg)
    total_len = r * length_factor
    x1 = int(cx - total_len * math.cos(angle_rad))
    y1 = int(cy - total_len * math.sin(angle_rad))
    x2 = int(cx + total_len * math.cos(angle_rad))
    y2 = int(cy + total_len * math.sin(angle_rad))
    cv2.line(img, (x1, y1), (x2, y2), color_bgr, thickness)
    return img


def _rotate_head_region(img: np.ndarray, cx: int, cy: int, r: int, angle_deg: float) -> np.ndarray:
    """Rotate the screw head region (circle) by angle_deg about (cx, cy)."""
    if abs(angle_deg) < 0.01:
        return img

    h, w = img.shape[:2]
    result = img.copy()

    # Create a mask for the circular head region
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)

    # Extract the head region with some padding
    pad = 2
    x1 = max(0, cx - r - pad)
    y1 = max(0, cy - r - pad)
    x2 = min(w, cx + r + pad)
    y2 = min(h, cy + r + pad)

    # Create rotation matrix centered on screw
    M = cv2.getRotationMatrix2D((float(cx), float(cy)), -angle_deg, 1.0)

    # Rotate the full image
    rotated_full = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Paste back only the head region
    result[mask > 0] = rotated_full[mask > 0]

    return result


def _add_noise_blur(img: np.ndarray, rng: np.random.Generator,
                    noise_std: float = 3.0, blur_k: int = 3) -> np.ndarray:
    """Add Gaussian noise and slight blur."""
    noise = rng.normal(0, noise_std, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if blur_k > 1:
        img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt: dict[str, dict] = {}
    seed = 42
    rng = np.random.default_rng(seed)

    idx = 0
    for color_name, color_bgr in PAINT_COLORS.items():
        for rot_deg, expected in ROTATIONS:
            # Base image
            bg = _metallic_background(rng, W, H)
            bg = _draw_screw_head(bg, CX, CY, RADIUS, rng)

            # Draw paint stripe at fixed angle (45°)
            stripe_angle = 45.0
            bg = _draw_paint_stripe(bg, CX, CY, RADIUS, color_bgr,
                                    stripe_angle_deg=stripe_angle, thickness=7,
                                    length_factor=1.7)

            # Rotate head region to simulate loosening
            bg = _rotate_head_region(bg, CX, CY, RADIUS, rot_deg)

            # Add slight noise/blur
            bg = _add_noise_blur(bg, rng, noise_std=2.5, blur_k=3)

            fn = f"{color_name}_rot{rot_deg:02d}.png"
            cv2.imwrite(str(OUT / fn), bg)

            gt[fn] = {
                "rotation_deg": rot_deg,
                "paint_color": color_name,
                "expected_decision": expected,
                "screw_cx": CX,
                "screw_cy": CY,
                "screw_radius": RADIUS,
            }
            idx += 1

    # Special case: missing paint on head
    bg = _metallic_background(rng, W, H)
    bg = _draw_screw_head(bg, CX, CY, RADIUS, rng)

    # Only draw paint on the housing side (outside circle)
    color_bgr = PAINT_COLORS["kırmızı"]
    stripe_angle = 45.0
    angle_rad = math.radians(stripe_angle)
    # Just the outer part of the stripe
    outer_start = RADIUS + 5
    outer_end = int(RADIUS * 1.7)
    for sign in [-1, 1]:
        x1 = int(CX + sign * outer_start * math.cos(angle_rad))
        y1 = int(CY + sign * outer_start * math.sin(angle_rad))
        x2 = int(CX + sign * outer_end * math.cos(angle_rad))
        y2 = int(CY + sign * outer_end * math.sin(angle_rad))
        cv2.line(bg, (x1, y1), (x2, y2), color_bgr, 7)

    bg = _add_noise_blur(bg, rng, noise_std=2.5, blur_k=3)
    fn = "kırmızı_missing_head.png"
    cv2.imwrite(str(OUT / fn), bg)
    gt[fn] = {
        "rotation_deg": 0,
        "paint_color": "kırmızı",
        "expected_decision": "KALDI",
        "screw_cx": CX,
        "screw_cy": CY,
        "screw_radius": RADIUS,
        "note": "paint missing on head",
    }

    (OUT / "ground_truth.json").write_text(
        json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{len(gt)} hedef → {OUT}")


if __name__ == "__main__":
    main()
