"""engines/sam.py - MobileSAM segmentation engine wrapper (ultralytics).

Lazy singleton on CPU. Provides segment(image, boxes) -> binary masks.
Falls back to SAM('mobile_sam.pt') auto-download if local models/mobile_sam.pt is missing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

import config


class MobileSAM:
    """MobileSAM wrapper running on CPU."""

    def __init__(self) -> None:
        self.model: Any = None
        self.error: str | None = None
        self.version: str = "mobile_sam"

        try:
            from ultralytics import SAM

            local_path = config.MODELS_DIR / "mobile_sam.pt"
            if local_path.exists():
                self.model = SAM(str(local_path))
                self.version = str(local_path.name)
            else:
                # Fallback to auto-download on HF Space or local environment
                self.model = SAM("mobile_sam.pt")
                self.version = "mobile_sam.pt (auto-download)"
        except Exception as e:
            self.error = f"MobileSAM yüklenemedi: {e}"
            self.model = None

    def segment(
        self,
        image: Image.Image | np.ndarray | str | Path,
        boxes: list[list[float]] | list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Segments given bounding boxes on the image using MobileSAM on CPU.

        Args:
            image: PIL Image, numpy array (RGB), or file path.
            boxes: list of boxes [x1, y1, x2, y2] or dicts with xmin, ymin, xmax, ymax.

        Returns:
            dict with:
                - available: bool
                - masks: list[np.ndarray] (binary np.uint8 HxW in original image size, {0, 1})
                - latency_ms: int
                - error: str | None
        """
        t0 = time.perf_counter()

        if self.model is None:
            return {
                "available": False,
                "masks": [],
                "latency_ms": 0,
                "error": self.error or "MobileSAM modeli mevcut değil",
            }

        # Convert image to PIL RGB to establish original dimensions
        try:
            if isinstance(image, Image.Image):
                pil_img = image.convert("RGB")
            elif isinstance(image, np.ndarray):
                pil_img = Image.fromarray(image).convert("RGB")
            else:
                pil_img = Image.open(image).convert("RGB")
        except Exception as e:
            return {
                "available": False,
                "masks": [],
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "error": f"Görsel açılamadı: {e}",
            }

        orig_w, orig_h = pil_img.size

        # Format and validate boxes
        formatted_boxes: list[list[float]] = []
        for b in boxes:
            if isinstance(b, dict):
                x1 = float(b.get("xmin", b.get("x1", 0)))
                y1 = float(b.get("ymin", b.get("y1", 0)))
                x2 = float(b.get("xmax", b.get("x2", 0)))
                y2 = float(b.get("ymax", b.get("y2", 0)))
            elif isinstance(b, (list, tuple)) and len(b) >= 4:
                x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            else:
                continue

            # Ensure proper min/max order and within image bounds
            xmin = max(0.0, min(float(orig_w), min(x1, x2)))
            ymin = max(0.0, min(float(orig_h), min(y1, y2)))
            xmax = max(0.0, min(float(orig_w), max(x1, x2)))
            ymax = max(0.0, min(float(orig_h), max(y1, y2)))

            if xmax > xmin and ymax > ymin:
                formatted_boxes.append([xmin, ymin, xmax, ymax])

        if not formatted_boxes:
            return {
                "available": True,
                "masks": [],
                "latency_ms": int((time.perf_counter() - t0) * 1000),
                "error": None,
            }

        try:
            # Predict using MobileSAM on CPU
            results = self.model.predict(
                pil_img,
                bboxes=formatted_boxes,
                device="cpu",
                verbose=False,
            )

            masks: list[np.ndarray] = []
            if results and results[0].masks is not None:
                r_masks = results[0].masks.data  # torch tensor (N, H, W)
                for i in range(len(r_masks)):
                    m = r_masks[i].cpu().numpy()
                    bin_mask = (m > 0).astype(np.uint8)

                    # Ensure size matches original image dimensions (orig_h, orig_w)
                    if bin_mask.shape != (orig_h, orig_w):
                        bin_mask = cv2.resize(
                            bin_mask,
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    masks.append(bin_mask)

            # Pad with empty masks if model returned fewer masks than requested boxes
            while len(masks) < len(formatted_boxes):
                masks.append(np.zeros((orig_h, orig_w), dtype=np.uint8))

            latency_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "available": True,
                "masks": masks,
                "latency_ms": latency_ms,
                "error": None,
            }
        except Exception as e:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            return {
                "available": False,
                "masks": [],
                "latency_ms": latency_ms,
                "error": f"SAM segmentasyon hatası: {e}",
            }


_sam_instance: MobileSAM | None = None


def get_sam() -> MobileSAM:
    """Returns the lazy singleton MobileSAM instance."""
    global _sam_instance
    if _sam_instance is None:
        _sam_instance = MobileSAM()
    return _sam_instance


def segment(
    image: Image.Image | np.ndarray | str | Path,
    boxes: list[list[float]] | list[dict[str, Any]],
) -> dict[str, Any]:
    """Module-level helper to run MobileSAM segmentation."""
    return get_sam().segment(image, boxes)
