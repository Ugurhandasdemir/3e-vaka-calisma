"""tests/test_boresight.py - Test optical axis boresight measurement against ground truth.

Runs measure on all generated synthetic collimator targets.
Asserts center error vs ground truth < 0.5 px for all targets.
Prints a formatted verification table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image

import config
import engines.boresight as boresight
import tools.make_boresight_targets as target_gen


def test_boresight_accuracy() -> None:
    gt_file = config.SAMPLES_DIR / "boresight" / "ground_truth.json"
    if not gt_file.exists():
        print("Ground truth targets not found, generating...")
        target_gen.generate_all_targets()

    with open(gt_file, "r", encoding="utf-8") as f:
        targets = json.load(f)

    assert len(targets) > 0, "No boresight targets found in ground truth file!"

    print("\n" + "=" * 94)
    print("3E ELEKTRO OPTİK - OPTİK EKSEN (BORESIGHT) DOĞRULAMA VE HASSASİYET TESTİ")
    print("=" * 94)
    header = (
        f"{'Dosya Adı':<25} | {'Gerçek Ofset':<14} | {'Ölçülen (px)':<14} | "
        f"{'Hata (px)':<10} | {'Açı (mrad)':<10} | {'Sonuç':<6}"
    )
    print(header)
    print("-" * 94)

    max_error_px = 0.0

    for item in targets:
        img_path = Path(item["path"])
        if not img_path.exists():
            img_path = config.SAMPLES_DIR / "boresight" / item["filename"]

        assert img_path.exists(), f"Target image not found: {img_path}"

        img = Image.open(img_path)
        res = boresight.measure(
            image=img,
            pixel_pitch_um=item["pixel_pitch_um"],
            focal_length_mm=item["focal_length_mm"],
            tolerance_mrad=item["tolerance_mrad"],
            channel=item["channel"],
        )

        rx, ry = res["reticle_center"]
        tx, ty = item["true_center"]

        error_px = float(((rx - tx)**2 + (ry - ty)**2)**0.5)
        max_error_px = max(max_error_px, error_px)

        true_off_str = f"({item['true_dx']:+.2f}, {item['true_dy']:+.2f})"
        meas_off_str = f"({res['dx_px']:+.2f}, {res['dy_px']:+.2f})"
        row_str = (
            f"{item['filename']:<25} | {true_off_str:<14} | {meas_off_str:<14} | "
            f"{error_px:<10.4f} | {res['err_mrad']:<10.3f} | {res['result']:<6}"
        )
        print(row_str)

        # Hard assertion: center error vs ground truth must be < 0.5 px
        assert error_px < 0.5, (
            f"Boresight center error {error_px:.4f} px exceeds 0.5 px limit for {item['filename']}! "
            f"(True: {tx, ty}, Detected: {rx, ry})"
        )

        # Validate PASS/FAIL agreement with ground truth
        assert res["passed"] == item["true_pass"], (
            f"Pass/fail mismatch for {item['filename']}: expected {item['true_pass']} but got {res['passed']}"
        )

    print("-" * 94)
    print(f"Toplam Test Sayısı: {len(targets)}")
    print(f"Maksimum Merkez Konumlandırma Hatası: {max_error_px:.4f} piksel (Tolerans Eşiği: < 0.5 px)")
    print(f"Ortalama Hata: {sum(((r['true_center'][0] - item['true_center'][0])**2 + (r['true_center'][1] - item['true_center'][1])**2)**0.5 for r, item in zip(targets, targets)):.4f} px")
    print("Durum: TÜM TESTLER BAŞARIYLA GEÇTİ (PASSED)")
    print("=" * 94 + "\n")


def test_compare_channels_and_drift() -> None:
    """Tests inter-channel comparison and drift calculation functions."""
    vis_path = config.SAMPLES_DIR / "boresight" / "vis_offset_3_2_az.png"
    thm_path = config.SAMPLES_DIR / "boresight" / "thm_offset_3_2_az.png"

    assert vis_path.exists() and thm_path.exists(), "Sample files missing for channel comparison!"

    # 1. Compare visible and thermal
    comp = boresight.compare_channels(
        visible_img=vis_path,
        thermal_img=thm_path,
        visible_pitch_um=3.45,
        visible_focal_mm=50.0,
        thermal_pitch_um=12.0,
        thermal_focal_mm=25.0,
        tolerance_mrad=2.0,
    )
    assert "inter_channel_mrad" in comp
    assert "comparison_image" in comp
    assert comp["inter_channel_mrad"] >= 0.0

    # 2. Drift calculation
    before = {"az_mrad": 0.12, "el_mrad": -0.05}
    after = {"az_mrad": 0.22, "el_mrad": 0.08}
    d_res = boresight.drift(before, after, tolerance_mrad=0.5)
    assert "drift_mrad" in d_res
    expected_drift = ((0.22 - 0.12)**2 + (0.08 - (-0.05))**2)**0.5
    assert abs(d_res["drift_mrad"] - expected_drift) < 1e-3
    assert d_res["passed"] is True

    print("Inter-channel comparison and drift calculations: OK")


if __name__ == "__main__":
    test_boresight_accuracy()
    test_compare_channels_and_drift()
    print("All boresight tests passed successfully.")
