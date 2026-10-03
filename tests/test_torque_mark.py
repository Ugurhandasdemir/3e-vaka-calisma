"""tests/test_torque_mark.py – Tork İşareti muayene testleri.

Sentetik hedeflerden açısal hata doğruluğu ve karar tutarlılığı kontrol edilir.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from engines.torque_mark import inspect  # noqa: E402

D = ROOT / "samples" / "torque_mark"


def test_torque_mark():
    gt = json.loads((D / "ground_truth.json").read_text(encoding="utf-8"))

    header = (
        f"{'dosya':30s} {'rot':>4s} {'m_ang':>6s} {'err':>6s} "
        f"{'exp':>8s} {'got':>8s} {'ok':>3s}  ms"
    )
    print(f"\n{header}")
    print("-" * len(header))

    max_err = 0.0
    correct = 0
    total = 0
    errors = []

    for fn, g in sorted(gt.items()):
        img = Image.open(D / fn)
        rot_deg = g["rotation_deg"]
        expected = g["expected_decision"]
        color = g["paint_color"]

        res = inspect(img, paint_color=color, tolerance_deg=5.0)
        measured = res["angle_deg"]
        decision = res["decision"]

        total += 1

        # Check decision
        dec_ok = decision == expected
        if dec_ok:
            correct += 1

        # Check angle accuracy (only for rotated cases where measurement is possible)
        angle_err = None
        if measured is not None and rot_deg > 0 and "missing" not in fn:
            angle_err = abs(measured - rot_deg)
            max_err = max(max_err, angle_err)
            if angle_err > 1.5:
                errors.append(f"{fn}: angle err {angle_err:.2f}° > 1.5°")

        if not dec_ok:
            errors.append(f"{fn}: expected {expected}, got {decision}")

        m_str = f"{measured:.1f}" if measured is not None else "  N/A"
        e_str = f"{angle_err:.2f}" if angle_err is not None else "  N/A"
        ok_str = "✓" if dec_ok else "✗"
        print(
            f"{fn:30s} {rot_deg:4d} {m_str:>6s} {e_str:>6s} "
            f"{expected:>8s} {decision:>8s} {ok_str:>3s}  {res['latency_ms']}"
        )

    print(f"\nMax angle error: {max_err:.2f}°")
    print(f"Decisions correct: {correct}/{total}")

    # Assertions
    for e in errors:
        print(f"  FAIL: {e}")
    assert len(errors) == 0, f"{len(errors)} failures: {errors}"
    assert max_err <= 1.5, f"Max angle error {max_err:.2f}° > 1.5°"

    return max_err, correct, total


if __name__ == "__main__":
    max_err, correct, total = test_torque_mark()
    print(f"\nOK — max_err={max_err:.2f}°, {correct}/{total} decisions correct")
