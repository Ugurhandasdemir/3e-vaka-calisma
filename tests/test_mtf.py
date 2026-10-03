import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from PIL import Image  # noqa: E402

from engines.mtf import measure_mtf  # noqa: E402

D = ROOT / "samples" / "mtf"


def test_mtf_accuracy():
    gt = json.loads((D / "ground_truth.json").read_text())
    print(f"\n{'dosya':22s} {'true50':>7s} {'meas50':>7s} {'err%':>6s} {'nyqT':>6s} {'nyqM':>6s} {'ang':>5s} ms")
    worst = {"Görünür": 0.0, "Termal": 0.0}
    for fn, g in gt.items():
        r = measure_mtf(Image.open(D / fn), g["pixel_pitch_um"], g["channel"])
        err = abs(r["mtf50_cy_px"] - g["mtf50_true_cy_px"]) / g["mtf50_true_cy_px"]
        worst[g["channel"]] = max(worst[g["channel"]], err)
        print(f"{fn:22s} {g['mtf50_true_cy_px']:7.4f} {r['mtf50_cy_px']:7.4f} {err*100:6.2f} "
              f"{g['mtf_nyquist_true']:6.3f} {r['mtf_nyquist']:6.3f} {r['edge_angle_deg']:5.2f} {r['latency_ms']}")
        assert abs(abs(r["edge_angle_deg"]) - g["angle_deg"]) < 0.5
        assert err < (0.05 if g["channel"] == "Görünür" else 0.12), fn
    print("max rel err:", worst)


def test_spec_passfail():
    r = measure_mtf(Image.open(D / "vis_sigma_0.6.png"), 3.45, "Görünür", spec_mtf50=0.25)
    assert r["passed"] is True
    r = measure_mtf(Image.open(D / "vis_sigma_1.0.png"), 3.45, "Görünür", spec_mtf50=0.25)
    assert r["passed"] is False


if __name__ == "__main__":
    test_mtf_accuracy(); test_spec_passfail(); print("OK")
