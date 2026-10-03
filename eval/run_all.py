"""Master runner for 3E QC evaluation: Dataset Split -> E1 Benchmark -> E3 Fusion."""
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.dataset import save_split
from eval.e1_anomaly import run_e1
from eval.e3_fusion import run_e3


def main():
    t_start = time.time()
    print("==================================================")
    print(" 3E QC POC: EVALUATION HARNESS (E1 + E3)")
    print("==================================================")

    # 1. Dataset split
    print("\n>>> STEP 1: Dataset Generation & Stratified Split")
    split_data = save_split(seed=0)
    for cat, stats in split_data["summary"].items():
        print(f"  {cat}: total={stats['total']}, tune={stats['tune']} "
              f"(good={stats['tune_good']}, defect={stats['tune_defect']}), "
              f"test={stats['test']} (good={stats['test_good']}, defect={stats['test_defect']})")
    print(f"  Total tune: {len(split_data['tune'])}, Total test: {len(split_data['test'])}")

    # 2. E1 Anomaly Benchmark
    print("\n>>> STEP 2: Experiment E1 (Anomaly Detection Engines)")
    summary_rows, _ = run_e1()

    # 3. E3 Fusion Benchmark & Tuning
    print("\n>>> STEP 3: Experiment E3 (Decision Fusion & Offline Tuning)")
    run_e3()

    print(f"\n==================================================")
    print(f" ALL EVALUATION TASKS COMPLETED in {time.time() - t_start:.1f} s")
    print(f"==================================================")


if __name__ == "__main__":
    main()
