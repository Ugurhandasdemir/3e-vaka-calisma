"""Dataset loader and deterministic split for 3E QC evaluation."""
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

import config

CATEGORIES = ["metal_nut", "transistor", "cable"]
DATASET_ROOT = config.ROOT / "eval_data" / "mvtec" / "mvtec" / "images"
RESULTS_DIR = config.ROOT / "eval_results"


def list_category_images(category: str) -> List[Dict]:
    """List all test images for a category with labels and mapped defect types."""
    cat_test_dir = DATASET_ROOT / category / "test"
    if not cat_test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {cat_test_dir}")

    items = []
    for folder in sorted(cat_test_dir.iterdir()):
        if not folder.is_dir():
            continue
        defect_folder = folder.name
        is_defect = (defect_folder != "good")
        label = 1 if is_defect else 0
        defect_type = config.map_class(defect_folder) if is_defect else "Yok"

        for img_path in sorted(folder.glob("*.png")):
            rel_path = img_path.relative_to(config.ROOT).as_posix()
            items.append({
                "path": rel_path,
                "abs_path": img_path.as_posix(),
                "category": category,
                "defect_folder": defect_folder,
                "label": label,
                "defect_type": defect_type,
            })
    return items


def list_all_test_images() -> List[Dict]:
    """List all test images across all 3 categories."""
    all_items = []
    for cat in CATEGORIES:
        all_items.extend(list_category_images(cat))
    return all_items


def create_split(seed: int = 0) -> Tuple[List[Dict], List[Dict], Dict]:
    """Deterministic 50% tune / 50% test split per category stratified by good/defect."""
    rng = np.random.RandomState(seed)
    tune_items = []
    test_items = []
    summary = {}

    for cat in CATEGORIES:
        items = list_category_images(cat)
        goods = [x for x in items if x["label"] == 0]
        defects = [x for x in items if x["label"] == 1]

        # Deterministic sort before shuffle
        goods = sorted(goods, key=lambda x: x["path"])
        defects = sorted(defects, key=lambda x: x["path"])

        # Permutations
        perm_g = rng.permutation(len(goods))
        perm_d = rng.permutation(len(defects))

        n_tune_g = len(goods) // 2
        n_tune_d = len(defects) // 2

        cat_tune_g = [goods[i] for i in perm_g[:n_tune_g]]
        cat_test_g = [goods[i] for i in perm_g[n_tune_g:]]

        cat_tune_d = [defects[i] for i in perm_d[:n_tune_d]]
        cat_test_d = [defects[i] for i in perm_d[n_tune_d:]]

        for item in cat_tune_g + cat_tune_d:
            item_copy = dict(item)
            item_copy["split"] = "tune"
            tune_items.append(item_copy)

        for item in cat_test_g + cat_test_d:
            item_copy = dict(item)
            item_copy["split"] = "test"
            test_items.append(item_copy)

        summary[cat] = {
            "total": len(items),
            "tune": len(cat_tune_g) + len(cat_tune_d),
            "test": len(cat_test_g) + len(cat_test_d),
            "tune_good": len(cat_tune_g),
            "tune_defect": len(cat_tune_d),
            "test_good": len(cat_test_g),
            "test_defect": len(cat_test_d),
        }

    return tune_items, test_items, summary


def save_split(output_file: Path = RESULTS_DIR / "split.json", seed: int = 0) -> Dict:
    """Generate and save split to JSON file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tune_items, test_items, summary = create_split(seed=seed)
    data = {
        "seed": seed,
        "summary": summary,
        "tune": tune_items,
        "test": test_items,
    }
    output_file.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


def load_split(split_file: Path = RESULTS_DIR / "split.json") -> Dict:
    """Load pre-computed split from file, or generate if missing."""
    if not split_file.exists():
        return save_split(split_file)
    return json.loads(split_file.read_text())


if __name__ == "__main__":
    split_data = save_split()
    print("Split generated successfully.")
    for cat, stats in split_data["summary"].items():
        print(f"  {cat}: total={stats['total']}, tune={stats['tune']} (good={stats['tune_good']}, defect={stats['tune_defect']}), "
              f"test={stats['test']} (good={stats['test_good']}, defect={stats['test_defect']})")
    print(f"Total tune: {len(split_data['tune'])}, Total test: {len(split_data['test'])}")
