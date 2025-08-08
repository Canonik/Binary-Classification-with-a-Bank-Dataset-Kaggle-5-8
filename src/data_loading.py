import sys
from pathlib import Path
import os

# Add the parent directory to sys.path before importing other modules
parent_dir = str(Path(__file__).resolve().parents[1])
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))

def standard_training_set():
    return pd.read_csv(PROJECT_ROOT / "data/playground-series-s5e8/train.csv").set_index("id")

def standard_test_set():
    return pd.read_csv(PROJECT_ROOT / "data/playground-series-s5e8/test.csv").set_index("id")

def full_original_database():
    full = pd.read_csv(PROJECT_ROOT / "data/playground-series-s5e8/bank-full.csv", sep=";")
    full["y"] = full["y"].replace({"yes": 1, "no": 0}).astype(int)
    return full
