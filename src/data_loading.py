import pandas as pd
from pathlib import Path


def standard_training_set():
    return pd.read_csv(Path("data/playground-series-s5e8/train.csv"))

def standard_test_set():
    return pd.read_csv(Path("data/playground-series-s5e8/test.csv"))

def full_original_database():
        return pd.read_csv(Path("data/playground-series-s5e8/bank-full.csv"))