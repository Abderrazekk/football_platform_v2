# FILE   : ml/utils.py
# ROLE   : small shared helpers (CSV, JSON, metrics)
# INPUT  : file paths / dataframes
# OUTPUT : dataframes / dicts
# RUN    : not executed directly (imported)

import json
from pathlib import Path

import pandas as pd


def load_csv(path):
    """Read a CSV file into a DataFrame."""
    return pd.read_csv(path)


def save_csv(df, path):
    """Write a DataFrame to CSV, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(data, path):
    """Write a dict to a JSON file, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)