# FILE   : 01_data_loading.py
# ROLE   : load every E0 season file and concatenate into one table
# INPUT  : data/raw/<season>/E0_Season_YYYYYYYY.csv  (any folder layout works)
# OUTPUT : data/processed/all_matches_raw.csv
# RUN    : python ml/match_model/01_data_loading.py

import sys
from pathlib import Path

# Make "import config" work when the script is run from the project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import config


def season_from_filename(path):
    """'E0_Season_20202021.csv' -> '2020-21'."""
    code = path.stem.replace("E0_Season_", "")   # "20202021"
    return f"{code[:4]}-{code[4:][-2:]}"         # "2020-21"


def load_one_season(season, path):
    """Read one CSV and add a 'season' column."""
    df = pd.read_csv(path)
    df["season"] = season
    return df


def report_columns(frames_info):
    """Print columns missing in some seasons (a warning sign)."""
    all_cols = set()
    for _, cols in frames_info:
        all_cols.update(cols)
    print(f"\nTotal distinct columns across seasons: {len(all_cols)}")
    for season, cols in frames_info:
        missing = all_cols - set(cols)
        if missing:
            sample = sorted(missing)[:3]
            print(f"  {season}: {len(missing)} missing column(s), e.g. {sample}")


def main():
    config.ensure_dirs()

    files = sorted(config.DATA_RAW.rglob("E0_Season_*.csv"))
    if not files:
        print(f"No E0 files found under {config.DATA_RAW}.")
        print("Expected filename pattern: E0_Season_YYYYYYYY.csv")
        return

    frames = []
    frames_info = []
    print(f"Found {len(files)} E0 file(s):")
    for path in files:
        season = season_from_filename(path)
        df = load_one_season(season, path)
        print(f"  {season}: {len(df):>4} rows, {len(df.columns):>3} columns   ({path.name})")
        frames.append(df)
        frames_info.append((season, df.columns))

    report_columns(frames_info)

    all_matches = pd.concat(frames, ignore_index=True, sort=False)
    out_path = config.DATA_PROCESSED / "all_matches_raw.csv"
    all_matches.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Total: {len(all_matches)} rows, {len(all_matches.columns)} columns")


if __name__ == "__main__":
    main()