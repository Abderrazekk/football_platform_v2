# FILE   : 02_data_exploration.py
# ROLE   : describe the raw matches and compute the BASELINE numbers
# INPUT  : data/processed/all_matches_raw.csv
# OUTPUT : ml/reports/exploration_summary.txt + 2 figures in ml/reports/figures/
# RUN    : python ml/match_model/02_data_exploration.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to file without opening a window
import matplotlib.pyplot as plt
import config


def load_matches():
    path = config.DATA_PROCESSED / "all_matches_raw.csv"
    return pd.read_csv(path)


def add_total_goals(df):
    """Add a column with the sum of both teams' goals."""
    df = df.copy()
    df["total_goals"] = df["Full_Time_Home_Goals"] + df["Full_Time_Away_Goals"]
    return df


def write_basics(df, f):
    """Write shape, columns, missing values, duplicates to the summary file."""
    print(f"Shape: {df.shape}", file=f)
    print(f"\nColumns ({len(df.columns)}):", file=f)
    for col in df.columns:
        print(f"  - {col}", file=f)
    print("\nMissing values (top 10):", file=f)
    print(df.isna().sum().sort_values(ascending=False).head(10), file=f)
    print(f"\nExact duplicate rows: {df.duplicated().sum()}", file=f)


def write_baseline(df, f):
    """Write H/D/A percentages and goal stats to the summary file."""
    overall = df[config.TARGET_COL].value_counts(normalize=True) * 100
    print("\n--- BASELINE: H/D/A percentages (overall) ---", file=f)
    print(overall.round(2), file=f)

    by_season = (df.groupby("season")[config.TARGET_COL]
                   .value_counts(normalize=True)
                   .unstack() * 100)
    print("\n--- H/D/A percentages per season ---", file=f)
    print(by_season.round(2), file=f)

    home_wins = (df[config.TARGET_COL] == "H").mean() * 100
    print(f"\n'Always predict Home win' baseline accuracy: {home_wins:.2f}%", file=f)

    print(f"\nAvg goals per match: {df['total_goals'].mean():.2f}", file=f)
    print(f"Avg home goals     : {df['Full_Time_Home_Goals'].mean():.2f}", file=f)
    print(f"Avg away goals     : {df['Full_Time_Away_Goals'].mean():.2f}", file=f)
    diff = (df['Full_Time_Home_Goals'] - df['Full_Time_Away_Goals']).mean()
    print(f"Home advantage (home - away goals): {diff:.2f}", file=f)


def plot_outcome_distribution(df, out_path):
    """Bar chart of H/D/A counts."""
    counts = df[config.TARGET_COL].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["#2a9d8f", "#e9c46a", "#e76f51"])
    ax.set_title("Match outcome distribution (H/D/A)")
    ax.set_xlabel("Result")
    ax.set_ylabel("Number of matches")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_goals_per_season(df, out_path):
    """Bar chart of average goals per match, by season."""
    means = df.groupby("season")["total_goals"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    means.plot(kind="bar", ax=ax, color="#264653")
    ax.set_title("Average goals per match, by season")
    ax.set_ylabel("Goals per match")
    ax.set_xticklabels(means.index, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    config.ensure_dirs()
    df = add_total_goals(load_matches())

    config.REPORTS.mkdir(parents=True, exist_ok=True)
    summary_path = config.REPORTS / "exploration_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        write_basics(df, f)
        write_baseline(df, f)
    print(f"Summary written to {summary_path}")

    config.FIGURES.mkdir(parents=True, exist_ok=True)
    plot_outcome_distribution(df, config.FIGURES / "result_distribution.png")
    plot_goals_per_season(df, config.FIGURES / "goals_per_season.png")
    print(f"Figures written to {config.FIGURES}")


if __name__ == "__main__":
    main()