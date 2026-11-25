import json
from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
MAPPING_FILE = Path("ml/artifacts/team_mapping.json")


def test_clean_matches_integrity():
    df = pd.read_csv(PROCESSED_DIR / "clean_matches.csv")

    # 1. No duplicate (date, home_team, away_team)
    dupes = df.duplicated(subset=["Match_Date", "Home_Team", "Away_Team"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate matches"

    # 2. Complete seasons have 380 matches and 20 teams
    complete_seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"]
    for season in complete_seasons:
        s_df = df[df["season"] == season]
        assert len(s_df) == 380, f"{season} has {len(s_df)} matches (expected 380)"
        teams = set(s_df["Home_Team"]) | set(s_df["Away_Team"])
        assert len(teams) == 20, f"{season} has {len(teams)} teams (expected 20)"

    # 3. All teams exist in team_mapping.json values
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        valid_teams = set(json.load(f).values())
    match_teams = set(df["Home_Team"]) | set(df["Away_Team"])
    assert match_teams.issubset(valid_teams), f"Unmapped teams: {match_teams - valid_teams}"


def test_player_data_integrity():
    gw_df = pd.read_csv(PROCESSED_DIR / "player_gw.csv", low_memory=False)

    # 1. No negative player minutes
    assert (gw_df["minutes"] >= 0).all(), "Found negative player minutes in player_gw.csv"

    # 2. Positions are standardized
    valid_positions = {"GK", "DEF", "MID", "FWD"}
    assert set(gw_df["position"].unique()).issubset(valid_positions), "Invalid position found"