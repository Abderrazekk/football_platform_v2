import json
from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
FEATURES_LIST_FILE = Path("ml/artifacts/features_list.json")

FORBIDDEN_POST_KICKOFF_COLS = {
    "Full_Time_Home_Goals",
    "Full_Time_Away_Goals",
    "Full_Time_Result",
    "FTHG",
    "FTAG",
    "FTR",
    "Home_Shots",
    "Away_Shots",
    "Home_Shots_On_Target",
    "Away_Shots_On_Target",
    "Home_Corners",
    "Away_Corners",
    "Home_Yellow_Cards",
    "Away_Yellow_Cards",
    "Home_Red_Cards",
    "Away_Red_Cards",
    "target",
}


def test_chronological_split_ordering():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv", parse_dates=["Match_Date"])
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv", parse_dates=["Match_Date"])
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv", parse_dates=["Match_Date"])

    assert train_df["Match_Date"].max() < val_df["Match_Date"].min(), (
        "Chronological leakage: train max date >= val min date"
    )
    assert val_df["Match_Date"].max() < test_df["Match_Date"].min(), (
        "Chronological leakage: val max date >= test min date"
    )


def test_no_post_kickoff_features():
    with open(FEATURES_LIST_FILE, "r", encoding="utf-8") as f:
        features = set(json.load(f))

    leaked = features & FORBIDDEN_POST_KICKOFF_COLS
    assert not leaked, f"Post-kickoff columns found in features_list.json: {leaked}"