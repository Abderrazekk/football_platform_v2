# FILE   : ml/config.py
# ROLE   : central configuration (paths, seed, seasons, labels)
# INPUT  : none
# OUTPUT : none (imported by other ML scripts)
# RUN    : not executed directly

from pathlib import Path


# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"

ML_DIR = BASE_DIR / "ml"
ARTIFACTS = ML_DIR / "artifacts" / "models"
REPORTS = ML_DIR / "reports"
FIGURES = REPORTS / "figures"


# ---------- Reproducibility ----------
SEED = 42


# ---------- Seasons ----------
ALL_SEASONS = [
    "2020-21", "2021-22", "2022-23", "2023-24",
    "2024-25", "2025-26", "2026-27",
]
TRAIN_SEASONS = ["2020-21", "2021-22", "2022-23", "2023-24"]
VAL_SEASON = "2024-25"
TEST_SEASON = "2025-26"
CURRENT_SEASON = "2026-27"


# ---------- Labels (target = Full_Time_Result) ----------
TARGET_COL = "Full_Time_Result"
LABEL_TO_INT = {"H": 0, "D": 1, "A": 2}
INT_TO_LABEL = {0: "H", 1: "D", 2: "A"}
CLASS_NAMES = ["H", "D", "A"]  # order used by scikit-learn


# ---------- Raw column names (as seen in E0 files) ----------
DATE_COL = "Match_Date"
HOME_TEAM_COL = "Home_Team"
AWAY_TEAM_COL = "Away_Team"


# ---------- Helpers ----------
def ensure_dirs():
    """Create output folders if missing (safe to call many times)."""
    for path in [DATA_RAW, DATA_PROCESSED, ARTIFACTS, REPORTS, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)