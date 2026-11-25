from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# Resolve project root (apps/matches/services.py -> project root)
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "ml" / "artifacts" / "models" / "match_model.joblib"
TEAM_STATE_PATH = BASE_DIR / "data" / "processed" / "team_latest_state.csv"
CLEAN_MATCHES_PATH = BASE_DIR / "data" / "processed" / "clean_matches.csv"

MAX_REST_DAYS = 14

# ---------------------------------------------------------------------------
# Load artifacts ONCE at module import for sub-second inference
# ---------------------------------------------------------------------------
_MODEL_BUNDLE = joblib.load(MODEL_PATH)
_TEAM_STATE_DF = pd.read_csv(TEAM_STATE_PATH).set_index("team")
_TEAM_STATE_DF["last_match_date"] = pd.to_datetime(_TEAM_STATE_DF["last_match_date"])

# Load historical matches once to compute exact H2H stats
if CLEAN_MATCHES_PATH.exists():
    _MATCHES_DF = pd.read_csv(CLEAN_MATCHES_PATH)
else:
    _MATCHES_DF = pd.DataFrame()


def get_available_teams() -> list[str]:
    """Return sorted list of teams available for prediction."""
    return sorted(_TEAM_STATE_DF.index.tolist())


def _compute_h2h_features(home_team: str, away_team: str) -> dict:
    """Compute historical H2H stats from the perspective of home_team vs away_team."""
    if _MATCHES_DF.empty:
        return {"h2h_home_wr": 0.43, "h2h_draw_r": 0.24, "h2h_away_wr": 0.33, "h2h_gd": 0.0}

    mask = (
        ((_MATCHES_DF["Home_Team"] == home_team) & (_MATCHES_DF["Away_Team"] == away_team))
        | ((_MATCHES_DF["Home_Team"] == away_team) & (_MATCHES_DF["Away_Team"] == home_team))
    )
    h2h_df = _MATCHES_DF.loc[mask]

    if h2h_df.empty:
        return {"h2h_home_wr": 0.43, "h2h_draw_r": 0.24, "h2h_away_wr": 0.33, "h2h_gd": 0.0}

    gds = []
    for _, r in h2h_df.iterrows():
        hg, ag = int(r["Full_Time_Home_Goals"]), int(r["Full_Time_Away_Goals"])
        gds.append(hg - ag if r["Home_Team"] == home_team else ag - hg)

    n = len(gds)
    return {
        "h2h_home_wr": sum(1 for gd in gds if gd > 0) / n,
        "h2h_draw_r": sum(1 for gd in gds if gd == 0) / n,
        "h2h_away_wr": sum(1 for gd in gds if gd < 0) / n,
        "h2h_gd": sum(gds) / n,
    }


def _build_feature_frame(home_team: str, away_team: str, match_date=None) -> pd.DataFrame:
    """Build a single-row DataFrame matching the exact feature order in features_list."""
    if home_team == away_team:
        raise ValueError("Home team and away team must be different.")
    if home_team not in _TEAM_STATE_DF.index:
        raise ValueError(f"Unknown home team: '{home_team}'.")
    if away_team not in _TEAM_STATE_DF.index:
        raise ValueError(f"Unknown away team: '{away_team}'.")

    h = _TEAM_STATE_DF.loc[home_team]
    a = _TEAM_STATE_DF.loc[away_team]

    ref_date = pd.Timestamp(match_date) if match_date else pd.Timestamp.today().normalize()
    h_rest = max(1, min(int((ref_date - h["last_match_date"]).days), MAX_REST_DAYS))
    a_rest = max(1, min(int((ref_date - a["last_match_date"]).days), MAX_REST_DAYS))

    h_mp = int(h["season_matches_played"])
    a_mp = int(a["season_matches_played"])
    gameweek = max(h_mp, a_mp) + 1

    h2h = _compute_h2h_features(home_team, away_team)

    row = {
        "gameweek": gameweek,
        "home_matches_played": h_mp,
        "away_matches_played": a_mp,
        # Rolling 5
        "home_ppg_5": float(h["ppg_5"]),
        "home_gf_5": float(h["gf_5"]),
        "home_ga_5": float(h["ga_5"]),
        "home_gd_5": float(h["gd_5"]),
        "home_wr_5": float(h["wr_5"]),
        "away_ppg_5": float(a["ppg_5"]),
        "away_gf_5": float(a["gf_5"]),
        "away_ga_5": float(a["ga_5"]),
        "away_gd_5": float(a["gd_5"]),
        "away_wr_5": float(a["wr_5"]),
        # Rolling 10
        "home_ppg_10": float(h["ppg_10"]),
        "home_gf_10": float(h["gf_10"]),
        "home_ga_10": float(h["ga_10"]),
        "home_gd_10": float(h["gd_10"]),
        "home_wr_10": float(h["wr_10"]),
        "away_ppg_10": float(a["ppg_10"]),
        "away_gf_10": float(a["gf_10"]),
        "away_ga_10": float(a["ga_10"]),
        "away_gd_10": float(a["gd_10"]),
        "away_wr_10": float(a["wr_10"]),
        # Venue-specific (last 5)
        "home_venue_ppg_5": float(h["venue_home_ppg_5"]),
        "home_venue_gf_5": float(h["venue_home_gf_5"]),
        "home_venue_ga_5": float(h["venue_home_ga_5"]),
        "away_venue_ppg_5": float(a["venue_away_ppg_5"]),
        "away_venue_gf_5": float(a["venue_away_gf_5"]),
        "away_venue_ga_5": float(a["venue_away_ga_5"]),
        # Rest days
        "home_rest_days": h_rest,
        "away_rest_days": a_rest,
        "rest_days_diff": h_rest - a_rest,
        # Form differentials
        "ppg_diff_5": float(h["ppg_5"] - a["ppg_5"]),
        "gd_diff_5": float(h["gd_5"] - a["gd_5"]),
        "ppg_diff_10": float(h["ppg_10"] - a["ppg_10"]),
        "gd_diff_10": float(h["gd_10"] - a["gd_10"]),
        # Head-to-head
        "h2h_home_wr": h2h["h2h_home_wr"],
        "h2h_draw_r": h2h["h2h_draw_r"],
        "h2h_away_wr": h2h["h2h_away_wr"],
        "h2h_gd": h2h["h2h_gd"],
    }

    features_list = _MODEL_BUNDLE["features_list"]
    return pd.DataFrame([row])[features_list]


def predict_match(home_team: str, away_team: str, match_date=None) -> dict[str, float]:
    """
    Predict match outcome probabilities for (home_team, away_team).
    Returns {"home": p_h, "draw": p_d, "away": p_a} in percentages summing to 100.0.
    """
    X_raw = _build_feature_frame(home_team, away_team, match_date=match_date)
    scaler = _MODEL_BUNDLE["scaler"]
    X_scaled = scaler.transform(X_raw)

    model_type = _MODEL_BUNDLE["model_type"]
    model_obj = _MODEL_BUNDLE["model"]

    if model_type == "stacking":
        X_1 = X_scaled if model_obj["model_1_uses_scaled"] else X_raw
        X_2 = X_scaled if model_obj["model_2_uses_scaled"] else X_raw
        meta_X = np.hstack([
            model_obj["model_1"].predict_proba(X_1),
            model_obj["model_2"].predict_proba(X_2),
        ])
        probs = model_obj["meta_model"].predict_proba(meta_X)[0]  # [P_A, P_D, P_H]
    else:
        X_in = X_scaled if _MODEL_BUNDLE.get("uses_scaled", False) else X_raw
        probs = model_obj.predict_proba(X_in)[0]  # [P_A, P_D, P_H]

    # Convert to percentages summing to exactly 100.0
    p_home = round(float(probs[2]) * 100, 1)
    p_draw = round(float(probs[1]) * 100, 1)
    p_away = round(100.0 - p_home - p_draw, 1)

    return {
        "home": p_home,
        "draw": p_draw,
        "away": p_away,
    }