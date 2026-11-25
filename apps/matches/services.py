from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .models import Prediction

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "ml" / "artifacts" / "models" / "match_model.joblib"
TEAM_STATE_PATH = BASE_DIR / "data" / "processed" / "team_latest_state.csv"
CLEAN_MATCHES_PATH = BASE_DIR / "data" / "processed" / "clean_matches.csv"

MAX_REST_DAYS = 14


class PredictionServiceError(Exception):
    """Raised when the ML model or team data is unavailable or invalid."""


# ---------------------------------------------------------------------------
# Load artifacts ONCE at module import (graceful fallback if missing)
# ---------------------------------------------------------------------------
try:
    _MODEL_BUNDLE = joblib.load(MODEL_PATH)
except Exception:
    _MODEL_BUNDLE = None

try:
    _TEAM_STATE_DF = pd.read_csv(TEAM_STATE_PATH).set_index("team")
    _TEAM_STATE_DF["last_match_date"] = pd.to_datetime(_TEAM_STATE_DF["last_match_date"])
except Exception:
    _TEAM_STATE_DF = pd.DataFrame()

try:
    _MATCHES_DF = pd.read_csv(CLEAN_MATCHES_PATH)
    _MATCHES_DF["Match_Date"] = pd.to_datetime(_MATCHES_DF["Match_Date"])
    _MATCHES_DF = _MATCHES_DF.sort_values("Match_Date")
except Exception:
    _MATCHES_DF = pd.DataFrame()


def _ensure_ready():
    if _MODEL_BUNDLE is None:
        raise PredictionServiceError(
            "The prediction model file (match_model.joblib) is currently unavailable. Please contact an administrator."
        )
    if _TEAM_STATE_DF.empty:
        raise PredictionServiceError(
            "Team statistics data (team_latest_state.csv) is currently unavailable."
        )


def _team_name(team) -> str:
    """Accept either a Team model instance or a plain string."""
    return team.name if hasattr(team, "name") else str(team)


def _compute_h2h_features(home_name: str, away_name: str) -> dict:
    if _MATCHES_DF.empty:
        return {"h2h_home_wr": 0.43, "h2h_draw_r": 0.24, "h2h_away_wr": 0.33, "h2h_gd": 0.0}

    mask = (
        ((_MATCHES_DF["Home_Team"] == home_name) & (_MATCHES_DF["Away_Team"] == away_name))
        | ((_MATCHES_DF["Home_Team"] == away_name) & (_MATCHES_DF["Away_Team"] == home_name))
    )
    h2h_df = _MATCHES_DF.loc[mask]

    if h2h_df.empty:
        return {"h2h_home_wr": 0.43, "h2h_draw_r": 0.24, "h2h_away_wr": 0.33, "h2h_gd": 0.0}

    gds = []
    for _, r in h2h_df.iterrows():
        hg, ag = int(r["Full_Time_Home_Goals"]), int(r["Full_Time_Away_Goals"])
        gds.append(hg - ag if r["Home_Team"] == home_name else ag - hg)

    n = len(gds)
    return {
        "h2h_home_wr": sum(1 for gd in gds if gd > 0) / n,
        "h2h_draw_r": sum(1 for gd in gds if gd == 0) / n,
        "h2h_away_wr": sum(1 for gd in gds if gd < 0) / n,
        "h2h_gd": sum(gds) / n,
    }


def _build_feature_row(home_name: str, away_name: str, match_date=None) -> tuple[pd.DataFrame, dict]:
    _ensure_ready()
    if home_name == away_name:
        raise PredictionServiceError("Home team and away team must be different.")
    if home_name not in _TEAM_STATE_DF.index:
        raise PredictionServiceError(f"No historical stats found for '{home_name}'.")
    if away_name not in _TEAM_STATE_DF.index:
        raise PredictionServiceError(f"No historical stats found for '{away_name}'.")

    h = _TEAM_STATE_DF.loc[home_name]
    a = _TEAM_STATE_DF.loc[away_name]

    ref_date = pd.Timestamp(match_date) if match_date else pd.Timestamp.today().normalize()
    h_rest = max(1, min(int((ref_date - h["last_match_date"]).days), MAX_REST_DAYS))
    a_rest = max(1, min(int((ref_date - a["last_match_date"]).days), MAX_REST_DAYS))

    h_mp = int(h["season_matches_played"])
    a_mp = int(a["season_matches_played"])
    gameweek = max(h_mp, a_mp) + 1
    h2h = _compute_h2h_features(home_name, away_name)

    row = {
        "gameweek": gameweek,
        "home_matches_played": h_mp,
        "away_matches_played": a_mp,
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
        "home_venue_ppg_5": float(h["venue_home_ppg_5"]),
        "home_venue_gf_5": float(h["venue_home_gf_5"]),
        "home_venue_ga_5": float(h["venue_home_ga_5"]),
        "away_venue_ppg_5": float(a["venue_away_ppg_5"]),
        "away_venue_gf_5": float(a["venue_away_gf_5"]),
        "away_venue_ga_5": float(a["venue_away_ga_5"]),
        "home_rest_days": h_rest,
        "away_rest_days": a_rest,
        "rest_days_diff": h_rest - a_rest,
        "ppg_diff_5": float(h["ppg_5"] - a["ppg_5"]),
        "gd_diff_5": float(h["gd_5"] - a["gd_5"]),
        "ppg_diff_10": float(h["ppg_10"] - a["ppg_10"]),
        "gd_diff_10": float(h["gd_10"] - a["gd_10"]),
        "h2h_home_wr": h2h["h2h_home_wr"],
        "h2h_draw_r": h2h["h2h_draw_r"],
        "h2h_away_wr": h2h["h2h_away_wr"],
        "h2h_gd": h2h["h2h_gd"],
    }

    features_list = _MODEL_BUNDLE["features_list"]
    return pd.DataFrame([row])[features_list], row


def _extract_key_factors(home_name: str, away_name: str, row: dict) -> list[str]:
    """Human-readable 'Facteurs clés' based on the model's top feature importances."""
    factors = [
        f"10-match Goal Diff edge: {home_name} ({row['home_gd_10']:+.2f}/g) vs {away_name} ({row['away_gd_10']:+.2f}/g)",
        f"10-match Points/Game: {home_name} ({row['home_ppg_10']:.2f}) vs {away_name} ({row['away_ppg_10']:.2f})",
        f"Recent 5-match Form (PPG): {home_name} ({row['home_ppg_5']:.2f}) vs {away_name} ({row['away_ppg_5']:.2f})",
        f"Head-to-Head Avg Goal Diff ({home_name} perspective): {row['h2h_gd']:+.2f}",
        f"Venue Form (last 5): {home_name} home {row['home_venue_ppg_5']:.2f} PPG vs {away_name} away {row['away_venue_ppg_5']:.2f} PPG",
    ]
    return factors


def predict(home_team, away_team, match_date=None) -> dict:
    """
    Predict match probabilities and key factors.
    Returns dict with home/draw/away probabilities (summing to 100.0),
    predicted_result ('H'/'D'/'A'), model_version, and key_factors.
    """
    home_name = _team_name(home_team)
    away_name = _team_name(away_team)

    X_raw, row = _build_feature_row(home_name, away_name, match_date=match_date)
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
        probs = model_obj.predict_proba(X_in)[0]

    p_home = round(float(probs[2]) * 100, 1)
    p_draw = round(float(probs[1]) * 100, 1)
    p_away = round(100.0 - p_home - p_draw, 1)

    best_idx = int(np.argmax(probs))  # 0=A, 1=D, 2=H
    predicted_result = {0: "A", 1: "D", 2: "H"}[best_idx]

    return {
        "home": p_home,
        "draw": p_draw,
        "away": p_away,
        "predicted_result": predicted_result,
        "model_version": _MODEL_BUNDLE.get("model_name", "stacking"),
        "key_factors": _extract_key_factors(home_name, away_name, row),
    }


def predict_match(home_team, away_team, match_date=None) -> dict[str, float]:
    """Backward-compatible helper returning only probabilities."""
    res = predict(home_team, away_team, match_date=match_date)
    return {"home": res["home"], "draw": res["draw"], "away": res["away"]}


def save_prediction(user, home_team, away_team, prediction_data: dict, match=None) -> Prediction:
    """Persist a Prediction record for the given user."""
    return Prediction.objects.create(
        user=user,
        home_team=home_team,
        away_team=away_team,
        match=match,
        p_home=prediction_data["home"],
        p_draw=prediction_data["draw"],
        p_away=prediction_data["away"],
        predicted_result=prediction_data["predicted_result"],
        model_version=prediction_data.get("model_version", "stacking"),
    )


def h2h_summary(team_a, team_b, limit: int = 5) -> dict:
    """Return head-to-head record and last `limit` meetings between team_a and team_b."""
    name_a, name_b = _team_name(team_a), _team_name(team_b)
    if _MATCHES_DF.empty:
        return {"played": 0, "team_a_wins": 0, "draws": 0, "team_b_wins": 0, "recent": []}

    mask = (
        ((_MATCHES_DF["Home_Team"] == name_a) & (_MATCHES_DF["Away_Team"] == name_b))
        | ((_MATCHES_DF["Home_Team"] == name_b) & (_MATCHES_DF["Away_Team"] == name_a))
    )
    sub = _MATCHES_DF.loc[mask].sort_values("Match_Date", ascending=False)

    a_wins = draws = b_wins = 0
    recent = []
    for _, r in sub.iterrows():
        hg, ag = int(r["Full_Time_Home_Goals"]), int(r["Full_Time_Away_Goals"])
        gd_a = hg - ag if r["Home_Team"] == name_a else ag - hg
        if gd_a > 0:
            a_wins += 1
        elif gd_a == 0:
            draws += 1
        else:
            b_wins += 1

        if len(recent) < limit:
            recent.append({
                "date": r["Match_Date"].strftime("%Y-%m-%d"),
                "home": r["Home_Team"],
                "away": r["Away_Team"],
                "score": f"{hg}-{ag}",
            })

    return {
        "played": len(sub),
        "team_a_wins": a_wins,
        "draws": draws,
        "team_b_wins": b_wins,
        "recent": recent,
    }


def recent_form(team, n: int = 5) -> dict:
    """Return last `n` match results (W/D/L) and points per game for a team."""
    name = _team_name(team)
    if _MATCHES_DF.empty:
        return {"results": [], "ppg": 0.0, "matches": []}

    mask = (_MATCHES_DF["Home_Team"] == name) | (_MATCHES_DF["Away_Team"] == name)
    sub = _MATCHES_DF.loc[mask].sort_values("Match_Date", ascending=False).head(n)

    results = []
    pts = 0
    matches = []
    for _, r in sub.iterrows():
        is_home = r["Home_Team"] == name
        hg, ag = int(r["Full_Time_Home_Goals"]), int(r["Full_Time_Away_Goals"])
        gf, ga = (hg, ag) if is_home else (ag, hg)
        if gf > ga:
            res = "W"
            pts += 3
        elif gf == ga:
            res = "D"
            pts += 1
        else:
            res = "L"
        results.append(res)
        matches.append({
            "date": r["Match_Date"].strftime("%Y-%m-%d"),
            "opponent": r["Away_Team"] if is_home else r["Home_Team"],
            "venue": "H" if is_home else "A",
            "score": f"{gf}-{ga}",
            "result": res,
        })

    return {
        "results": results,
        "ppg": round(pts / len(results), 2) if results else 0.0,
        "matches": matches,
    }