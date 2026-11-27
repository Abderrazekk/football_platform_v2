from pathlib import Path
import joblib
import numpy as np
from django.conf import settings

from django.db.models import Avg, Sum
from apps.players.models import Player, PlayerGameweekStats

MODEL_PATH = Path(settings.BASE_DIR) / "ml" / "artifacts" / "models" / "player_model.joblib"
_MODEL_CACHE = None


def _load_player_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None and MODEL_PATH.exists():
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def build_player_feature_vector(
    player: Player,
    was_home: int = 1,
    opponent_def_strength: float = 1.40,
) -> dict | None:
    """Compute last-3 and last-5 gameweek rolling averages from PlayerGameweekStats."""
    recent_gws = list(
        PlayerGameweekStats.objects.filter(player=player)
        .order_by("-season__name", "-round_num")[:5]
    )
    if not recent_gws:
        return None

    last3 = recent_gws[:3]
    last5 = recent_gws

    def _avg(rows, attr):
        vals = [float(getattr(r, attr, 0) or 0) for r in rows]
        return sum(vals) / len(vals) if vals else 0.0

    pos = (player.position or "").upper().replace("GKP", "GK")
    return {
        "minutes_roll3": _avg(last3, "minutes"),
        "minutes_roll5": _avg(last5, "minutes"),
        "total_points_roll3": _avg(last3, "total_points"),
        "total_points_roll5": _avg(last5, "total_points"),
        "bps_roll3": _avg(last3, "bps"),
        "bps_roll5": _avg(last5, "bps"),
        "goals_scored_roll3": _avg(last3, "goals_scored"),
        "goals_scored_roll5": _avg(last5, "goals_scored"),
        "assists_roll3": _avg(last3, "assists"),
        "assists_roll5": _avg(last5, "assists"),
        "ict_index_roll3": _avg(last3, "ict_index"),
        "ict_index_roll5": _avg(last5, "ict_index"),
        "xP_roll3": _avg(last3, "xp"),
        "xP_roll5": _avg(last5, "xp"),
        "was_home": int(was_home),
        "opponent_defensive_strength": float(opponent_def_strength),
        "pos_GK": 1 if pos == "GK" else 0,
        "pos_DEF": 1 if pos == "DEF" else 0,
        "pos_MID": 1 if pos == "MID" else 0,
        "pos_FWD": 1 if pos == "FWD" else 0,
    }


def predict_player_points(
    player: Player,
    was_home: int = 1,
    opponent_def_strength: float = 1.40,
) -> float | None:
    """Predict expected gameweek points for a single player."""
    bundle = _load_player_model()
    feats = build_player_feature_vector(player, was_home, opponent_def_strength)
    if bundle is None or feats is None:
        return None

    X = np.array([[feats.get(col, 0.0) for col in bundle["feature_cols"]]], dtype=float)
    X_sc = bundle["scaler"].transform(X)
    pred = float(bundle["model"].predict(X_sc)[0])
    return round(max(0.0, pred), 2)


def get_match_stars_and_risks(home_team, away_team, top_n: int = 3) -> dict:
    """Return top expected scorers (match stars) and low-minute/form players at risk."""
    players = Player.objects.filter(team__in=[home_team, away_team]).select_related("team")

    scored, at_risk = [], []
    for p in players:
        was_home = 1 if p.team_id == home_team.pk else 0
        feats = build_player_feature_vector(p, was_home=was_home)
        if not feats:
            continue
        exp_pts = predict_player_points(p, was_home=was_home)
        if exp_pts is None:
            continue

        entry = {
            "player": p,
            "expected_points": exp_pts,
            "avg_minutes_3": round(feats["minutes_roll3"], 1),
            "avg_points_3": round(feats["total_points_roll3"], 2),
        }
        scored.append(entry)

        if feats["minutes_roll3"] < 45.0 or (feats["total_points_roll3"] < 1.5 and feats["minutes_roll5"] >= 45.0):
            at_risk.append(entry)

    scored.sort(key=lambda x: -x["expected_points"])
    at_risk.sort(key=lambda x: x["avg_minutes_3"])

    return {
        "stars": scored[:top_n],
        "at_risk": at_risk[:5],
    }
    
def get_player_season_summary(player: Player) -> dict:
    """Aggregate season-level stats for the player's most recent season."""
    latest_gw = (
        PlayerGameweekStats.objects.filter(player=player)
        .select_related("season")
        .order_by("-season__name", "-round_num")
        .first()
    )
    if not latest_gw:
        return {
            "season": "N/A",
            "matches": 0,
            "minutes": 0,
            "total_points": 0,
            "goals": 0,
            "assists": 0,
            "avg_points": 0.0,
            "avg_bps": 0.0,
            "avg_ict": 0.0,
        }

    qs = PlayerGameweekStats.objects.filter(player=player, season=latest_gw.season)
    agg = qs.aggregate(
        sum_minutes=Sum("minutes"),
        sum_points=Sum("total_points"),
        sum_goals=Sum("goals_scored"),
        sum_assists=Sum("assists"),
        avg_points=Avg("total_points"),
        avg_bps=Avg("bps"),
        avg_ict=Avg("ict_index"),
    )
    return {
        "season": latest_gw.season.name,
        "matches": qs.filter(minutes__gt=0).count(),
        "minutes": int(agg["sum_minutes"] or 0),
        "total_points": int(agg["sum_points"] or 0),
        "goals": int(agg["sum_goals"] or 0),
        "assists": int(agg["sum_assists"] or 0),
        "avg_points": round(float(agg["avg_points"] or 0.0), 2),
        "avg_bps": round(float(agg["avg_bps"] or 0.0), 1),
        "avg_ict": round(float(agg["avg_ict"] or 0.0), 1),
    }


def get_player_form_series(player: Player, limit: int = 10) -> dict:
    """Return chronological labels, points, and minutes for Chart.js."""
    gws = list(
        PlayerGameweekStats.objects.filter(player=player)
        .select_related("season")
        .order_by("-season__name", "-round_num")[:limit]
    )
    gws.reverse()
    return {
        "labels": [f"GW {g.round_num}" for g in gws],
        "points": [int(g.total_points or 0) for g in gws],
        "minutes": [int(g.minutes or 0) for g in gws],
    }


def compare_players(player_a: Player, player_b: Player) -> dict:
    """Build side-by-side season stats + next-fixture expected points."""
    return {
        "player_a": {
            "player": player_a,
            "summary": get_player_season_summary(player_a),
            "expected_points_home": predict_player_points(player_a, was_home=1),
            "expected_points_away": predict_player_points(player_a, was_home=0),
        },
        "player_b": {
            "player": player_b,
            "summary": get_player_season_summary(player_b),
            "expected_points_home": predict_player_points(player_b, was_home=1),
            "expected_points_away": predict_player_points(player_b, was_home=0),
        },
    }


def get_match_stars_and_risks(home_team, away_team, top_n: int = 3) -> dict:
    """
    Return:
      - stars: top `top_n` expected performers across both squads.
      - at_risk: lowest expected points among regular starters (avg_minutes_3 >= 45).
    """
    players = Player.objects.filter(team__in=[home_team, away_team]).select_related("team")

    scored = []
    starters = []

    for p in players:
        was_home = 1 if p.team_id == home_team.pk else 0
        feats = build_player_feature_vector(p, was_home=was_home)
        if not feats:
            continue
        exp_pts = predict_player_points(p, was_home=was_home)
        if exp_pts is None:
            continue

        entry = {
            "player": p,
            "expected_points": exp_pts,
            "avg_minutes_3": round(feats["minutes_roll3"], 1),
            "avg_points_3": round(feats["total_points_roll3"], 2),
        }
        scored.append(entry)
        if feats["minutes_roll3"] >= 45.0:
            starters.append(entry)

    scored.sort(key=lambda x: -x["expected_points"])
    starters.sort(key=lambda x: x["expected_points"])

    return {
        "stars": scored[:top_n],
        "at_risk": starters[:top_n],
    }    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    