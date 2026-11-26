from pathlib import Path
import json
import pandas as pd
from django.conf import settings
from django.db.models import Q

from apps.matches.models import Match, Prediction, Season, Team

METRICS_PATH = Path(settings.BASE_DIR) / "ml" / "artifacts" / "metrics.json"
TEAM_CLUSTERS_PATH = Path(settings.BASE_DIR) / "data" / "processed" / "team_clusters.csv"


def load_metrics() -> dict:
    """Load ml/artifacts/metrics.json safely."""
    if not METRICS_PATH.exists():
        return {}
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def load_team_cluster_stats() -> dict:
    """Return dict keyed by team_name with per-game tactical metrics from team_clusters.csv."""
    if not TEAM_CLUSTERS_PATH.exists():
        return {}
    try:
        df = pd.read_csv(TEAM_CLUSTERS_PATH)
        return df.set_index("team_name").to_dict(orient="index")
    except Exception:
        return {}


def get_season_standings(season_name: str | None = None) -> tuple[Season | None, list[dict]]:
    """Compute league table for a given season (defaults to latest season with played matches)."""
    seasons = Season.objects.all().order_by("-name")
    if season_name:
        season = seasons.filter(name=season_name).first()
    else:
        season = seasons.filter(matches__is_played=True).distinct().first() or seasons.first()

    if not season:
        return None, []

    played = (
        Match.objects.filter(season=season, is_played=True)
        .select_related("home_team", "away_team")
    )

    table = {}

    def _ensure(team: Team):
        if team.pk not in table:
            table[team.pk] = {
                "team": team,
                "played": 0,
                "won": 0,
                "drawn": 0,
                "lost": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "points": 0,
            }
        return table[team.pk]

    for m in played:
        hg = m.home_goals or 0
        ag = m.away_goals or 0
        h = _ensure(m.home_team)
        a = _ensure(m.away_team)

        h["played"] += 1
        a["played"] += 1
        h["gf"] += hg
        h["ga"] += ag
        a["gf"] += ag
        a["ga"] += hg

        if m.result == "H":
            h["won"] += 1
            h["points"] += 3
            a["lost"] += 1
        elif m.result == "A":
            a["won"] += 1
            a["points"] += 3
            h["lost"] += 1
        elif m.result == "D":
            h["drawn"] += 1
            a["drawn"] += 1
            h["points"] += 1
            a["points"] += 1

    rows = []
    for row in table.values():
        row["gd"] = row["gf"] - row["ga"]
        rows.append(row)

    rows.sort(key=lambda r: (-r["points"], -r["gd"], -r["gf"], r["team"].name))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return season, rows


def get_team_trend_data(team: Team, limit: int = 15) -> dict:
    """Return chronological match-by-match points & goals trend for Chart.js."""
    matches = list(
        Match.objects.filter(is_played=True)
        .filter(Q(home_team=team) | Q(away_team=team))
        .select_related("home_team", "away_team", "season")
        .order_by("-match_date")[:limit]
    )
    matches.reverse()

    labels, cum_points, gf_series, ga_series, recent_list = [], [], [], [], []
    running_pts = 0
    total_gf = total_ga = wins = draws = losses = 0

    for m in matches:
        is_home = m.home_team_id == team.pk
        gf = (m.home_goals if is_home else m.away_goals) or 0
        ga = (m.away_goals if is_home else m.home_goals) or 0
        opp = m.away_team.name if is_home else m.home_team.name

        if gf > ga:
            res, pts = "W", 3
            wins += 1
        elif gf == ga:
            res, pts = "D", 1
            draws += 1
        else:
            res, pts = "L", 0
            losses += 1

        running_pts += pts
        total_gf += gf
        total_ga += ga

        labels.append(f"{m.match_date.strftime('%d/%m')} vs {opp[:3].upper()}")
        cum_points.append(running_pts)
        gf_series.append(gf)
        ga_series.append(ga)
        recent_list.append({
            "date": m.match_date,
            "season": m.season.name,
            "opponent": opp,
            "venue": "Dom" if is_home else "Ext",
            "score": f"{gf}-{ga}",
            "result": res,
        })

    recent_list.reverse()
    n = len(matches)
    return {
        "labels": labels,
        "cum_points": cum_points,
        "gf_series": gf_series,
        "ga_series": ga_series,
        "recent_matches": recent_list,
        "summary": {
            "played": n,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "avg_gf": round(total_gf / n, 2) if n else 0.0,
            "avg_ga": round(total_ga / n, 2) if n else 0.0,
            "ppg": round(running_pts / n, 2) if n else 0.0,
        },
    }


def get_home_kpis(user) -> dict:
    """Compute Dashboard KPIs for home.html."""
    matches_played = Match.objects.filter(is_played=True).count()
    metrics = load_metrics()
    live_acc = None
    if metrics.get("live_2026_metrics") and "accuracy" in metrics["live_2026_metrics"]:
        live_acc = round(metrics["live_2026_metrics"]["accuracy"] * 100, 1)

    user_rate = None
    user_total = 0
    if user.is_authenticated:
        user_preds = Prediction.objects.filter(user=user)
        user_total = user_preds.count()
        evaluated = user_preds.exclude(actual_result__isnull=True).exclude(actual_result="")
        eval_count = evaluated.count()
        if eval_count > 0:
            correct = sum(1 for p in evaluated if p.predicted_result == p.actual_result)
            user_rate = round((correct / eval_count) * 100, 1)

    return {
        "matches_played": matches_played,
        "live_2026_accuracy": live_acc,
        "winning_model": metrics.get("winning_model_on_test", "stacking"),
        "user_total_predictions": user_total,
        "user_success_rate": user_rate,
    }