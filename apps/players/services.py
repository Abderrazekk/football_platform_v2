from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.db.models import Avg, Sum

from apps.matches.models import Match, Team
from apps.players.models import Player, PlayerGameweekStats, PlayerSeasonStats

BASE_DIR = Path(settings.BASE_DIR)
MODEL_PATH = BASE_DIR / "ml" / "artifacts" / "models" / "player_model.joblib"
PLAYER_GW_PATH = BASE_DIR / "data" / "processed" / "player_gw.csv"
CLEAN_MATCHES_PATH = BASE_DIR / "data" / "processed" / "clean_matches.csv"
TEAM_MAPPING_PATH = BASE_DIR / "ml" / "artifacts" / "team_mapping.json"

_MODEL_CACHE = None
_VS_OPP_CACHE = None
_TEAM_DEF_CACHE = None

DEPARTED_PLAYERS_2026_27 = {
    "mohamed salah",
    "heung-min son",
    "son heung-min",
    "kevin de bruyne",
    "trent alexander-arnold",
    "darwin nunez",
    "luis diaz",
    "ederson santana de moraes",
    "ederson",
    "bernardo mota veiga de carvalho e silva",
    "bernardo silva",
    "carlos henrique casimiro",
    "casemiro",
}


def _load_player_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None and MODEL_PATH.exists():
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    return _MODEL_CACHE


def _load_team_mapping() -> dict:
    if TEAM_MAPPING_PATH.exists():
        try:
            with open(TEAM_MAPPING_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _get_team_defensive_strength(team: Team) -> float:
    """Return average goals conceded per match for `team` from clean_matches.csv."""
    global _TEAM_DEF_CACHE
    if _TEAM_DEF_CACHE is None:
        _TEAM_DEF_CACHE = {}
        if CLEAN_MATCHES_PATH.exists():
            try:
                m_df = pd.read_csv(CLEAN_MATCHES_PATH).dropna(
                    subset=["Full_Time_Home_Goals", "Full_Time_Away_Goals"]
                )
                home_ga = m_df.groupby("Home_Team")["Full_Time_Away_Goals"].agg(["sum", "count"])
                away_ga = m_df.groupby("Away_Team")["Full_Time_Home_Goals"].agg(["sum", "count"])
                combined = home_ga.add(away_ga, fill_value=0)
                for t_name, row in combined.iterrows():
                    if row["count"] > 0:
                        _TEAM_DEF_CACHE[str(t_name).strip()] = round(float(row["sum"] / row["count"]), 3)
            except Exception:
                pass
    return _TEAM_DEF_CACHE.get(team.name, 1.40)


def _get_player_vs_opponent_index() -> dict:
    """
    Build and cache a fast lookup {(player_key, opponent_name): stats_dict}
    across ALL seasons in player_gw.csv.
    """
    global _VS_OPP_CACHE
    if _VS_OPP_CACHE is not None:
        return _VS_OPP_CACHE

    _VS_OPP_CACHE = {}
    if not PLAYER_GW_PATH.exists():
        return _VS_OPP_CACHE

    try:
        usecols = [
            "player_key", "season", "team", "was_home", "opponent_team",
            "kickoff_time", "minutes", "total_points", "goals_scored",
            "assists", "clean_sheets", "bps",
        ]
        avail_cols = pd.read_csv(PLAYER_GW_PATH, nrows=0).columns.tolist()
        gw = pd.read_csv(PLAYER_GW_PATH, usecols=[c for c in usecols if c in avail_cols], low_memory=False)

        gw["minutes"] = pd.to_numeric(gw["minutes"], errors="coerce").fillna(0)
        gw = gw[gw["minutes"] > 0].copy()
        if gw.empty:
            return _VS_OPP_CACHE

        team_map = _load_team_mapping()
        gw["player_key"] = gw["player_key"].astype(str).str.strip().str.lower()
        gw["team"] = gw["team"].astype(str).str.strip().replace(team_map)
        gw["was_home_bool"] = gw["was_home"].astype(str).str.lower().isin(["true", "1", "yes"])

        # Resolve opponent_name (handles both string names and numeric FPL IDs)
        opp_str = gw["opponent_team"].astype(str).str.strip()
        is_numeric_opp = opp_str.str.isdigit()
        gw["opponent_name"] = np.where(~is_numeric_opp, opp_str.replace(team_map), np.nan)

        if is_numeric_opp.any() and CLEAN_MATCHES_PATH.exists() and "kickoff_time" in gw.columns:
            m_df = pd.read_csv(CLEAN_MATCHES_PATH)
            m_df["m_date"] = pd.to_datetime(m_df["Match_Date"], errors="coerce").dt.normalize()
            m_df = m_df.dropna(subset=["m_date"])

            # Build lookup for home and away teams with +-1 day tolerance for UTC/BST kickoff shifts
            base_home = m_df[["season", "m_date", "Home_Team", "Away_Team"]].rename(
                columns={"Home_Team": "team", "Away_Team": "resolved_opp"}
            )
            base_home["was_home_bool"] = True

            base_away = m_df[["season", "m_date", "Away_Team", "Home_Team"]].rename(
                columns={"Away_Team": "team", "Home_Team": "resolved_opp"}
            )
            base_away["was_home_bool"] = False

            schedule = pd.concat([base_home, base_away], ignore_index=True)
            expanded_schedules = []
            for delta in (0, -1, 1):
                tmp = schedule.copy()
                tmp["gw_date"] = tmp["m_date"] + pd.to_timedelta(delta, unit="D")
                expanded_schedules.append(tmp[["season", "team", "was_home_bool", "gw_date", "resolved_opp"]])
            sched_lookup = pd.concat(expanded_schedules, ignore_index=True).drop_duplicates(
                subset=["season", "team", "was_home_bool", "gw_date"]
            )

            gw["gw_date"] = pd.to_datetime(gw["kickoff_time"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
            gw = gw.merge(sched_lookup, on=["season", "team", "was_home_bool", "gw_date"], how="left")
            gw["opponent_name"] = gw["opponent_name"].fillna(gw["resolved_opp"])

            # Propagate resolved (season, numeric opponent_team) -> opponent_name for any remaining rows
            id_map_df = (
                gw[is_numeric_opp & gw["opponent_name"].notna()]
                .groupby(["season", "opponent_team"])["opponent_name"]
                .first()
                .reset_index()
                .rename(columns={"opponent_name": "mapped_opp"})
            )
            if not id_map_df.empty:
                gw = gw.merge(id_map_df, on=["season", "opponent_team"], how="left")
                gw["opponent_name"] = gw["opponent_name"].fillna(gw["mapped_opp"])

        gw = gw.dropna(subset=["opponent_name"]).copy()
        for col in ["total_points", "goals_scored", "assists", "clean_sheets", "bps"]:
            if col in gw.columns:
                gw[col] = pd.to_numeric(gw[col], errors="coerce").fillna(0.0)
            else:
                gw[col] = 0.0

        grouped = gw.groupby(["player_key", "opponent_name"], as_index=False).agg(
            matches=("minutes", "count"),
            minutes=("minutes", "sum"),
            total_points=("total_points", "sum"),
            goals=("goals_scored", "sum"),
            assists=("assists", "sum"),
            clean_sheets=("clean_sheets", "sum"),
            bps=("bps", "sum"),
        )

        for row in grouped.itertuples(index=False):
            n = int(row.matches)
            _VS_OPP_CACHE[(row.player_key, row.opponent_name)] = {
                "matches": n,
                "minutes": int(row.minutes),
                "total_points": int(row.total_points),
                "avg_points": round(float(row.total_points) / n, 2),
                "goals": int(row.goals),
                "assists": int(row.assists),
                "clean_sheets": int(row.clean_sheets),
                "avg_bps": round(float(row.bps) / n, 1),
            }
    except Exception:
        _VS_OPP_CACHE = {}

    return _VS_OPP_CACHE


def get_player_vs_opponent_stats(player: Player, opponent_team: Team) -> dict:
    """Return historical stats for `player` against `opponent_team` across all seasons."""
    index = _get_player_vs_opponent_index()
    key = (player.player_key.strip().lower(), opponent_team.name.strip())
    return index.get(
        key,
        {
            "matches": 0,
            "minutes": 0,
            "total_points": 0,
            "avg_points": 0.0,
            "goals": 0,
            "assists": 0,
            "clean_sheets": 0,
            "avg_bps": 0.0,
        },
    )


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


def _score_player_for_match(
    player: Player,
    opponent_team: Team,
    was_home: int,
    opp_def_strength: float,
    career_stats_map: dict,
) -> dict | None:
    """
    Score a single player for a specific fixture by combining:
      1. ML expected gameweek points (player_model.joblib)
      2. All-season career & recent form baseline
      3. Bayesian-smoothed historical H2H performance vs `opponent_team`
      4. Starter minutes reliability factor
    """
    feats = build_player_feature_vector(
        player, was_home=was_home, opponent_def_strength=opp_def_strength
    )
    if not feats:
        return None

    ml_xpts = predict_player_points(
        player, was_home=was_home, opponent_def_strength=opp_def_strength
    )
    if ml_xpts is None:
        ml_xpts = feats["total_points_roll5"]

    career = career_stats_map.get(player.pk, {"avg_pts": ml_xpts, "total_pts": 0, "total_mins": 0})
    base_xpts = 0.70 * float(ml_xpts) + 0.30 * float(career["avg_pts"])

    # Historical performance vs this exact opponent across all seasons
    vs_opp = get_player_vs_opponent_stats(player, opponent_team)
    n_vs = vs_opp["matches"]
    if n_vs > 0:
        # Bayesian smoothing: weight grows as sample size n_vs increases
        vs_weight = n_vs / (n_vs + 4.0)
        blended_xpts = (1.0 - vs_weight) * base_xpts + vs_weight * vs_opp["avg_points"]
    else:
        blended_xpts = base_xpts

    # Starter availability weight based on recent 5-match minutes
    avg_mins = feats["minutes_roll5"]
    if avg_mins >= 60.0:
        mins_factor = 1.0
    elif avg_mins >= 30.0:
        mins_factor = 0.82
    elif avg_mins >= 10.0:
        mins_factor = 0.55
    else:
        mins_factor = 0.30

    match_score = round(max(0.1, blended_xpts * mins_factor), 2)

    # Human-readable highlight badge for UI
    if n_vs > 0 and (vs_opp["goals"] > 0 or vs_opp["assists"] > 0):
        badge = f"vs {opponent_team.name}: {vs_opp['goals']}B, {vs_opp['assists']}PD ({n_vs}m)"
    elif n_vs > 0 and player.position in ("GK", "DEF") and vs_opp["clean_sheets"] > 0:
        badge = f"vs {opponent_team.name}: {vs_opp['clean_sheets']} CS ({n_vs}m)"
    elif n_vs > 0:
        badge = f"vs {opponent_team.name}: {vs_opp['avg_points']} pts/m ({n_vs}m)"
    else:
        badge = f"Forme 5m: {feats['total_points_roll5']:.1f} pts/m"

    return {
        "player": player,
        "team_name": player.team.name if player.team else "",
        "position": (player.position or "MID").upper().replace("GKP", "GK"),
        "match_score": match_score,
        "ml_xpts": round(float(ml_xpts), 2),
        "avg_minutes": round(avg_mins, 0),
        "vs_opp": vs_opp,
        "badge": badge,
    }


def _select_valid_xi(candidates: list[dict]) -> dict:
    """
    Pick the optimal 11-player lineup respecting valid football formation rules:
      1 GK + 3-5 DEF + 2-5 MID + 1-3 FWD = 11 players.
    """
    by_pos = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for c in sorted(candidates, key=lambda x: -x["match_score"]):
        pos = c["position"]
        if pos in by_pos:
            by_pos[pos].append(c)

    # Fallback if a squad lacks a position in test fixtures
    if not by_pos["GK"] or len(by_pos["DEF"]) < 3 or len(by_pos["MID"]) < 2 or len(by_pos["FWD"]) < 1:
        top11 = sorted(candidates, key=lambda x: -x["match_score"])[:11]
        captain = top11[0] if top11 else None
        return {
            "players": top11,
            "by_pos": {
                "GK": [p for p in top11 if p["position"] == "GK"],
                "DEF": [p for p in top11 if p["position"] == "DEF"],
                "MID": [p for p in top11 if p["position"] == "MID"],
                "FWD": [p for p in top11 if p["position"] == "FWD"],
            },
            "formation": "Auto",
            "captain": captain,
            "total_score": round(sum(p["match_score"] for p in top11), 1),
        }

    # Mandatory core: 1 GK + 3 DEF + 2 MID + 1 FWD = 7 players
    chosen = (
        by_pos["GK"][:1]
        + by_pos["DEF"][:3]
        + by_pos["MID"][:2]
        + by_pos["FWD"][:1]
    )
    chosen_ids = {p["player"].pk for p in chosen}
    counts = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
    max_limits = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}

    # Fill remaining 4 outfield slots with highest-scoring eligible players
    flex_pool = sorted(
        [p for p in by_pos["DEF"][3:] + by_pos["MID"][2:] + by_pos["FWD"][1:] if p["player"].pk not in chosen_ids],
        key=lambda x: -x["match_score"],
    )
    for p in flex_pool:
        if len(chosen) == 11:
            break
        pos = p["position"]
        if counts[pos] < max_limits[pos]:
            chosen.append(p)
            chosen_ids.add(p["player"].pk)
            counts[pos] += 1

    xi_by_pos = {
        "GK": [p for p in chosen if p["position"] == "GK"],
        "DEF": sorted([p for p in chosen if p["position"] == "DEF"], key=lambda x: -x["match_score"]),
        "MID": sorted([p for p in chosen if p["position"] == "MID"], key=lambda x: -x["match_score"]),
        "FWD": sorted([p for p in chosen if p["position"] == "FWD"], key=lambda x: -x["match_score"]),
    }
    formation = f"{len(xi_by_pos['DEF'])}-{len(xi_by_pos['MID'])}-{len(xi_by_pos['FWD'])}"
    captain = max(chosen, key=lambda x: x["match_score"]) if chosen else None

    return {
        "players": chosen,
        "by_pos": xi_by_pos,
        "formation": formation,
        "captain": captain,
        "total_score": round(sum(p["match_score"] for p in chosen), 1),
    }


def predict_match_best_lineups(home_team: Team, away_team: Team) -> dict:
    """
    Predict the best 11-player lineup for `home_team` vs `away_team`,
    `away_team` vs `home_team`, and the Combined Match Best XI.
    """
    active_qs = Player.objects.filter(team__in=[home_team, away_team]).select_related("team")
    if active_qs.filter(last_season="2026-27").exists():
        active_qs = active_qs.filter(last_season="2026-27")

    players = [
        p for p in active_qs
        if p.player_key.strip().lower() not in DEPARTED_PLAYERS_2026_27
    ]

    # Pre-aggregate all-season career points per match in a single query
    career_rows = (
        PlayerGameweekStats.objects.filter(player__in=players, minutes__gt=0)
        .values("player_id")
        .annotate(avg_pts=Avg("total_points"), total_pts=Sum("total_points"), total_mins=Sum("minutes"))
    )
    career_map = {
        r["player_id"]: {
            "avg_pts": float(r["avg_pts"] or 0.0),
            "total_pts": int(r["total_pts"] or 0),
            "total_mins": int(r["total_mins"] or 0),
        }
        for r in career_rows
    }

    home_def_strength = _get_team_defensive_strength(home_team)
    away_def_strength = _get_team_defensive_strength(away_team)

    home_candidates = []
    away_candidates = []

    for p in players:
        if p.team_id == home_team.pk:
            scored = _score_player_for_match(
                p,
                opponent_team=away_team,
                was_home=1,
                opp_def_strength=away_def_strength,
                career_stats_map=career_map,
            )
            if scored:
                home_candidates.append(scored)
        elif p.team_id == away_team.pk:
            scored = _score_player_for_match(
                p,
                opponent_team=home_team,
                was_home=0,
                opp_def_strength=home_def_strength,
                career_stats_map=career_map,
            )
            if scored:
                away_candidates.append(scored)

    return {
        "combined_xi": _select_valid_xi(home_candidates + away_candidates),
        "home_xi": _select_valid_xi(home_candidates),
        "away_xi": _select_valid_xi(away_candidates),
    }