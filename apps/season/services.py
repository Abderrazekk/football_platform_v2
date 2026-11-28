from datetime import date
from pathlib import Path
import joblib
import numpy as np
from django.conf import settings
from django.db.models import Sum

from apps.matches.models import Match, Season, Team
from apps.matches.services import predict_match
from apps.players.models import Player, PlayerGameweekStats, PlayerSeasonStats
from apps.season.models import SeasonForecast

SEASON_MODEL_PATH = Path(settings.BASE_DIR) / "ml" / "artifacts" / "models" / "season_model.joblib"
_SEASON_MODEL_CACHE = None

SQUAD_QUOTAS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_BUDGET = 100.0
MAX_PER_CLUB = 3
MIN_PRICE_BY_POS = {"GK": 4.0, "DEF": 4.0, "MID": 4.5, "FWD": 4.5}

# Confirmed departures excluded from 2026-27 projections when raw 2026-27 CSV
# was seeded from the 2025-26 squad template
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


def _load_season_model():
    global _SEASON_MODEL_CACHE
    if _SEASON_MODEL_CACHE is None and SEASON_MODEL_PATH.exists():
        _SEASON_MODEL_CACHE = joblib.load(SEASON_MODEL_PATH)
    return _SEASON_MODEL_CACHE


def _normalize_price_millions(raw_val) -> float:
    """Convert FPL tenths (e.g. 125 -> 12.5) or raw float into £M, clamped to >= 4.0M."""
    if raw_val is None:
        return 5.0
    val = float(raw_val)
    if val > 25.0:
        val = val / 10.0
    return round(max(4.0, val), 1)


def _build_prior_season_features(player: Player, target_season: str = "2026-27") -> dict | None:
    """Build feature dictionary from the player's latest prior season stats."""
    latest_hist = (
        PlayerSeasonStats.objects.filter(player=player)
        .exclude(season_name=target_season)
        .order_by("-season_name")
        .first()
    )

    if latest_hist and latest_hist.minutes > 0:
        mins = float(latest_hist.minutes)
        pts = float(latest_hist.total_points)
        goals = float(latest_hist.goals_scored)
        assists = float(latest_hist.assists)
        bps = float(latest_hist.bps)
        ict = float(latest_hist.ict_index)
        price = _normalize_price_millions(latest_hist.end_cost or player.value)
        games = round(mins / 75.0, 1)
    else:
        gw_qs = PlayerGameweekStats.objects.filter(player=player).exclude(season__name=target_season)
        if not gw_qs.exists():
            gw_qs = PlayerGameweekStats.objects.filter(player=player)
        if not gw_qs.exists():
            return None

        latest_gw = gw_qs.select_related("season").order_by("-season__name").first()
        s_qs = gw_qs.filter(season=latest_gw.season)
        agg = s_qs.aggregate(
            m=Sum("minutes"),
            p=Sum("total_points"),
            g=Sum("goals_scored"),
            a=Sum("assists"),
            b=Sum("bps"),
            i=Sum("ict_index"),
        )
        mins = float(agg["m"] or 0.0)
        if mins <= 0:
            return None
        pts = float(agg["p"] or 0.0)
        goals = float(agg["g"] or 0.0)
        assists = float(agg["a"] or 0.0)
        bps = float(agg["b"] or 0.0)
        ict = float(agg["i"] or 0.0)
        price = _normalize_price_millions(player.value)
        games = float(s_qs.filter(minutes__gt=0).count())

    mins_90 = max(mins / 90.0, 0.1)
    pos = (player.position or "MID").upper().replace("GKP", "GK")

    return {
        "prev_minutes": mins,
        "prev_total_points": pts,
        "prev_goals": goals,
        "prev_assists": assists,
        "prev_bps": bps,
        "prev_ict_index": ict,
        "prev_price": price,
        "prev_games_played": games,
        "prev_points_per_90": pts / mins_90,
        "prev_goal_involvements_per_90": (goals + assists) / mins_90,
        "pos_GK": 1 if pos == "GK" else 0,
        "pos_DEF": 1 if pos == "DEF" else 0,
        "pos_MID": 1 if pos == "MID" else 0,
        "pos_FWD": 1 if pos == "FWD" else 0,
    }


def generate_or_get_season_forecasts(
    season_name: str = "2026-27",
    force_refresh: bool = False,
):
    """Return cached SeasonForecast rows for active players in `season_name`."""
    qs = SeasonForecast.objects.filter(season_name=season_name).select_related("player", "player__team")
    if qs.exists() and not force_refresh:
        return qs

    bundle = _load_season_model()

    players = Player.objects.filter(last_season=season_name).select_related("team")
    if not players.exists():
        players = Player.objects.select_related("team").all()

    forecast_objs = []
    for p in players:
        if season_name == "2026-27" and p.player_key.lower().strip() in DEPARTED_PLAYERS_2026_27:
            continue

        feats = _build_prior_season_features(p, target_season=season_name)
        price_m = _normalize_price_millions(p.value)

        if bundle is not None and feats is not None:
            feature_cols = bundle["feature_cols"]
            X = np.array([[feats.get(c, 0.0) for c in feature_cols]], dtype=float)
            X_sc = bundle["scaler"].transform(X)
            pred_pts = float(bundle["model"].predict(X_sc)[0])
            model_ver = bundle.get("model_name", "ridge")
        elif feats is not None:
            pred_pts = feats["prev_total_points"] * 0.85 + 15.0
            model_ver = "fallback_baseline"
        else:
            continue

        pred_pts = round(max(0.0, pred_pts), 1)
        val_score = round(pred_pts / max(price_m, 0.1), 2)

        forecast_objs.append(
            SeasonForecast(
                player=p,
                season_name=season_name,
                projected_points=pred_pts,
                price_millions=price_m,
                value_score=val_score,
                model_version=model_ver,
            )
        )

    SeasonForecast.objects.filter(season_name=season_name).delete()
    SeasonForecast.objects.bulk_create(forecast_objs, batch_size=500, ignore_conflicts=True)
    return SeasonForecast.objects.filter(season_name=season_name).select_related("player", "player__team")


def get_best_players(season_name: str = "2026-27", limit: int = 10) -> list[SeasonForecast]:
    """Top `limit` projected players for the season."""
    qs = generate_or_get_season_forecasts(season_name=season_name)
    return list(qs.order_by("-projected_points")[:limit])


def _is_young_player(player: Player, ref_date: date = date(2026, 8, 15)) -> bool:
    """
    Return True if player age < 22, or if the player is genuinely in their
    1st or 2nd Premier League season (no appearances in 2024-25 or earlier).
    """
    bdate = getattr(player, "birth_date", None)
    if bdate:
        age = ref_date.year - bdate.year - ((ref_date.month, ref_date.day) < (bdate.month, bdate.day))
        return age < 22

    # 1. Check permanent FPL element_code across all seasons
    codes = list(
        PlayerSeasonStats.objects.filter(player=player, element_code__isnull=False)
        .values_list("element_code", flat=True)
        .distinct()
    )
    if codes:
        career_seasons = set(
            PlayerSeasonStats.objects.filter(element_code__in=codes).values_list("season_name", flat=True)
        )
        # Any season in 2024-25 or earlier means >= 3rd PL season by 2026-27
        if any(s <= "2024-25" for s in career_seasons):
            return False
        return 1 <= len(career_seasons) <= 2

    # 2. Check PlayerSeasonStats & PlayerGameweekStats directly on this player
    hist_seasons = set(
        PlayerSeasonStats.objects.filter(player=player).values_list("season_name", flat=True)
    )
    gw_seasons = set(
        PlayerGameweekStats.objects.filter(player=player).values_list("season__name", flat=True)
    )
    all_seasons = hist_seasons | gw_seasons
    if any(s <= "2024-25" for s in all_seasons):
        return False

    # 3. Cross-check renamed records sharing first & last name in <= 2024-25
    tokens = [t for t in player.player_key.split() if len(t) >= 3]
    if len(tokens) >= 2:
        first_tok, last_tok = tokens[0], tokens[-1]
        played_earlier = PlayerGameweekStats.objects.filter(
            player__player_key__icontains=first_tok,
            season__name__lte="2024-25",
            minutes__gt=0,
        ).filter(player__player_key__icontains=last_tok).exists()
        if played_earlier:
            return False

    return 1 <= len(all_seasons) <= 2


def get_best_young_players(season_name: str = "2026-27", limit: int = 10) -> list[SeasonForecast]:
    """Top `limit` projected young players (age < 22 or <= 2 PL seasons)."""
    qs = generate_or_get_season_forecasts(season_name=season_name).order_by("-projected_points")
    young = []
    for fc in qs:
        if _is_young_player(fc.player):
            young.append(fc)
            if len(young) >= limit:
                break
    return young


def build_best_xi(season_name: str = "2026-27", budget: float = MAX_BUDGET) -> dict:
    """
    Greedy Best XI selector:
      1. Sort candidates by projected_points / price (value_score) descending.
      2. Pick 15 players (2 GK + 5 DEF + 5 MID + 3 FWD), total price <= 100.0M,
         max 3 players per club, reserving minimum funds for remaining slots.
      3. Select optimal 11-player starting formation (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD)
         and assign Captain to the highest-projected starter.
    """
    forecasts = list(
        generate_or_get_season_forecasts(season_name=season_name)
        .filter(player__team__isnull=False)
        .order_by("-value_score", "-projected_points")
    )

    selected: list[SeasonForecast] = []
    selected_ids: set[int] = set()
    pos_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    club_counts: dict[int, int] = {}
    remaining_budget = round(float(budget), 2)

    def _min_remaining_cost(cur_pos_counts: dict) -> float:
        total_min = 0.0
        for pos_key, quota in SQUAD_QUOTAS.items():
            needed = quota - cur_pos_counts[pos_key]
            if needed > 0:
                total_min += needed * MIN_PRICE_BY_POS[pos_key]
        return total_min

    # Pass 1: Greedy by points-per-million with reserve-budget protection
    for fc in forecasts:
        if len(selected) == 15:
            break
        pos = (fc.player.position or "").upper().replace("GKP", "GK")
        if pos not in SQUAD_QUOTAS or pos_counts[pos] >= SQUAD_QUOTAS[pos]:
            continue
        team_id = fc.player.team_id
        if club_counts.get(team_id, 0) >= MAX_PER_CLUB:
            continue

        next_counts = dict(pos_counts)
        next_counts[pos] += 1
        min_needed_after = _min_remaining_cost(next_counts)

        if round(remaining_budget - fc.price_millions, 2) >= round(min_needed_after, 2):
            selected.append(fc)
            selected_ids.add(fc.player_id)
            pos_counts[pos] += 1
            club_counts[team_id] = club_counts.get(team_id, 0) + 1
            remaining_budget = round(remaining_budget - fc.price_millions, 2)

    # Pass 2: Cheapest-first fallback if any slot remained open
    if len(selected) < 15:
        cheapest_pool = sorted(forecasts, key=lambda x: (x.price_millions, -x.projected_points))
        for fc in cheapest_pool:
            if len(selected) == 15:
                break
            if fc.player_id in selected_ids:
                continue
            pos = (fc.player.position or "").upper().replace("GKP", "GK")
            if pos not in SQUAD_QUOTAS or pos_counts[pos] >= SQUAD_QUOTAS[pos]:
                continue
            team_id = fc.player.team_id
            if club_counts.get(team_id, 0) >= MAX_PER_CLUB:
                continue
            if round(remaining_budget - fc.price_millions, 2) >= 0:
                selected.append(fc)
                selected_ids.add(fc.player_id)
                pos_counts[pos] += 1
                club_counts[team_id] = club_counts.get(team_id, 0) + 1
                remaining_budget = round(remaining_budget - fc.price_millions, 2)

    # Select Starting XI (1 GK + 3-5 DEF + 2-5 MID + 1-3 FWD = 11 players)
    gks = sorted([f for f in selected if f.player.position == "GK"], key=lambda x: -x.projected_points)
    defs = sorted([f for f in selected if f.player.position == "DEF"], key=lambda x: -x.projected_points)
    mids = sorted([f for f in selected if f.player.position == "MID"], key=lambda x: -x.projected_points)
    fwds = sorted([f for f in selected if f.player.position == "FWD"], key=lambda x: -x.projected_points)

    starting_gk = gks[:1]
    bench_gk = gks[1:]

    mandatory_outfield = defs[:3] + mids[:2] + fwds[:1]
    flex_pool = sorted(defs[3:] + mids[2:] + fwds[1:], key=lambda x: -x.projected_points)
    starting_outfield = mandatory_outfield + flex_pool[:4]
    bench_outfield = flex_pool[4:]

    starting_xi = starting_gk + starting_outfield
    bench = bench_gk + bench_outfield

    captain = max(starting_xi, key=lambda x: x.projected_points) if starting_xi else None
    vice_candidates = [p for p in starting_xi if captain and p.player_id != captain.player_id]
    vice_captain = max(vice_candidates, key=lambda x: x.projected_points) if vice_candidates else None

    xi_by_pos = {
        "GK": [p for p in starting_xi if p.player.position == "GK"],
        "DEF": [p for p in starting_xi if p.player.position == "DEF"],
        "MID": [p for p in starting_xi if p.player.position == "MID"],
        "FWD": [p for p in starting_xi if p.player.position == "FWD"],
    }
    formation = f"{len(xi_by_pos['DEF'])}-{len(xi_by_pos['MID'])}-{len(xi_by_pos['FWD'])}"

    total_cost = round(sum(f.price_millions for f in selected), 1)
    xi_points = sum(f.projected_points for f in starting_xi)
    captain_bonus = captain.projected_points if captain else 0.0

    return {
        "squad": selected,
        "starting_xi": starting_xi,
        "xi_by_pos": xi_by_pos,
        "bench": bench,
        "captain": captain,
        "vice_captain": vice_captain,
        "formation": formation,
        "total_cost": total_cost,
        "remaining_budget": round(budget - total_cost, 1),
        "total_projected_points": round(xi_points + captain_bonus, 1),
        "pos_counts": pos_counts,
    }


def get_projected_season_table(season_name: str = "2026-27") -> tuple[Season | None, list[dict]]:
    """
    Combine actual points from played matches with expected points
    (3 * P_win + 1 * P_draw) from match_model.joblib for remaining fixtures.
    """
    season = Season.objects.filter(name=season_name).first() or Season.objects.order_by("-name").first()
    if not season:
        return None, []

    matches = list(
        Match.objects.filter(season=season).select_related("home_team", "away_team")
    )
    table: dict[int, dict] = {}

    def _ensure_team(team: Team) -> dict:
        if team.pk not in table:
            table[team.pk] = {
                "team": team,
                "played": 0,
                "remaining": 0,
                "won": 0,
                "drawn": 0,
                "lost": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "current_points": 0,
                "projected_remaining_points": 0.0,
                "projected_total_points": 0.0,
            }
        return table[team.pk]

    for m in matches:
        h = _ensure_team(m.home_team)
        a = _ensure_team(m.away_team)

        if m.is_played:
            hg, ag = m.home_goals or 0, m.away_goals or 0
            h["played"] += 1
            a["played"] += 1
            h["gf"] += hg
            h["ga"] += ag
            a["gf"] += ag
            a["ga"] += hg

            if m.result == "H":
                h["won"] += 1
                h["current_points"] += 3
                a["lost"] += 1
            elif m.result == "A":
                a["won"] += 1
                a["current_points"] += 3
                h["lost"] += 1
            else:
                h["drawn"] += 1
                a["drawn"] += 1
                h["current_points"] += 1
                a["current_points"] += 1
        else:
            h["remaining"] += 1
            a["remaining"] += 1
            try:
                probs = predict_match(m.home_team.name, m.away_team.name, match_date=m.match_date)
                p_h, p_d, p_a = probs["home"] / 100.0, probs["draw"] / 100.0, probs["away"] / 100.0
            except Exception:
                p_h, p_d, p_a = 0.45, 0.27, 0.28

            h["projected_remaining_points"] += 3.0 * p_h + 1.0 * p_d
            a["projected_remaining_points"] += 3.0 * p_a + 1.0 * p_d

    rows = []
    for row in table.values():
        row["gd"] = row["gf"] - row["ga"]
        row["projected_remaining_points"] = round(row["projected_remaining_points"], 1)
        row["projected_total_points"] = round(
            row["current_points"] + row["projected_remaining_points"], 1
        )
        rows.append(row)

    rows.sort(key=lambda r: (-r["projected_total_points"], -r["current_points"], -r["gd"], -r["gf"]))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return season, rows