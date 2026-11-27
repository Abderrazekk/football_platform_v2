import json
from django.core.paginator import Paginator
from django.db.models import Q
from django.shortcuts import get_object_or_404, render

from apps.matches.models import Match, Team
from apps.players.models import Player
from apps.players.services import (
    compare_players,
    get_match_stars_and_risks,
    get_player_form_series,
    get_player_season_summary,
    predict_player_points,
)


def player_list(request):
    q = request.GET.get("q", "").strip()
    position = request.GET.get("position", "").strip().upper()
    team_id = request.GET.get("team", "").strip()

    qs = Player.objects.select_related("team").order_by("name")
    if q:
        qs = qs.filter(Q(name__icontains=q) | Q(team__name__icontains=q))
    if position in {"GK", "DEF", "MID", "FWD"}:
        qs = qs.filter(position=position)
    if team_id.isdigit():
        qs = qs.filter(team_id=int(team_id))

    paginator = Paginator(qs, 24)
    page_obj = paginator.get_page(request.GET.get("page"))

    context = {
        "page_obj": page_obj,
        "teams": Team.objects.order_by("name"),
        "positions": ["GK", "DEF", "MID", "FWD"],
        "q": q,
        "selected_position": position,
        "selected_team": team_id,
    }
    return render(request, "players/player_list.html", context)


def player_detail(request, pk: int):
    player = get_object_or_404(Player.objects.select_related("team"), pk=pk)
    summary = get_player_season_summary(player)
    form_series = get_player_form_series(player, limit=10)
    expected_home = predict_player_points(player, was_home=1)
    expected_away = predict_player_points(player, was_home=0)

    # Graceful cluster profile support (works before & after clustering phase)
    cluster_profile = getattr(player, "cluster_label", None) or getattr(player, "cluster_profile", None)

    context = {
        "player": player,
        "summary": summary,
        "expected_home": expected_home,
        "expected_away": expected_away,
        "cluster_profile": cluster_profile,
        "form_json": json.dumps(form_series),
    }
    return render(request, "players/player_detail.html", context)


def player_compare(request):
    all_players = Player.objects.select_related("team").order_by("name")
    p1_id = request.GET.get("p1", "").strip()
    p2_id = request.GET.get("p2", "").strip()

    comparison = None
    if p1_id.isdigit() and p2_id.isdigit():
        player_a = get_object_or_404(Player.objects.select_related("team"), pk=int(p1_id))
        player_b = get_object_or_404(Player.objects.select_related("team"), pk=int(p2_id))
        comparison = compare_players(player_a, player_b)

    context = {
        "players": all_players,
        "p1_id": p1_id,
        "p2_id": p2_id,
        "comparison": comparison,
    }
    return render(request, "players/player_compare.html", context)


def match_stars(request):
    upcoming_matches = (
        Match.objects.filter(home_goals__isnull=True)
        .select_related("home_team", "away_team")
        .order_by("match_date")[:30]
    )
    if not upcoming_matches:
        upcoming_matches = (
            Match.objects.select_related("home_team", "away_team")
            .order_by("-match_date")[:20]
        )

    match_id = request.GET.get("match", "").strip()
    selected_match = None
    result = None

    if match_id.isdigit():
        selected_match = get_object_or_404(
            Match.objects.select_related("home_team", "away_team"), pk=int(match_id)
        )
    elif upcoming_matches:
        selected_match = upcoming_matches[0]

    if selected_match:
        result = get_match_stars_and_risks(
            selected_match.home_team, selected_match.away_team, top_n=3
        )

    context = {
        "matches": upcoming_matches,
        "selected_match": selected_match,
        "result": result,
    }
    return render(request, "players/match_stars.html", context)