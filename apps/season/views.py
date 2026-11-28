from django.shortcuts import render
from apps.season.services import (
    build_best_xi,
    get_best_players,
    get_best_young_players,
    get_projected_season_table,
)


def best_player_view(request):
    top_players = get_best_players(season_name="2026-27", limit=10)
    return render(
        request,
        "season/best_player.html",
        {"top_players": top_players, "season_name": "2026-27"},
    )


def best_young_view(request):
    young_players = get_best_young_players(season_name="2026-27", limit=10)
    return render(
        request,
        "season/best_young.html",
        {"young_players": young_players, "season_name": "2026-27"},
    )


def best_xi_view(request):
    best_xi = build_best_xi(season_name="2026-27", budget=100.0)
    return render(
        request,
        "season/best_xi.html",
        {"best_xi": best_xi, "season_name": "2026-27"},
    )


def projected_table_view(request):
    season, standings = get_projected_season_table(season_name="2026-27")
    return render(
        request,
        "season/table.html",
        {"season": season, "standings": standings},
    )