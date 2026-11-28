from pathlib import Path
from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404, render

from apps.matches.models import Season, Team
from .services import (
    get_home_kpis,
    get_season_standings,
    get_team_trend_data,
    load_metrics,
    load_team_cluster_stats,
)

FIGURES_DIR = Path(settings.BASE_DIR) / "ml" / "reports" / "figures"


def home_view(request):
    kpis = get_home_kpis(request.user)
    teams = Team.objects.all().order_by("name")
    return render(request, "home.html", {"kpis": kpis, "teams": teams})


def teams_list_view(request):
    seasons = Season.objects.all().order_by("-name")
    selected_season = request.GET.get("season")
    season, standings = get_season_standings(selected_season)
    return render(
        request,
        "teams/list.html",
        {
            "seasons": seasons,
            "current_season": season,
            "standings": standings,
        },
    )


def team_detail_view(request, pk):
    team = get_object_or_404(Team, pk=pk)
    trend = get_team_trend_data(team, limit=15)
    cluster_stats = load_team_cluster_stats().get(team.name)
    return render(
        request,
        "teams/detail.html",
        {
            "team": team,
            "trend": trend,
            "cluster_stats": cluster_stats,
        },
    )


def team_compare_view(request):
    teams = Team.objects.all().order_by("name")
    team_a_id = request.GET.get("team_a")
    team_b_id = request.GET.get("team_b")

    team_a = Team.objects.filter(pk=team_a_id).first() if team_a_id else teams.first()
    team_b = (
        Team.objects.filter(pk=team_b_id).first()
        if team_b_id
        else (teams[1] if teams.count() > 1 else teams.first())
    )

    stats_map = load_team_cluster_stats()
    stats_a = stats_map.get(team_a.name) if team_a else None
    stats_b = stats_map.get(team_b.name) if team_b else None

    return render(
        request,
        "teams/compare.html",
        {
            "teams": teams,
            "team_a": team_a,
            "team_b": team_b,
            "stats_a": stats_a,
            "stats_b": stats_b,
        },
    )


def model_performance_view(request):
    metrics = load_metrics()
    winner = metrics.get("winning_model_on_test", "stacking")

    cm_file = f"confusion_matrix_{winner}.png"
    figures = {
        "confusion_matrix": cm_file if (FIGURES_DIR / cm_file).exists() else None,
        "top_features": "top_feature_importances.png" if (FIGURES_DIR / "top_feature_importances.png").exists() else None,
        "team_elbow": "clustering_team_elbow_silhouette.png" if (FIGURES_DIR / "clustering_team_elbow_silhouette.png").exists() else None,
        "team_profiles": "clustering_team_profiles.png" if (FIGURES_DIR / "clustering_team_profiles.png").exists() else None,
        "player_elbow": "clustering_player_elbow_silhouette.png" if (FIGURES_DIR / "clustering_player_elbow_silhouette.png").exists() else None,
    }

    return render(
        request,
        "core/model_performance.html",
        {
            "metrics": metrics,
            "figures": figures,
        },
    )


def ml_figure_view(request, filename):
    """Safely serve generated PNG plots from ml/reports/figures/."""
    safe_name = Path(filename).name
    file_path = FIGURES_DIR / safe_name
    if not file_path.exists() or file_path.suffix.lower() != ".png":
        raise Http404("Figure not found.")
    return FileResponse(open(file_path, "rb"), content_type="image/png")