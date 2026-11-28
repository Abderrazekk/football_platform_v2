from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from apps.players.services import predict_match_best_lineups
from apps.core.templatetags.team_tags import team_logo_url
from .forms import PredictionForm
from .models import Prediction, Team
from .services import (
    PredictionServiceError,
    h2h_summary,
    predict,
    recent_form,
    save_prediction,
)


@login_required
def predict_view(request):
    initial = {}
    home_name = request.GET.get("home", "").strip()
    away_name = request.GET.get("away", "").strip()
    if home_name:
        h_team = Team.objects.filter(name__iexact=home_name).first()
        if h_team:
            initial["home_team"] = h_team
    if away_name:
        a_team = Team.objects.filter(name__iexact=away_name).first()
        if a_team:
            initial["away_team"] = a_team

    form = PredictionForm(request.POST or None, initial=initial)

    if request.method == "POST" and form.is_valid():
        home_team = form.cleaned_data["home_team"]
        away_team = form.cleaned_data["away_team"]
        try:
            result = predict(home_team, away_team)
            prediction = save_prediction(request.user, home_team, away_team, result)
            return redirect("matches:prediction_detail", pk=prediction.pk)
        except PredictionServiceError as exc:
            messages.error(request, str(exc))

    return render(request, "matches/predict.html", {
        "form": form,
        "team_logos": {str(team.pk): team_logo_url(team) for team in Team.objects.all()},
    })


@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user).select_related(
        "home_team", "away_team"
    )
    paginator = Paginator(qs, 15)
    page_obj = paginator.get_page(request.GET.get("page"))
    return render(
        request,
        "matches/history.html",
        {"predictions": page_obj, "page_obj": page_obj},
    )


@login_required
def prediction_detail_view(request, pk):
    prediction = get_object_or_404(
        Prediction.objects.select_related("home_team", "away_team"),
        pk=pk,
        user=request.user,
    )
    try:
        live_info = predict(prediction.home_team, prediction.away_team)
        key_factors = live_info["key_factors"]
    except PredictionServiceError:
        key_factors = []

    lineups = predict_match_best_lineups(prediction.home_team, prediction.away_team)

    context = {
        "prediction": prediction,
        "key_factors": key_factors,
        "h2h": h2h_summary(prediction.home_team, prediction.away_team),
        "home_form": recent_form(prediction.home_team, n=5),
        "away_form": recent_form(prediction.away_team, n=5),
        "lineups": lineups,
    }
    return render(request, "matches/prediction_detail.html", context)


@login_required
@require_POST
def delete_prediction_view(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, user=request.user)
    prediction.delete()
    messages.success(request, "Prédiction supprimée.")
    return redirect("matches:history")


def upcoming_fixtures(request):
    fixtures = [
        {"date": "2026-11-28 15:00", "league": "Premier League", "home": "Arsenal", "away": "Chelsea"},
        {"date": "2026-11-28 17:30", "league": "Premier League", "home": "Liverpool", "away": "Man City"},
        {"date": "2026-11-29 20:00", "league": "Premier League", "home": "Tottenham", "away": "Man United"},
    ]
    return render(request, "matches/upcoming.html", {"fixtures": fixtures})