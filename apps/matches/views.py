from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from .forms import PredictionForm
from .models import Prediction
from .services import (
    PredictionServiceError,
    h2h_summary,
    predict,
    recent_form,
    save_prediction,
)


@login_required
def predict_view(request):
    form = PredictionForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        home_team = form.cleaned_data["home_team"]
        away_team = form.cleaned_data["away_team"]
        try:
            result = predict(home_team, away_team)
            prediction = save_prediction(request.user, home_team, away_team, result)
            return redirect("matches:prediction_detail", pk=prediction.pk)
        except PredictionServiceError as exc:
            messages.error(request, str(exc))

    return render(request, "matches/predict.html", {"form": form})


@login_required
def history_view(request):
    predictions = Prediction.objects.filter(user=request.user).select_related(
        "home_team", "away_team"
    )
    return render(request, "matches/history.html", {"predictions": predictions})


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

    context = {
        "prediction": prediction,
        "key_factors": key_factors,
        "h2h": h2h_summary(prediction.home_team, prediction.away_team),
        "home_form": recent_form(prediction.home_team, n=5),
        "away_form": recent_form(prediction.away_team, n=5),
    }
    return render(request, "matches/prediction_detail.html", context)


@login_required
@require_POST
def delete_prediction_view(request, pk):
    prediction = get_object_or_404(Prediction, pk=pk, user=request.user)
    prediction.delete()
    messages.success(request, "Prediction deleted.")
    return redirect("matches:history")