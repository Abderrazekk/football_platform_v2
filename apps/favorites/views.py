from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render

from apps.matches.services import recent_form
from .forms import FavoriteTeamForm
from .models import FavoriteTeam


@login_required
def favorite_list(request):
    favorites = (
        FavoriteTeam.objects.filter(user=request.user)
        .select_related("team")
        .order_by("priority", "team__name")
    )
    items = [
        {
            "favorite": fav,
            "form_5": recent_form(fav.team, n=5),
        }
        for fav in favorites
    ]
    return render(request, "favorites/list.html", {"items": items})


@login_required
def favorite_create(request):
    if request.method == "POST":
        form = FavoriteTeamForm(request.POST, user=request.user)
        if form.is_valid():
            fav = form.save(commit=False)
            fav.user = request.user
            fav.save()
            messages.success(request, "Équipe ajoutée")
            return redirect("favorites:list")
    else:
        form = FavoriteTeamForm(user=request.user)

    return render(
        request,
        "favorites/form.html",
        {"form": form, "is_edit": False},
    )


@login_required
def favorite_detail(request, pk):
    favorite = get_object_or_404(
        FavoriteTeam.objects.select_related("team"),
        pk=pk,
        user=request.user,
    )
    form_5 = recent_form(favorite.team, n=5)
    return render(
        request,
        "favorites/detail.html",
        {"favorite": favorite, "form_5": form_5},
    )


@login_required
def favorite_update(request, pk):
    favorite = get_object_or_404(
        FavoriteTeam.objects.select_related("team"),
        pk=pk,
        user=request.user,
    )
    if request.method == "POST":
        form = FavoriteTeamForm(request.POST, instance=favorite, user=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Favori modifié")
            return redirect("favorites:list")
    else:
        form = FavoriteTeamForm(instance=favorite, user=request.user)

    return render(
        request,
        "favorites/form.html",
        {"form": form, "favorite": favorite, "is_edit": True},
    )


@login_required
def favorite_delete(request, pk):
    favorite = get_object_or_404(
        FavoriteTeam.objects.select_related("team"),
        pk=pk,
        user=request.user,
    )
    if request.method == "POST":
        favorite.delete()
        messages.success(request, "Favori supprimé")
        return redirect("favorites:list")

    return render(request, "favorites/confirm_delete.html", {"favorite": favorite})