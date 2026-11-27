from django.urls import path
from apps.players import views

app_name = "players"

urlpatterns = [
    path("", views.player_list, name="list"),
    path("compare/", views.player_compare, name="compare"),
    path("match-stars/", views.match_stars, name="match_stars"),
    path("<int:pk>/", views.player_detail, name="detail"),
]