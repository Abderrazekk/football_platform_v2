from django.urls import path
from apps.season import views

app_name = "season"

urlpatterns = [
    path("best-player/", views.best_player_view, name="best_player"),
    path("best-young/", views.best_young_view, name="best_young"),
    path("best-xi/", views.best_xi_view, name="best_xi"),
    path("table/", views.projected_table_view, name="table"),
]