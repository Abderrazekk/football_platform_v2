from django.contrib import admin
from .models import Player, PlayerSeasonStats, PlayerGameweekStats


@admin.register(Player)
class PlayerAdmin(admin.ModelAdmin):
    list_display = ("name", "position", "team", "value", "last_season", "cluster_label")
    list_filter = ("position", "team", "last_season")
    search_fields = ("name", "player_key")


@admin.register(PlayerSeasonStats)
class PlayerSeasonStatsAdmin(admin.ModelAdmin):
    list_display = ("player", "season_name", "minutes", "goals_scored", "assists", "total_points")
    list_filter = ("season_name",)
    search_fields = ("player__name",)


@admin.register(PlayerGameweekStats)
class PlayerGameweekStatsAdmin(admin.ModelAdmin):
    list_display = ("player", "season", "round_num", "minutes", "goals_scored", "assists", "total_points")
    list_filter = ("season", "round_num")
    search_fields = ("player__name",)