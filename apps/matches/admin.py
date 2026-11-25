from django.contrib import admin
from .models import Season, Team, Match


@admin.register(Season)
class SeasonAdmin(admin.ModelAdmin):
    list_display = ("name", "is_current")


@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ("name", "cluster_label")
    search_fields = ("name",)


@admin.register(Match)
class MatchAdmin(admin.ModelAdmin):
    list_display = ("match_date", "season", "home_team", "away_team", "home_goals", "away_goals", "result", "is_played")
    list_filter = ("season", "is_played", "result")
    search_fields = ("home_team__name", "away_team__name")