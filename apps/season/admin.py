from django.contrib import admin
from apps.season.models import SeasonForecast


@admin.register(SeasonForecast)
class SeasonForecastAdmin(admin.ModelAdmin):
    list_display = ("player", "season_name", "projected_points", "price_millions", "value_score", "model_version", "updated_at")
    list_filter = ("season_name", "model_version", "player__position")
    search_fields = ("player__name", "player__team__name")