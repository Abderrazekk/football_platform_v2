from django.db import models
from apps.players.models import Player


class SeasonForecast(models.Model):
    player = models.ForeignKey(
        Player,
        on_delete=models.CASCADE,
        related_name="season_forecasts",
    )
    season_name = models.CharField(max_length=16, default="2026-27", db_index=True)
    projected_points = models.FloatField()
    price_millions = models.FloatField(default=5.0)
    value_score = models.FloatField(default=0.0)
    model_version = models.CharField(max_length=64, default="ridge")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-projected_points"]
        constraints = [
            models.UniqueConstraint(
                fields=["player", "season_name"],
                name="unique_player_season_forecast",
            )
        ]

    def __str__(self):
        return f"{self.player.name} ({self.season_name}): {self.projected_points:.1f} pts"