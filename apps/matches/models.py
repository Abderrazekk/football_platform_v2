from django.conf import settings
from django.db import models


class Season(models.Model):
    name = models.CharField(max_length=15, unique=True)  # e.g. "2025-26"
    is_current = models.BooleanField(default=False)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Team(models.Model):
    name = models.CharField(max_length=100, unique=True)
    cluster_label = models.CharField(max_length=100, null=True, blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Match(models.Model):
    season = models.ForeignKey(Season, on_delete=models.CASCADE, related_name="matches")
    match_date = models.DateField()
    home_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="home_matches")
    away_team = models.ForeignKey(Team, on_delete=models.CASCADE, related_name="away_matches")

    is_played = models.BooleanField(default=True)
    home_goals = models.IntegerField(null=True, blank=True)
    away_goals = models.IntegerField(null=True, blank=True)
    result = models.CharField(max_length=1, null=True, blank=True)  # H, D, A

    home_shots = models.IntegerField(null=True, blank=True)
    away_shots = models.IntegerField(null=True, blank=True)
    home_shots_on_target = models.IntegerField(null=True, blank=True)
    away_shots_on_target = models.IntegerField(null=True, blank=True)
    home_corners = models.IntegerField(null=True, blank=True)
    away_corners = models.IntegerField(null=True, blank=True)
    home_yellow_cards = models.IntegerField(null=True, blank=True)
    away_yellow_cards = models.IntegerField(null=True, blank=True)
    home_red_cards = models.IntegerField(null=True, blank=True)
    away_red_cards = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-match_date", "home_team__name"]
        constraints = [
            models.UniqueConstraint(
                fields=["match_date", "home_team", "away_team"],
                name="unique_match_date_teams",
            )
        ]

    def __str__(self):
        score = f"{self.home_goals}-{self.away_goals}" if self.is_played else "vs"
        return f"{self.match_date} | {self.home_team} {score} {self.away_team}"
    

class Prediction(models.Model):
    RESULT_CHOICES = [
        ("H", "Home Win"),
        ("D", "Draw"),
        ("A", "Away Win"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="predictions",
    )
    home_team = models.ForeignKey(
        "Team",
        on_delete=models.CASCADE,
        related_name="home_predictions",
    )
    away_team = models.ForeignKey(
        "Team",
        on_delete=models.CASCADE,
        related_name="away_predictions",
    )
    match = models.ForeignKey(
        "Match",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="predictions",
    )
    p_home = models.FloatField()
    p_draw = models.FloatField()
    p_away = models.FloatField()
    predicted_result = models.CharField(max_length=1, choices=RESULT_CHOICES)
    actual_result = models.CharField(
        max_length=1,
        choices=RESULT_CHOICES,
        null=True,
        blank=True,
    )
    model_version = models.CharField(max_length=64, default="stacking")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.home_team} vs {self.away_team} ({self.predicted_result}) - {self.user.username}"    