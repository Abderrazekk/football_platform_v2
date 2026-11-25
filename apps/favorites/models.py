from django.conf import settings
from django.db import models
from apps.matches.models import Team


class FavoriteTeam(models.Model):
    class Priority(models.IntegerChoices):
        HAUTE = 1, "Haute"
        MOYENNE = 2, "Moyenne"
        BASSE = 3, "Basse"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="favorite_teams",
    )
    team = models.ForeignKey(
        Team,
        on_delete=models.CASCADE,
        related_name="favorited_by",
    )
    priority = models.PositiveSmallIntegerField(
        choices=Priority.choices,
        default=Priority.MOYENNE,
    )
    note = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["priority", "team__name"]
        constraints = [
            models.UniqueConstraint(
                fields=["user", "team"],
                name="unique_user_favorite_team",
            )
        ]

    def __str__(self):
        return f"{self.user.username} - {self.team.name} ({self.get_priority_display()})"