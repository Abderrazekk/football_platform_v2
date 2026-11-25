from django.db import models
from apps.matches.models import Season, Team


class Player(models.Model):
    POSITION_CHOICES = [
        ("GK", "Goalkeeper"),
        ("DEF", "Defender"),
        ("MID", "Midfielder"),
        ("FWD", "Forward"),
    ]

    player_key = models.CharField(max_length=150, unique=True)
    name = models.CharField(max_length=150)
    position = models.CharField(max_length=3, choices=POSITION_CHOICES, default="MID")
    team = models.ForeignKey(Team, on_delete=models.SET_NULL, null=True, blank=True, related_name="players")
    value = models.IntegerField(null=True, blank=True)  # In tenths of £M (e.g. 55 = £5.5M)
    last_season = models.CharField(max_length=15, blank=True)
    cluster_label = models.CharField(max_length=100, null=True, blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return f"{self.name} ({self.position} - {self.team or 'N/A'})"


class PlayerSeasonStats(models.Model):
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="season_stats")
    season_name = models.CharField(max_length=15)
    element_code = models.IntegerField(null=True, blank=True)

    minutes = models.IntegerField(default=0)
    goals_scored = models.IntegerField(default=0)
    assists = models.IntegerField(default=0)
    clean_sheets = models.IntegerField(default=0)
    goals_conceded = models.IntegerField(default=0)
    bps = models.IntegerField(default=0)
    ict_index = models.FloatField(default=0.0)
    total_points = models.IntegerField(default=0)
    start_cost = models.IntegerField(null=True, blank=True)
    end_cost = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-season_name", "-total_points"]
        constraints = [
            models.UniqueConstraint(
                fields=["player", "season_name"],
                name="unique_player_season_stats",
            )
        ]

    def __str__(self):
        return f"{self.player.name} ({self.season_name}): {self.total_points} pts"


class PlayerGameweekStats(models.Model):
    player = models.ForeignKey(Player, on_delete=models.CASCADE, related_name="gameweek_stats")
    season = models.ForeignKey(Season, on_delete=models.CASCADE, related_name="player_gameweeks")
    round_num = models.IntegerField()
    kickoff_time = models.DateTimeField(null=True, blank=True)
    was_home = models.BooleanField(default=False)

    minutes = models.IntegerField(default=0)
    goals_scored = models.IntegerField(default=0)
    assists = models.IntegerField(default=0)
    clean_sheets = models.IntegerField(default=0)
    goals_conceded = models.IntegerField(default=0)
    bps = models.IntegerField(default=0)
    ict_index = models.FloatField(default=0.0)
    xp = models.FloatField(null=True, blank=True)
    total_points = models.IntegerField(default=0)
    value = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-season__name", "-round_num"]
        constraints = [
            models.UniqueConstraint(
                fields=["player", "season", "round_num", "kickoff_time"],
                name="unique_player_gw_entry",
            )
        ]

    def __str__(self):
        return f"{self.player.name} | {self.season.name} GW{self.round_num}: {self.total_points} pts"