from pathlib import Path
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from apps.matches.models import Season, Team, Match
from apps.players.models import Player, PlayerSeasonStats, PlayerGameweekStats


def safe_int(val, default=0):
    if pd.isna(val):
        return default
    return int(float(val))


def safe_float(val, default=0.0):
    if pd.isna(val):
        return default
    return float(val)


class Command(BaseCommand):
    help = "Import cleaned CSV datasets from data/processed/ into SQLite (idempotent)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete existing records before importing.",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        processed_dir = Path(settings.BASE_DIR) / "data" / "processed"

        if options["reset"]:
            self.stdout.write(self.style.WARNING("Resetting database tables..."))
            PlayerGameweekStats.objects.all().delete()
            PlayerSeasonStats.objects.all().delete()
            Player.objects.all().delete()
            Match.objects.all().delete()
            Team.objects.all().delete()
            Season.objects.all().delete()

        # 1. Import Teams
        teams_df = pd.read_csv(processed_dir / "teams.csv")
        team_map = {}
        for _, row in teams_df.iterrows():
            team_obj, _ = Team.objects.update_or_create(name=str(row["team_name"]).strip())
            team_map[team_obj.name] = team_obj
        self.stdout.write(f"Teams synced: {len(team_map)}")

        # 2. Import Seasons & Matches
        matches_df = pd.read_csv(processed_dir / "clean_matches.csv")
        season_map = {}
        for s_name in sorted(matches_df["season"].unique()):
            season_obj, _ = Season.objects.update_or_create(
                name=s_name,
                defaults={"is_current": (s_name == "2026-27")},
            )
            season_map[s_name] = season_obj

        match_objs = []
        for _, row in matches_df.iterrows():
            is_played = pd.notna(row["Full_Time_Result"])
            match_objs.append(
                Match(
                    season=season_map[row["season"]],
                    match_date=row["Match_Date"],
                    home_team=team_map[row["Home_Team"]],
                    away_team=team_map[row["Away_Team"]],
                    is_played=is_played,
                    home_goals=safe_int(row["Full_Time_Home_Goals"], None),
                    away_goals=safe_int(row["Full_Time_Away_Goals"], None),
                    result=str(row["Full_Time_Result"]) if is_played else None,
                    home_shots=safe_int(row["Home_Shots"], None),
                    away_shots=safe_int(row["Away_Shots"], None),
                    home_shots_on_target=safe_int(row["Home_Shots_On_Target"], None),
                    away_shots_on_target=safe_int(row["Away_Shots_On_Target"], None),
                    home_corners=safe_int(row["Home_Corners"], None),
                    away_corners=safe_int(row["Away_Corners"], None),
                    home_yellow_cards=safe_int(row["Home_Yellow_Cards"], None),
                    away_yellow_cards=safe_int(row["Away_Yellow_Cards"], None),
                    home_red_cards=safe_int(row["Home_Red_Cards"], None),
                    away_red_cards=safe_int(row["Away_Red_Cards"], None),
                )
            )
        Match.objects.bulk_create(match_objs, batch_size=500, ignore_conflicts=True)
        self.stdout.write(f"Matches in DB: {Match.objects.count()}")

        # 3. Import Players
        players_df = pd.read_csv(processed_dir / "players.csv")
        player_map = {}
        for _, row in players_df.iterrows():
            p_key = str(row["player_key"]).strip()
            p_obj, _ = Player.objects.update_or_create(
                player_key=p_key,
                defaults={
                    "name": str(row["name"]).strip(),
                    "position": str(row["position"]).strip(),
                    "team": team_map.get(str(row["team"]).strip()),
                    "value": safe_int(row["value"], None),
                    "last_season": str(row.get("last_season", "")),
                },
            )
            player_map[p_key] = p_obj
        self.stdout.write(f"Players synced: {len(player_map)}")

        # 4. Import Player Season History
        hist_df = pd.read_csv(processed_dir / "player_history.csv", low_memory=False)
        hist_objs = []
        for _, row in hist_df.iterrows():
            p_obj = player_map.get(str(row.get("player_key", "")).strip())
            if not p_obj:
                continue
            hist_objs.append(
                PlayerSeasonStats(
                    player=p_obj,
                    season_name=str(row["season_name"]).strip(),
                    element_code=safe_int(row.get("element_code"), None),
                    minutes=safe_int(row.get("minutes")),
                    goals_scored=safe_int(row.get("goals_scored")),
                    assists=safe_int(row.get("assists")),
                    clean_sheets=safe_int(row.get("clean_sheets")),
                    goals_conceded=safe_int(row.get("goals_conceded")),
                    bps=safe_int(row.get("bps")),
                    ict_index=safe_float(row.get("ict_index")),
                    total_points=safe_int(row.get("total_points")),
                    start_cost=safe_int(row.get("start_cost"), None),
                    end_cost=safe_int(row.get("end_cost"), None),
                )
            )
        PlayerSeasonStats.objects.bulk_create(hist_objs, batch_size=1000, ignore_conflicts=True)
        self.stdout.write(f"PlayerSeasonStats in DB: {PlayerSeasonStats.objects.count()}")

        # 5. Import Player Gameweek Stats
        gw_df = pd.read_csv(processed_dir / "player_gw.csv", low_memory=False)
        gw_df["kickoff_time"] = pd.to_datetime(gw_df["kickoff_time"], errors="coerce", utc=True)

        gw_objs = []
        for _, row in gw_df.iterrows():
            p_obj = player_map.get(str(row["player_key"]).strip())
            s_obj = season_map.get(str(row["season"]).strip())
            if not p_obj or not s_obj:
                continue
            kt = row["kickoff_time"].to_pydatetime() if pd.notna(row["kickoff_time"]) else None
            gw_objs.append(
                PlayerGameweekStats(
                    player=p_obj,
                    season=s_obj,
                    round_num=safe_int(row.get("round")),
                    kickoff_time=kt,
                    was_home=bool(row.get("was_home", False)),
                    minutes=safe_int(row.get("minutes")),
                    goals_scored=safe_int(row.get("goals_scored")),
                    assists=safe_int(row.get("assists")),
                    clean_sheets=safe_int(row.get("clean_sheets")),
                    goals_conceded=safe_int(row.get("goals_conceded")),
                    bps=safe_int(row.get("bps")),
                    ict_index=safe_float(row.get("ict_index")),
                    xp=safe_float(row.get("xP"), None),
                    total_points=safe_int(row.get("total_points")),
                    value=safe_int(row.get("value"), None),
                )
            )
        PlayerGameweekStats.objects.bulk_create(gw_objs, batch_size=2000, ignore_conflicts=True)
        self.stdout.write(f"PlayerGameweekStats in DB: {PlayerGameweekStats.objects.count()}")

        self.stdout.write(self.style.SUCCESS("Data import completed successfully!"))