from collections import Counter
from datetime import date
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from apps.matches.models import Match, Season, Team
from apps.players.models import Player, PlayerSeasonStats
from apps.season.models import SeasonForecast
from apps.season.services import build_best_xi, get_best_young_players, get_projected_season_table


class SeasonModuleTests(TestCase):
    def setUp(self):
        self.season_prev = Season.objects.create(name="2025-26")
        self.season_curr = Season.objects.create(name="2026-27", is_current=True)

        # Create 6 clubs and 24 players (4 per club) to test max-3-per-club & budget constraints
        self.teams = [Team.objects.create(name=f"Club {i}") for i in range(1, 7)]
        positions = ["GK", "DEF", "DEF", "MID", "MID", "FWD"]

        idx = 1
        for team in self.teams:
            for pos in positions:
                p = Player.objects.create(
                    player_key=f"player {idx}",
                    name=f"Player {idx}",
                    position=pos,
                    team=team,
                    value=60,  # £6.0M
                )
                # Give odd-indexed players 1 season (eligible for Best Young) and even-indexed 3 seasons
                n_seasons = 1 if idx % 2 == 1 else 3
                for s_offset in range(n_seasons):
                    PlayerSeasonStats.objects.create(
                        player=p,
                        season_name=f"202{5 - s_offset}-2{6 - s_offset}",
                        minutes=2500,
                        total_points=100 + idx,
                        goals_scored=5,
                        assists=5,
                        bps=400,
                        ict_index=120.0,
                        end_cost=60,
                    )
                SeasonForecast.objects.create(
                    player=p,
                    season_name="2026-27",
                    projected_points=round(110.0 + idx * 1.5, 1),
                    price_millions=6.0,
                    value_score=round((110.0 + idx * 1.5) / 6.0, 2),
                )
                idx += 1

        Match.objects.create(
            season=self.season_curr,
            match_date=date(2026, 8, 16),
            home_team=self.teams[0],
            away_team=self.teams[1],
            is_played=True,
            home_goals=2,
            away_goals=1,
            result="H",
        )
        Match.objects.create(
            season=self.season_curr,
            match_date=date(2026, 12, 5),
            home_team=self.teams[1],
            away_team=self.teams[0],
            is_played=False,
        )

    def test_greedy_best_xi_constraints(self):
        res = build_best_xi(season_name="2026-27", budget=100.0)
        squad = res["squad"]
        starting_xi = res["starting_xi"]

        # 1. Exact 15-player squad & 11-player starting XI
        self.assertEqual(len(squad), 15)
        self.assertEqual(len(starting_xi), 11)
        self.assertEqual(len(res["bench"]), 4)

        # 2. Exact positional quotas (2 GK, 5 DEF, 5 MID, 3 FWD)
        pos_counts = Counter(f.player.position for f in squad)
        self.assertEqual(pos_counts["GK"], 2)
        self.assertEqual(pos_counts["DEF"], 5)
        self.assertEqual(pos_counts["MID"], 5)
        self.assertEqual(pos_counts["FWD"], 3)

        # 3. Budget <= 100.0M
        self.assertLessEqual(res["total_cost"], 100.0)

        # 4. Max 3 players per club
        club_counts = Counter(f.player.team_id for f in squad)
        self.assertTrue(all(cnt <= 3 for cnt in club_counts.values()))

        # 5. Valid starting formation & captain assigned
        xi_pos = Counter(f.player.position for f in starting_xi)
        self.assertEqual(xi_pos["GK"], 1)
        self.assertTrue(3 <= xi_pos["DEF"] <= 5)
        self.assertTrue(2 <= xi_pos["MID"] <= 5)
        self.assertTrue(1 <= xi_pos["FWD"] <= 3)
        self.assertIsNotNone(res["captain"])
        self.assertIn(res["captain"], starting_xi)

    def test_best_young_players_filter(self):
        young = get_best_young_players(season_name="2026-27", limit=10)
        self.assertEqual(len(young), 10)
        # Only odd-indexed players had <= 2 seasons
        for fc in young:
            num = int(fc.player.name.split()[-1])
            self.assertEqual(num % 2, 1)

    @patch("apps.season.services.predict_match", return_value={"home": 50.0, "draw": 25.0, "away": 25.0})
    def test_projected_table_calculation(self, _mock_pred):
        season, rows = get_projected_season_table(season_name="2026-27")
        self.assertEqual(season.name, "2026-27")
        by_name = {r["team"].name: r for r in rows}
        # Club 1 won match 1 (3 pts) + away in match 2 (3*0.25 + 1*0.25 = 1.0 pt) = 4.0
        self.assertEqual(by_name["Club 1"]["projected_total_points"], 4.0)
        # Club 2 lost match 1 (0 pts) + home in match 2 (3*0.50 + 1*0.25 = 1.75 -> 1.8)
        self.assertEqual(by_name["Club 2"]["projected_total_points"], 1.8)

    @patch("apps.season.services.predict_match", return_value={"home": 45.0, "draw": 30.0, "away": 25.0})
    def test_season_views_render(self, _mock_pred):
        for url_name in ("season:best_player", "season:best_young", "season:best_xi", "season:table"):
            resp = self.client.get(reverse(url_name))
            self.assertEqual(resp.status_code, 200)