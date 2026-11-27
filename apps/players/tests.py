from datetime import date
from unittest.mock import MagicMock, patch

from django.test import TestCase
from django.urls import reverse

from apps.matches.models import Match, Season, Team
from apps.players.models import Player, PlayerGameweekStats
from apps.players.services import (
    build_player_feature_vector,
    compare_players,
    get_match_stars_and_risks,
    predict_player_points,
)


def _mock_bundle(return_value=5.25):
    scaler = MagicMock()
    scaler.transform.side_effect = lambda X: X
    model = MagicMock()
    model.predict.return_value = [return_value]
    return {
        "model_name": "xgboost",
        "model": model,
        "scaler": scaler,
        "feature_cols": ["minutes_roll3", "total_points_roll3", "was_home"],
    }


class PlayersModuleTests(TestCase):
    def setUp(self):
        self.season = Season.objects.create(name="2025-26")
        self.arsenal = Team.objects.create(name="Arsenal")
        self.chelsea = Team.objects.create(name="Chelsea")

        self.saka = Player.objects.create(
            player_key="bukayo saka", name="Bukayo Saka", position="MID", team=self.arsenal
        )
        self.palmer = Player.objects.create(
            player_key="cole palmer", name="Cole Palmer", position="MID", team=self.chelsea
        )

        for gw in range(1, 5):
            PlayerGameweekStats.objects.create(
                player=self.saka,
                season=self.season,
                round_num=gw,
                minutes=90,
                total_points=8,
                bps=30,
                goals_scored=1,
                assists=0,
                ict_index=9.5,
            )
            PlayerGameweekStats.objects.create(
                player=self.palmer,
                season=self.season,
                round_num=gw,
                minutes=85,
                total_points=6,
                bps=24,
                goals_scored=0,
                assists=1,
                ict_index=7.2,
            )

        self.match = Match.objects.create(
            season=self.season,
            home_team=self.arsenal,
            away_team=self.chelsea,
            match_date=date(2026, 3, 15),
        )

    def test_build_feature_vector(self):
        feats = build_player_feature_vector(self.saka, was_home=1)
        self.assertIsNotNone(feats)
        self.assertEqual(feats["minutes_roll3"], 90.0)
        self.assertEqual(feats["total_points_roll3"], 8.0)
        self.assertEqual(feats["pos_MID"], 1)

    @patch("apps.players.services._load_player_model", return_value=_mock_bundle(6.4))
    def test_predict_and_compare_players(self, _mock):
        pts = predict_player_points(self.saka, was_home=1)
        self.assertEqual(pts, 6.4)
        comp = compare_players(self.saka, self.palmer)
        self.assertEqual(comp["player_a"]["summary"]["total_points"], 32)
        self.assertEqual(comp["player_b"]["expected_points_away"], 6.4)

    @patch("apps.players.services._load_player_model", return_value=_mock_bundle(5.0))
    def test_match_stars_and_risks(self, _mock):
        res = get_match_stars_and_risks(self.arsenal, self.chelsea, top_n=3)
        self.assertEqual(len(res["stars"]), 2)
        self.assertEqual(len(res["at_risk"]), 2)

    def test_player_list_and_detail_views(self):
        res_list = self.client.get(reverse("players:list"), {"q": "Saka"})
        self.assertEqual(res_list.status_code, 200)
        self.assertContains(res_list, "Bukayo Saka")

        res_detail = self.client.get(reverse("players:detail", args=[self.saka.pk]))
        self.assertEqual(res_detail.status_code, 200)
        self.assertContains(res_detail, "Bukayo Saka")

    @patch("apps.players.services._load_player_model", return_value=_mock_bundle(7.1))
    def test_compare_and_match_stars_views(self, _mock):
        res_cmp = self.client.get(
            reverse("players:compare"), {"p1": self.saka.pk, "p2": self.palmer.pk}
        )
        self.assertEqual(res_cmp.status_code, 200)
        self.assertContains(res_cmp, "Cole Palmer")

        res_stars = self.client.get(
            reverse("players:match_stars"), {"match": self.match.pk}
        )
        self.assertEqual(res_stars.status_code, 200)
        self.assertContains(res_stars, "Note on Individual Match Variance")