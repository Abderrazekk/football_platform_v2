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
        
@patch("apps.players.services._load_player_model", return_value=_mock_bundle(6.0))
@patch(
        "apps.players.services._get_player_vs_opponent_index",
        return_value={
            ("bukayo saka", "Chelsea"): {
                "matches": 6,
                "minutes": 520,
                "total_points": 54,
                "avg_points": 9.0,
                "goals": 4,
                "assists": 2,
                "clean_sheets": 2,
                "avg_bps": 34.0,
            }
        },
    )
def test_predict_match_best_lineups_with_h2h(self, _mock_vs, _mock_model):
        from apps.players.services import predict_match_best_lineups

        # Populate a full 11-player squad for Arsenal (1 GK, 4 DEF, 4 MID, 2 FWD)
        squad_specs = [
            ("david raya", "David Raya", "GK"),
            ("william saliba", "William Saliba", "DEF"),
            ("gabriel magalhaes", "Gabriel Magalhaes", "DEF"),
            ("ben white", "Ben White", "DEF"),
            ("jurrien timber", "Jurrien Timber", "DEF"),
            ("martin odegaard", "Martin Odegaard", "MID"),
            ("declan rice", "Declan Rice", "MID"),
            ("gabriel martinelli", "Gabriel Martinelli", "MID"),
            ("kai havertz", "Kai Havertz", "FWD"),
            ("leandro trossard", "Leandro Trossard", "FWD"),
        ]
        for key, name, pos in squad_specs:
            p = Player.objects.create(
                player_key=key, name=name, position=pos, team=self.arsenal, last_season="2026-27"
            )
            PlayerGameweekStats.objects.create(
                player=p, season=self.season, round_num=1, minutes=90, total_points=5
            )

        lineups = predict_match_best_lineups(self.arsenal, self.chelsea)
        home_xi = lineups["home_xi"]
        self.assertEqual(len(home_xi["players"]), 11)
        self.assertEqual(len(home_xi["by_pos"]["GK"]), 1)
        self.assertTrue(3 <= len(home_xi["by_pos"]["DEF"]) <= 5)
        self.assertTrue(2 <= len(home_xi["by_pos"]["MID"]) <= 5)
        self.assertTrue(1 <= len(home_xi["by_pos"]["FWD"]) <= 3)

        # Saka's strong H2H record vs Chelsea (9.0 avg pts) boosts his match_score above 6.0
        saka_entry = next(x for x in home_xi["players"] if x["player"].pk == self.saka.pk)
        self.assertGreater(saka_entry["match_score"], 6.5)
        self.assertIn("4B, 2PD", saka_entry["badge"])    