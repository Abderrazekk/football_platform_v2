from datetime import date

from django.test import TestCase
from django.urls import reverse

from apps.matches.models import Match, Season, Team
from apps.core.templatetags.team_tags import team_logo, team_logo_url


class TeamLogoTagTests(TestCase):
    def test_every_team_has_a_crest_file(self):
        missing = [
            team.name
            for team in Team.objects.all()
            if not team_logo_url(team)
        ]
        self.assertEqual(missing, [], f"Missing crests for: {missing}")

    def test_all_seeded_club_names_resolve(self):
        # Names currently stored in the project database.
        for name in [
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
            "Burnley", "Chelsea", "Coventry", "Crystal Palace", "Everton",
            "Fulham", "Hull", "Ipswich", "Leeds", "Leicester", "Liverpool",
            "Luton", "Man City", "Man United", "Newcastle", "Norwich",
            "Nottingham Forest", "Sheffield United", "Southampton",
            "Sunderland", "Tottenham", "Watford", "West Brom", "West Ham",
            "Wolves",
        ]:
            with self.subTest(team=name):
                self.assertTrue(team_logo_url(name))

    def test_logo_tag_renders_an_image(self):
        html = str(team_logo("Man City", 32))
        self.assertIn("logos/man-city.png", html)
        self.assertIn('width="32"', html)

    def test_logo_tag_accepts_a_team_instance(self):
        team = Team.objects.create(name="Test United")
        html = str(team_logo(team, 24))
        self.assertIn(team.name, html)
        self.assertIn("TU", html)

    def test_logo_tag_falls_back_to_initials(self):
        html = str(team_logo("Unknown Athletic", 28))
        self.assertIn("Unknown Athletic", html)
        self.assertIn("UA", html)

    def test_logo_tag_handles_empty_values(self):
        self.assertEqual(str(team_logo(None)), "")


class TeamLogoPageTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.season = Season.objects.create(name="2025-26", is_current=True)
        cls.home = Team.objects.create(name="Arsenal")
        cls.away = Team.objects.create(name="Chelsea")
        Match.objects.create(
            season=cls.season,
            match_date=date(2025, 8, 15),
            home_team=cls.home,
            away_team=cls.away,
            is_played=True,
            home_goals=2,
            away_goals=1,
            result="H",
        )

    def test_home_page_shows_all_team_crests(self):
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        for team in Team.objects.all():
            self.assertIn(team_logo_url(team), html)

    def test_teams_list_shows_crests(self):
        response = self.client.get(reverse("teams_list"))
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertIn(team_logo_url("Arsenal"), html)
        self.assertIn(team_logo_url("Chelsea"), html)
