from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.matches.models import Team
from .models import FavoriteTeam

User = get_user_model()


class FavoriteTeamCRUDTests(TestCase):
    def setUp(self):
        self.user_a = User.objects.create_user(username="alice", password="password123")
        self.user_b = User.objects.create_user(username="bob", password="password123")
        self.arsenal = Team.objects.create(name="Arsenal")
        self.chelsea = Team.objects.create(name="Chelsea")

    def test_anonymous_redirected_to_login(self):
        for url_name in ("favorites:list", "favorites:create"):
            resp = self.client.get(reverse(url_name))
            self.assertEqual(resp.status_code, 302)
            self.assertIn("/accounts/login/", resp.url)

    def test_create_favorite(self):
        self.client.login(username="alice", password="password123")
        resp = self.client.post(
            reverse("favorites:create"),
            {"team": self.arsenal.pk, "priority": 1, "note": "Top contender"},
            follow=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(
            FavoriteTeam.objects.filter(user=self.user_a, team=self.arsenal, priority=1).exists()
        )
        self.assertContains(resp, "Équipe ajoutée")

    def test_duplicate_favorite_rejected(self):
        FavoriteTeam.objects.create(user=self.user_a, team=self.arsenal, priority=2)
        self.client.login(username="alice", password="password123")
        resp = self.client.post(
            reverse("favorites:create"),
            {"team": self.arsenal.pk, "priority": 1, "note": "Duplicate"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(FavoriteTeam.objects.filter(user=self.user_a, team=self.arsenal).count(), 1)
        self.assertContains(resp, "déjà partie de vos favoris")

    def test_list_scoped_to_user(self):
        FavoriteTeam.objects.create(user=self.user_a, team=self.arsenal, priority=1)
        FavoriteTeam.objects.create(user=self.user_b, team=self.chelsea, priority=1)
        self.client.login(username="alice", password="password123")
        resp = self.client.get(reverse("favorites:list"))
        self.assertContains(resp, "Arsenal")
        self.assertNotContains(resp, "Chelsea")

    def test_update_favorite(self):
        fav = FavoriteTeam.objects.create(user=self.user_a, team=self.arsenal, priority=2, note="Old")
        self.client.login(username="alice", password="password123")
        resp = self.client.post(
            reverse("favorites:update", args=[fav.pk]),
            {"priority": 1, "note": "Updated note"},
            follow=True,
        )
        self.assertEqual(resp.status_code, 200)
        fav.refresh_from_db()
        self.assertEqual(fav.priority, 1)
        self.assertEqual(fav.note, "Updated note")
        self.assertEqual(fav.team, self.arsenal)  # team field stayed locked
        self.assertContains(resp, "Favori modifié")

    def test_delete_favorite(self):
        fav = FavoriteTeam.objects.create(user=self.user_a, team=self.arsenal, priority=3)
        self.client.login(username="alice", password="password123")
        resp = self.client.post(reverse("favorites:delete", args=[fav.pk]), follow=True)
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(FavoriteTeam.objects.filter(pk=fav.pk).exists())
        self.assertContains(resp, "Favori supprimé")

    def test_cross_user_access_returns_404(self):
        fav_b = FavoriteTeam.objects.create(user=self.user_b, team=self.chelsea, priority=1)
        self.client.login(username="alice", password="password123")
        for url_name in ("favorites:detail", "favorites:update", "favorites:delete"):
            resp = self.client.get(reverse(url_name, args=[fav_b.pk]))
            self.assertEqual(resp.status_code, 404)