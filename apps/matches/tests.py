from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

User = get_user_model()


class MatchesViewsTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username="tester", password="StrongPass123!")

    def test_upcoming_page_loads(self):
        resp = self.client.get(reverse("matches:upcoming"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Prédire")

    def test_predict_page_loads_for_authenticated_user(self):
        self.client.force_login(self.user)
        resp = self.client.get(reverse("matches:predict"))
        self.assertEqual(resp.status_code, 200)

    def test_history_requires_login(self):
        resp = self.client.get(reverse("matches:history"))
        self.assertEqual(resp.status_code, 302)

    def test_history_loads_for_authenticated_user(self):
        self.client.force_login(self.user)
        resp = self.client.get(reverse("matches:history"))
        self.assertEqual(resp.status_code, 200)