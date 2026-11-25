from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

User = get_user_model()


class AccountsTests(TestCase):
    def test_register_creates_user(self):
        resp = self.client.post(
            reverse("accounts:register"),
            {
                "username": "newuser",
                "email": "new@example.com",
                "password1": "StrongPass123!",
                "password2": "StrongPass123!",
            },
            follow=True,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(User.objects.filter(username="newuser").exists())

    def test_login_and_logout(self):
        User.objects.create_user(username="alice", password="StrongPass123!")
        login_resp = self.client.post(
            reverse("accounts:login"),
            {"username": "alice", "password": "StrongPass123!"},
            follow=True,
        )
        self.assertEqual(login_resp.status_code, 200)
        self.assertTrue(login_resp.context["user"].is_authenticated)

        logout_resp = self.client.post(reverse("accounts:logout"), follow=True)
        self.assertEqual(logout_resp.status_code, 200)
        self.assertFalse(logout_resp.context["user"].is_authenticated)