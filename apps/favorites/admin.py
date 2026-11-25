from django.contrib import admin
from .models import FavoriteTeam


@admin.register(FavoriteTeam)
class FavoriteTeamAdmin(admin.ModelAdmin):
    list_display = ("user", "team", "priority", "note", "created_at")
    list_filter = ("priority", "created_at")
    search_fields = ("user__username", "team__name", "note")