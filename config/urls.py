"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/6.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from apps.core import views as core_views

urlpatterns = [
    path("", core_views.home_view, name="home"),
    path("admin/", admin.site.urls),
    path("accounts/", include("apps.accounts.urls")),
    path("matches/", include("apps.matches.urls")),
    path("favorites/", include("apps.favorites.urls")),
    # Phase 6: Teams & Model Performance
    path("teams/", core_views.teams_list_view, name="teams_list"),
    path("teams/compare/", core_views.team_compare_view, name="team_compare"),
    path("teams/<int:pk>/", core_views.team_detail_view, name="team_detail"),
    path("model/performance/", core_views.model_performance_view, name="model_performance"),
    path("ml-figures/<str:filename>/", core_views.ml_figure_view, name="ml_figure"),
]