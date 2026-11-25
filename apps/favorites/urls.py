from django.urls import path
from . import views

app_name = "favorites"

urlpatterns = [
    path("", views.favorite_list, name="list"),
    path("add/", views.favorite_create, name="create"),
    path("<int:pk>/", views.favorite_detail, name="detail"),
    path("<int:pk>/edit/", views.favorite_update, name="update"),
    path("<int:pk>/delete/", views.favorite_delete, name="delete"),
]