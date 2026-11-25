from django.urls import path
from . import views

app_name = "matches"

urlpatterns = [
    path("predict/", views.predict_view, name="predict"),
    path("history/", views.history_view, name="history"),
    path("predictions/<int:pk>/", views.prediction_detail_view, name="prediction_detail"),
    path("predictions/<int:pk>/delete/", views.delete_prediction_view, name="delete_prediction"),
]