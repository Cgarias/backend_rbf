from . import views
from django.urls import path
from .views import TrainRBFView

urlpatterns = [
    path('train/', TrainRBFView.as_view(), name='train_rbf'),
    path("predict/", views.predict_rbf_view, name="predict_rbf"),
]
