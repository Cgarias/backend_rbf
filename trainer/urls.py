from django.urls import path
from trainer.views import (
    DatasetInfoView,
    TrainRBFView,
    PredictTestView,
    PredictManualView,
    HealthCheckView,
)

urlpatterns = [
    path("dataset-info/", DatasetInfoView.as_view(), name="dataset-info"),
    path("train-rbf/", TrainRBFView.as_view(), name="train-rbf"),
    path("predict/test/", PredictTestView.as_view(), name="predict-test"),
    path("predict/manual/", PredictManualView.as_view(), name="predict-manual"),
    path("health/", HealthCheckView.as_view(), name="health"),
]
