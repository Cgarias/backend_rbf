# En trainer/urls.py
from django.urls import path
from trainer.views import (
    DatasetInfoView,
    TrainRBFView,
    PredictTestView,
    PredictManualView,
    HealthCheckView,
)

urlpatterns = [
    # POST /api/dataset-info/ (Para subir y crear)
    path("dataset-info/", DatasetInfoView.as_view(), name="dataset-info"),
    
    # POST /api/dataset/<int:dataset_id>/train/
    path("dataset/<int:dataset_id>/train/", TrainRBFView.as_view(), name="train-rbf"),
    
    # GET /api/model/<int:model_id>/predict/test/
    path("model/<int:model_id>/predict/test/", PredictTestView.as_view(), name="predict-test"),
    
    # POST /api/model/<int:model_id>/predict/manual/
    path("model/<int:model_id>/predict/manual/", PredictManualView.as_view(), name="predict-manual"),
    
    # GET /api/health/
    path("health/", HealthCheckView.as_view(), name="health"),
]