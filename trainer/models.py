from django.db import models
# from django.contrib.auth.models import User # (Opcional: para saber qué usuario lo subió)

class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='datasets/') # Se guarda en MEDIA_ROOT/datasets/
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # owner = models.ForeignKey(User, on_delete=models.CASCADE)

class TrainedModel(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name="models")
    model_file = models.FileField(upload_to='models/') # Se guarda en MEDIA_ROOT/models/
    metrics = models.JSONField() # Almacena aquí los resultados (EG, MAE, etc.)
    config = models.JSONField() # Almacena los hiperparámetros (num_centers, etc.)
    created_at = models.DateTimeField(auto_now_add=True)
    data_mappings = models.JSONField(null=True, blank=True)
    # owner = models.ForeignKey(User, on_delete=models.CASCADE)