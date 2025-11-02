from rest_framework import serializers

class DatasetUploadSerializer(serializers.Serializer):
    file = serializers.FileField(required=False)
    google_drive_url = serializers.URLField(required=False)

    def validate(self, data):
        if not data.get('file') and not data.get('google_drive_url'):
            raise serializers.ValidationError("Debe proporcionar 'file' o 'google_drive_url'.")
        return data

class TrainRBFSerializer(serializers.Serializer):
    num_centers = serializers.IntegerField(default=3, min_value=1)
    optimal_error = serializers.FloatField(default=0.1, min_value=0.0)
    # Ya no necesitamos el dataset_id aquí si usamos URLs anidadas

class ManualPredictSerializer(serializers.Serializer):
    inputs = serializers.ListField(
        child=serializers.ListField(
            child=serializers.JSONField(), # <-- Permite str, int, float
            min_length=1  
        ),
        min_length=1 
    )
    # Nota: esto valida [[1.0, 2.0]]. 
    # Si quieres aceptar [1.0, 2.0], la lógica del serializer es más compleja.