# trainer/views.py

import joblib
import numpy as np
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from sklearn.model_selection import train_test_split

from trainer.core.rbf_model import predict_rbf # Necesario para subida de archivos

# --- Imports de la refactorización ---
from .serializers import DatasetUploadSerializer  # Debes crear este serializer
from .services import create_dataset_from_upload  # Debes crear este servicio
# -------------------------------------

# --- Imports para las OTRAS vistas ---
from .models import Dataset, TrainedModel
from .serializers import TrainRBFSerializer, ManualPredictSerializer
from .services import train_model_from_dataset
from .ml.prediction import predict_from_file
import os # (Si es necesario)


# --- ASEGÚRATE DE QUE ESTA CLASE EXISTA ---
class DatasetInfoView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        """
        Sube un archivo (o URL de Drive) usando el Serializer
        y crea un objeto Dataset usando el Service.
        """
        # 1. Validar la entrada
        serializer = DatasetUploadSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            validated_data = serializer.validated_data
            
            # 2. Delegar la lógica al servicio
            dataset_obj, dataset_info = create_dataset_from_upload(
                file=validated_data.get('file'),
                google_drive_url=validated_data.get('google_drive_url')
            )
            
            # 3. Devolver la info (incluyendo el nuevo ID)
            return Response(dataset_info, status=status.HTTP_201_CREATED)
        
        except Exception as e:
            # En producción, loggear el error 'e'
            return Response({"error": f"Error al procesar el dataset: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class TrainRBFView(APIView):
    def post(self, request, dataset_id, *args, **kwargs): # Asumiendo que cambiaste la URL a /dataset/<int:dataset_id>/train/
        try:
            dataset = Dataset.objects.get(pk=dataset_id)
        except Dataset.DoesNotExist:
            return Response({"error": "Dataset no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        serializer = TrainRBFSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            validated_data = serializer.validated_data
            trained_model_obj = train_model_from_dataset(
                dataset=dataset,
                num_centers=validated_data.get('num_centers'),
                optimal_error=validated_data.get('optimal_error')
            )
            return Response(
                {
                    "message": "Entrenamiento completado",
                    "model_id": trained_model_obj.id,
                    "metrics": trained_model_obj.metrics
                },
                status=status.HTTP_201_CREATED
            )
        except Exception as e:
            return Response({"error": f"Error interno durante el entrenamiento: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictTestView(APIView):
    def get(self, request, model_id, *args, **kwargs):
        """
        Recalcula el split de prueba para obtener los inputs originales
        y devuelve las predicciones de prueba decodificadas.
        """
        try:
            model = TrainedModel.objects.get(pk=model_id)
        except TrainedModel.DoesNotExist:
            return Response({"error": "Modelo no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        try:
            # 1. Cargar el dataset original
            df = pd.read_csv(model.dataset.file.path)

            # 2. Cargar el bundle .pkl (que tiene el scaler, W, centers, maps)
            model_bundle = joblib.load(model.model_file.path)
            
            # 3. Cargar artefactos del modelo
            centers = np.array(model_bundle["centers"])
            W = np.array(model_bundle["W"])
            scaler = model_bundle["scaler"]
            
            # 4. Cargar mapas de codificación
            if "data_mappings" not in model_bundle:
                 # Compatibilidad con modelos antiguos que guardaban X_test
                if "X_test" in model_bundle:
                    return Response({"error": "Este modelo es antiguo. Intente /predict/test/ con un modelo re-entrenado."}, status=400)
                raise ValueError("Model bundle corrupto, no contiene 'data_mappings'.")

            mappings = model_bundle["data_mappings"]
            processed_cols = mappings.get("processed_input_columns")
            output_map = mappings.get("output_map")

            # 5. Re-crear la división de prueba (70/30)
            X_df = df.iloc[:, :-1]
            Y_s = df.iloc[:, -1] # Y como Serie de pandas

            # Dividimos los datos originales
            _, X_test_df, _, Y_test_s = train_test_split(
                X_df, Y_s, test_size=0.3, random_state=42
            )

            # 6. Procesar X_test_df para la predicción (igual que en predict_manual)
            X_processed_df = pd.get_dummies(X_test_df, dtype=float)
            X_reindexed_df = X_processed_df.reindex(columns=processed_cols, fill_value=0.0)
            X_scaled = scaler.transform(X_reindexed_df)

            # 7. Realizar la predicción
            Y_pred_numeric = predict_rbf(X_scaled, centers, W).flatten()

            # 8. Formatear la respuesta
            predicciones = []
            
            # Convertimos X_test_df a lista de listas para los inputs
            X_test_inputs_list = X_test_df.values.tolist()
            
            for i in range(len(X_test_inputs_list)):
                inputs_reales = X_test_inputs_list[i]
                Yd_real = Y_test_s.iloc[i] # Valor original (texto o num)
                Yr_numeric = Y_pred_numeric[i]
                
                if output_map:
                    # Es clasificación
                    Yr_label = output_map.get(str(int(np.round(Yr_numeric))), f"Pred. Num. {Yr_numeric:.2f}")
                    
                    predicciones.append({
                        "inputs": inputs_reales, # <-- INPUTS REALES
                        "Yd": Yd_real,
                        "Yr": Yr_label
                    })
                else:
                    # Es regresión
                    predicciones.append({
                        "inputs": inputs_reales, # <-- INPUTS REALES
                        "Yd": float(Yd_real),
                        "Yr": float(Yr_numeric)
                    })
            
            return Response({"predicciones": predicciones}, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({"error": f"Error al procesar la predicción de prueba: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class PredictManualView(APIView):
    def post(self, request, model_id, *args, **kwargs):
        """
        Recibe JSON { "inputs": [[x1, "texto", x3...]] }.
        Usa el 'model_bundle' para codificar, escalar, predecir y decodificar.
        """
        try:
            model = TrainedModel.objects.get(pk=model_id)
        except TrainedModel.DoesNotExist:
            return Response({"error": "Modelo no encontrado"}, status=status.HTTP_404_NOT_FOUND)

        serializer = ManualPredictSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        try:
            inputs = serializer.validated_data.get('inputs')
            
            # Usamos el path del FileField del modelo.
            # La función predict_from_file ahora hace todo el pipeline.
            preds = predict_from_file(model.model_file.path, inputs)
            
            if "error" in preds:
                 return Response(preds, status=status.HTTP_400_BAD_REQUEST)
            return Response(preds, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class HealthCheckView(APIView):
    def get(self, request, *args, **kwargs):
        return Response({"status": "ok", "message": "Servidor funcionando correctamente 🚀"})