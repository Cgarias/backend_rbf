import os
import json
import joblib
import pandas as pd
import numpy as np
import gdown

from django.conf import settings
from django.http import JsonResponse, FileResponse
from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from sklearn.model_selection import train_test_split

from trainer.core.rbf_model import train_rbf_model, predict_rbf, predict_rbf_model

# Directorio donde guardamos datasets y modelos
MODEL_DIR = os.path.join(settings.BASE_DIR, "trainer", "models_data")
os.makedirs(MODEL_DIR, exist_ok=True)


class DatasetInfoView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        """
        Sube un archivo CSV (o descarga desde Drive si se proporciona URL) y guarda como last_dataset.csv.
        Devuelve info del dataset y 70% de entrenamiento (columns+values).
        """
        file = request.FILES.get("file")
        google_drive_url = request.data.get("google_drive_url")

        if not file and not google_drive_url:
            return JsonResponse({"error": "Debe proporcionar un archivo o una URL de Google Drive"}, status=400)

        try:
            dataset_path = os.path.join(MODEL_DIR, "last_dataset.csv")

            if file:
                # Guardar el archivo subido
                with open(dataset_path, "wb+") as dest:
                    for chunk in file.chunks():
                        dest.write(chunk)
                df = pd.read_csv(dataset_path)
            else:
                # Descargar desde Google Drive
                file_id = None
                if "id=" in google_drive_url:
                    file_id = google_drive_url.split("id=")[1].split("&")[0]
                elif "drive.google.com/file/d/" in google_drive_url:
                    file_id = google_drive_url.split("/d/")[1].split("/")[0]

                if not file_id:
                    return JsonResponse({"error": "URL de Drive no válida"}, status=400)

                gdown.download(f"https://drive.google.com/uc?id={file_id}", dataset_path, quiet=True)
                df = pd.read_csv(dataset_path)

            # Información básica
            num_patterns = int(df.shape[0])
            columns = df.columns.tolist()
            detected_target = columns[-1] if len(columns) > 1 else None
            input_columns = columns[:-1] if len(columns) > 1 else columns
            output_columns = [detected_target] if detected_target else []

            # Proveer 70% train sample to frontend (list of lists)
            df_train, _ = train_test_split(df, test_size=0.3, random_state=42)
            train_data = {"columns": df_train.columns.tolist(), "values": df_train.values.tolist()}

            dataset_info = {
                "patterns_total": num_patterns,
                "inputs_total": len(input_columns),
                "outputs_total": len(output_columns),
                "input_columns": input_columns,
                "output_columns": output_columns,
                "detected_target": detected_target,
                "saved_path": dataset_path,
                "format": "CSV",
                "train_data": train_data,
            }
            return JsonResponse(dataset_info, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


class TrainRBFView(APIView):
    def post(self, request, *args, **kwargs):
        """
        Usa last_dataset.csv guardado para entrenar.
        Solo usa columnas numéricas (si hay no-numéricas, se descartan).
        """
        try:
            num_centers = int(request.data.get("num_centers", 3))
            optimal_error = float(request.data.get("optimal_error", 0.1))

            dataset_path = os.path.join(MODEL_DIR, "last_dataset.csv")
            if not os.path.exists(dataset_path):
                return JsonResponse({"error": "No hay dataset cargado. Sube uno desde /dataset-info/."}, status=400)

            df = pd.read_csv(dataset_path)
            df_num = df.select_dtypes(include=[np.number])
            if df_num.shape[1] < 2:
                return JsonResponse({"error": "El dataset debe tener al menos 1 entrada y 1 salida numérica."}, status=400)

            X = df_num.iloc[:, :-1].values
            Y = df_num.iloc[:, -1].values.reshape(-1, 1)

            results = train_rbf_model(X, Y, num_centers=num_centers, optimal_error=optimal_error)

            # Ya que train_rbf_model guarda model_rbf.pkl en trainer/models_data, devolvemos results.
            return JsonResponse({"message": "Entrenamiento completado", "results": results}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


class PredictTestView(APIView):
    def get(self, request, *args, **kwargs):
        """
        Devuelve predicciones del 30% de prueba usando el modelo guardado.
        """
        try:
            model_path = os.path.join(MODEL_DIR, "model_rbf.pkl")
            dataset_path = os.path.join(MODEL_DIR, "last_dataset.csv")

            if not os.path.exists(model_path):
                return JsonResponse({"error": "No hay modelo entrenado."}, status=400)
            if not os.path.exists(dataset_path):
                return JsonResponse({"error": "No hay dataset cargado."}, status=400)

            # Cargamos el modelo (contiene centers, W, scaler, X_test, Y_test)
            model_bundle = joblib.load(model_path)
            centers = np.array(model_bundle["centers"])
            W = model_bundle["W"]
            scaler = model_bundle["scaler"]

            # Si X_test/Y_test fueron guardados en el bundle, los usamos directamente (recomendado)
            if "X_test" in model_bundle and "Y_test" in model_bundle:
                X_test = model_bundle["X_test"]
                Y_test = model_bundle["Y_test"]
            else:
                # fallback: recomputar split desde el dataset
                df = pd.read_csv(dataset_path).select_dtypes(include=[np.number])
                X = df.iloc[:, :-1].values
                Y = df.iloc[:, -1].values.reshape(-1, 1)
                X_scaled = scaler.transform(X)
                _, X_test, _, Y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

            Y_pred = predict_rbf(X_test, centers, W)

            predicciones = [
                {"inputs": X_test[i].tolist(), "Yd": float(Y_test[i]), "Yr": float(Y_pred[i])}
                for i in range(len(Y_pred))
            ]
            return JsonResponse({"predicciones": predicciones}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


class PredictManualView(APIView):
    def post(self, request, *args, **kwargs):
        """
        Recibe JSON { "inputs": [x1, x2, ...] } o { "inputs": [[...],[...]] }.
        Devuelve {"input": [...], "predicted_output": [...]} o error.
        """
        try:
            inputs = request.data.get("inputs")
            if inputs is None:
                return JsonResponse({"error": "No se enviaron valores de entrada."}, status=400)

            preds = predict_rbf_model(inputs)
            if "error" in preds:
                return JsonResponse(preds, status=400)
            return JsonResponse(preds, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)


class HealthCheckView(APIView):
    def get(self, request, *args, **kwargs):
        return JsonResponse({"status": "ok", "message": "Servidor funcionando correctamente 🚀"})
