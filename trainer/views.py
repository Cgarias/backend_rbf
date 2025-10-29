from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Importar el modelo base RBF
from .core.rbf_model import train_rbf_model, rbf_activation

# Variable global para mantener el último modelo entrenado en memoria
LAST_MODEL = {}

# -------------------------------------------------------------
# 🧩 1️⃣ PREDICCIÓN (SIMULACIÓN)
# -------------------------------------------------------------
@api_view(["POST"])
def predict_rbf_view(request):
    """
    Endpoint para realizar simulaciones (predicciones) con el modelo RBF ya entrenado.
    """
    global LAST_MODEL
    try:
        if not LAST_MODEL:
            return Response(
                {"error": "No hay modelo entrenado aún. Entrena uno primero."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        data = request.data
        inputs = data.get("inputs", None)

        if inputs is None or not isinstance(inputs, list):
            return Response(
                {"error": "Debes enviar una lista numérica en 'inputs'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # --- Recuperar modelo ---
        centers = np.array(LAST_MODEL["weights"]["Centers_R"])
        W0 = LAST_MODEL["weights"]["W0_Umbral"]
        W_centers = np.array(LAST_MODEL["weights"]["W_Centers"]).reshape(-1, 1)
        scaler = LAST_MODEL["scaler"]

        # --- Calcular distancias y activaciones ---
        X = np.array([inputs])
        X_scaled = scaler.transform(X)
        num_centers = centers.shape[0]

        distances = np.zeros((1, num_centers))
        for j in range(num_centers):
            distances[0, j] = np.linalg.norm(X_scaled[0, :] - centers[j, :])

        FA_matrix = rbf_activation(distances)
        A = np.hstack((np.ones((1, 1)), FA_matrix))
        Yr = A.dot(np.vstack(([W0], W_centers)))

        return Response({"inputs": inputs, "Yr": Yr.flatten().tolist()})

    except Exception as e:
        print("❌ Error en predict:", e)
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# -------------------------------------------------------------
# 🧩 2️⃣ ENTRENAMIENTO
# -------------------------------------------------------------
class TrainRBFView(APIView):
    """
    Entrena un modelo RBF con autodetección de variable objetivo.
    - Si no se envía 'target', el sistema la detecta automáticamente.
    - Codifica texto a números (LabelEncoder).
    - Devuelve métricas, pesos, configuración y mapeos de etiquetas.
    """

    def post(self, request):
        global LAST_MODEL
        try:
            file = request.FILES.get("file")
            num_centers = int(request.data.get("num_centers", 3))
            optimal_error = float(request.data.get("optimal_error", 0.1))
            user_target = request.data.get("target", None)

            if not file:
                return Response(
                    {"error": "Debe enviar un archivo CSV, XLSX o JSON."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            # --- 1. Leer dataset ---
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
            elif file.name.endswith(".json"):
                df = pd.read_json(file)
            else:
                return Response(
                    {"error": "Formato no soportado (solo CSV, XLSX o JSON)."},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if df.empty:
                return Response({"error": "El archivo está vacío."}, status=400)

            # --- 2. Detectar automáticamente la columna target ---
            if user_target and user_target in df.columns:
                target = user_target
            else:
                nunique = df.nunique()
                likely_target = nunique.idxmin()
                if nunique[likely_target] <= 10 or df.columns[-1] == likely_target:
                    target = likely_target
                else:
                    target = df.columns[-1]

            # --- 3. Codificar columnas categóricas ---
            label_mappings = {}
            for col in df.columns:
                if df[col].dtype == "object" or df[col].dtype == "string":
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    label_mappings[col] = dict(
                        zip(encoder.classes_, encoder.transform(encoder.classes_))
                    )

            # --- 4. Convertir a numérico y limpiar ---
            df = df.apply(pd.to_numeric, errors="coerce").dropna()

            if target not in df.columns:
                return Response(
                {"error": f"No se pudo identificar una columna objetivo válida."},
                status=400,
            )

            # 🔹 Información del dataset para el frontend
            dataset_info = {
               "patterns_total": df.shape[0],
               "input_columns": [col for col in df.columns if col != target],
               "output_column": target
            }

            X = df.drop(columns=[target]).values
            Y = df[[target]].values

            if X.shape[0] < 3 or X.shape[1] < 1:
               return Response(
                   {"error": "El dataset es demasiado pequeño para entrenar."},
                   status=400,
               )

            # --- 5. Entrenamiento RBF ---
            results = train_rbf_model(
                X, Y, num_centers=num_centers, optimal_error=optimal_error
            )

            # --- 6. Guardar modelo para predicción posterior ---
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
            LAST_MODEL.clear()
            LAST_MODEL.update(
              {
                  "weights": results["weights"],
                  "scaler": scaler,
              }
            )

            # --- 7. Agregar metadata adicional ---
            results["detected_target"] = target
            results["dataset_info"] = dataset_info  # 🔹 Añadido para frontend
            if label_mappings:
               results["label_mappings"] = label_mappings

            return Response(results, status=status.HTTP_200_OK)


            # --- 5. Entrenamiento RBF ---
            results = train_rbf_model(
                X, Y, num_centers=num_centers, optimal_error=optimal_error
            )

            # --- 6. Guardar modelo para predicción posterior ---
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
            LAST_MODEL.clear()
            LAST_MODEL.update(
                {
                    "weights": results["weights"],
                    "scaler": scaler,
                }
            )

            # --- 7. Agregar metadata adicional ---
            results["detected_target"] = target
            if label_mappings:
                results["label_mappings"] = label_mappings

            return Response(results, status=status.HTTP_200_OK)

        except Exception as e:
            print("❌ Error en entrenamiento:", e)
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
