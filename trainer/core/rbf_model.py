import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Path para guardar modelos/resultados (carpeta trainer/models_data)
BASE_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_data")
os.makedirs(BASE_MODEL_DIR, exist_ok=True)


def rbf_activation(distances):
    """
    Activación RBF de la forma FA(d) = d^2 * ln(d) (con manejo de 0/NaN).
    distances: array (n_samples, n_centers) de distancias.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        activations = (distances ** 2) * np.log(distances)
        activations = np.nan_to_num(activations, nan=0.0, posinf=0.0, neginf=0.0)
    return activations


def calculate_metrics(y_true, y_pred, optimal_error=0.1):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    eg = float(np.mean(np.abs(y_true - y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    converges = bool(eg <= optimal_error)
    return {"EG": eg, "MAE": mae, "RMSE": rmse, "converged": converges}


def predict_rbf(X_scaled, centers, W):
    """
    Predice a partir de X escalado, centros y W (vector de pesos incluyendo bias en W[0]).
    X_scaled: np.array shape (n_samples, n_features)
    centers: np.array shape (n_centers, n_features)
    W: np.array shape (n_centers+1, 1) o (n_centers+1,)
    """
    num_patterns = X_scaled.shape[0]
    num_centers = centers.shape[0]
    distances = np.zeros((num_patterns, num_centers))
    for i in range(num_patterns):
        for j in range(num_centers):
            distances[i, j] = np.linalg.norm(X_scaled[i, :] - centers[j, :])
    FA_matrix = rbf_activation(distances)
    A = np.hstack((np.ones((num_patterns, 1)), FA_matrix))
    # A.dot(W) works for W shape (n_centers+1, 1) or (n_centers+1,)
    Y_r = A.dot(W)
    return Y_r


def train_rbf_model(X, Y, num_centers=3, optimal_error=0.1, test_split_ratio=0.3):
    """
    Entrena RBF sencillo:
      - escala con MinMaxScaler
      - inicializa centros aleatorios dentro de rango de X_train
      - calcula la matriz A con rbf_activation
      - calcula W por pseudoinversa
    Guarda:
      - results JSON: results.json
      - modelo binario joblib: model_rbf.pkl (contiene centers, W, scaler, X_test, Y_test)
    Devuelve results_data (dict) para enviar al frontend.
    """
    os.makedirs(BASE_MODEL_DIR, exist_ok=True)

    # Forzar arrays numpy
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)

    # Escalamiento
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # División
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=test_split_ratio, random_state=42
    )

    # Inicializar centros aleatorios dentro del rango de X_train
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)
    centers = np.random.uniform(low=min_vals, high=max_vals, size=(num_centers, X_train.shape[1]))

    # Distancias (train)
    distances_train = np.zeros((X_train.shape[0], num_centers))
    for i in range(X_train.shape[0]):
        for j in range(num_centers):
            distances_train[i, j] = np.linalg.norm(X_train[i, :] - centers[j, :])

    A_train = np.hstack((np.ones((X_train.shape[0], 1)), rbf_activation(distances_train)))

    # W por pseudoinversa (A_train^+ * Y_train)
    try:
        W = np.linalg.pinv(A_train).dot(Y_train)
    except Exception as e:
        return {"error": f"Fallo al calcular pesos W: {str(e)}"}

    # Predicciones
    Y_train_pred = predict_rbf(X_train, centers, W)
    Y_test_pred = predict_rbf(X_test, centers, W)

    # Métricas
    metrics_train = calculate_metrics(Y_train, Y_train_pred, optimal_error)
    metrics_test = calculate_metrics(Y_test, Y_test_pred, optimal_error)

    # Estructura de resultados
    results_data = {
        "config": {
            "num_centers": int(num_centers),
            "optimal_error": float(optimal_error),
            "patterns_total": int(len(X)),
            "patterns_train": int(len(X_train)),
            "patterns_test": int(len(X_test)),
        },
        "weights": {
            "W0_Umbral": float(W[0][0]) if W.ndim > 1 else float(W[0]),
            "W_Centers": W[1:].flatten().tolist(),
            "Centers_R": centers.tolist(),
        },
        "metrics": {"train": metrics_train, "test": metrics_test},
        "graphs": {
            "train": {"Yd": Y_train.flatten().tolist(), "Yr": Y_train_pred.flatten().tolist()},
            "test": {"Yd": Y_test.flatten().tolist(), "Yr": Y_test_pred.flatten().tolist()},
        },
    }

    # Guardar results.json
    results_json_path = os.path.join(BASE_MODEL_DIR, "results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    # Guardar modelo binario con todo lo necesario para predecir
    model_bundle = {
        "centers": centers,
        "W": W,
        "scaler": scaler,
        "X_test": X_test,
        "Y_test": Y_test,
    }
    model_path = os.path.join(BASE_MODEL_DIR, "model_rbf.pkl")
    joblib.dump(model_bundle, model_path)

    return results_data


def predict_rbf_model(new_data):
    """
    Predicción manual usando el modelo guardado en trainer/models_data/model_rbf.pkl
    new_data: lista o array (1D) o lista de listas (n_samples x n_features)
    """
    model_path = os.path.join(BASE_MODEL_DIR, "model_rbf.pkl")
    if not os.path.exists(model_path):
        return {"error": "No existe un modelo entrenado. Entrena primero con /train-rbf/"}

    model_bundle = joblib.load(model_path)
    centers = np.array(model_bundle["centers"])
    W = model_bundle["W"]
    scaler = model_bundle["scaler"]

    X_new = np.array(new_data, dtype=float)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    # Escalar según el scaler entrenado
    try:
        X_scaled = scaler.transform(X_new)
    except Exception as e:
        return {"error": f"Error al escalar entradas: {str(e)}"}

    Y_pred = predict_rbf(X_scaled, centers, W)

    return {"input": X_new.tolist(), "predicted_output": [float(x) for x in Y_pred.flatten()]}
