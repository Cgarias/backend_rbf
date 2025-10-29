import numpy as np
import matplotlib.pyplot as plt
import json
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ===============================================================
# 🔹 Función de Activación según apuntes FA(d) = d² * ln(d)
# ===============================================================
def rbf_activation(distances):
    with np.errstate(divide='ignore', invalid='ignore'):
        activations = (distances ** 2) * np.log(distances)
    activations = np.nan_to_num(activations, nan=0.0)
    return activations


# ===============================================================
# 🔹 Métricas de rendimiento
# ===============================================================
def calculate_metrics(y_true, y_pred, optimal_error):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    errors = y_true - y_pred
    eg = np.sum(np.abs(errors)) / len(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    converges = bool(eg <= optimal_error)

    return {
        "EG": round(float(eg), 6),
        "MAE": round(float(mae), 6),
        "RMSE": round(float(rmse), 6),
        "converged": converges
    }


# ===============================================================
# 🔹 Predicción (Simulación) - CORREGIDA
# ===============================================================
# Esta función ahora espera datos YA ESCALADOS (X_scaled)
def predict_rbf(X_scaled, centers, W):
    num_patterns = X_scaled.shape[0]
    num_centers = centers.shape[0]

    distances = np.zeros((num_patterns, num_centers))
    for i in range(num_patterns):
        for j in range(num_centers):
            # Usamos X_scaled directamente
            distances[i, j] = np.linalg.norm(X_scaled[i, :] - centers[j, :])

    FA_matrix = rbf_activation(distances)
    A = np.hstack((np.ones((num_patterns, 1)), FA_matrix))
    Y_r = A.dot(W)
    return Y_r


# ===============================================================
# 🔹 Entrenamiento principal - CORREGIDO
# ===============================================================
def train_rbf_model(X, Y, num_centers, optimal_error, test_split_ratio=0.3):
    # Crear carpeta de resultados si no existe
    os.makedirs("results", exist_ok=True)

    # 1️⃣ Escalamiento
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # 2️⃣ División de datos
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=test_split_ratio, random_state=42
    )
    N_train = X_train.shape[0]

    # 3️⃣ Inicialización de centros
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)
    centers = np.random.uniform(low=min_vals, high=max_vals, size=(num_centers, X_train.shape[1]))

    # 4️⃣ Cálculo de distancias y matriz A
    distances_train = np.zeros((N_train, num_centers))
    for i in range(N_train):
        for j in range(num_centers):
            distances_train[i, j] = np.linalg.norm(X_train[i, :] - centers[j, :])

    FA_matrix_train = rbf_activation(distances_train)
    A_train = np.hstack((np.ones((N_train, 1)), FA_matrix_train))

    # 5️⃣ Pesos (W) por pseudoinversa
    try:
        W = np.linalg.pinv(A_train).dot(Y_train)
    except np.linalg.LinAlgError:
        return {"error": "Error de matriz singular. Entrenamiento fallido."}

    # 6️⃣ Simulación - CORREGIDA
    # Llamamos a la función sin el 'scaler'
    Y_r_train = predict_rbf(X_train, centers, W)
    Y_r_test = predict_rbf(X_test, centers, W)

    # 7️⃣ Métricas
    metrics_train = calculate_metrics(Y_train, Y_r_train, optimal_error)
    metrics_test = calculate_metrics(Y_test, Y_r_test, optimal_error)

    # 8️⃣ Datos para graficar en el frontend
    train_graph = {
        "Yd": Y_train.flatten().tolist(),
        "Yr": Y_r_train.flatten().tolist(),
    }

    test_graph = {
        "Yd": Y_test.flatten().tolist(),
        "Yr": Y_r_test.flatten().tolist(),
    }


    # Guardar resultados
    results_data = {
        "config": {
            "num_centers": num_centers,
            "optimal_error": optimal_error,
            "patterns_total": len(X),
            "patterns_train": len(X_train),
            "patterns_test": len(X_test),
        },
        "weights": {
            # Asegurarse de que W0 sea un float nativo de Python para JSON
            "W0_Umbral": float(W[0][0]), 
            "W_Centers": W[1:].flatten().tolist(),
            "Centers_R": centers.tolist(),
        },
        "metrics": {"train": metrics_train, "test": metrics_test},
        "graphs": {
            "train": train_graph,
            "test": test_graph
        }
    }


    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    # Guardar modelo completo
    # Guardamos el 'scaler' para futuras predicciones con datos NUEVOS
    joblib.dump(
        {"centers": centers, "weights": W, "scaler": scaler},
        "results/model.pkl"
    )

    return results_data