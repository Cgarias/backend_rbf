# trainer/ml/training.py

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from .core import rbf_activation, calculate_metrics, predict_rbf # Funciones auxiliares

# NOTA: Ya no importa 'joblib' ni 'os'. No guarda nada.


def train_rbf(X, Y, num_centers=3, optimal_error=0.1, test_split_ratio=0.3):
    """
    Entrena RBF y DEVUELVE el bundle del modelo y los resultados,
    PERO NO GUARDA NADA EN DISCO.
    """
    
    # --- INICIO DE LA LÓGICA DE ENTRENAMIENTO FALTANTE ---

    # Forzar arrays numpy (ya hecho en services.py, pero por seguridad)
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)

    # 1. Escalamiento
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    # 2. División
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_scaled, Y, test_size=test_split_ratio, random_state=42
    )

    # 3. Inicializar centros aleatorios dentro del rango de X_train
    min_vals = np.min(X_train, axis=0)
    max_vals = np.max(X_train, axis=0)
    centers = np.random.uniform(low=min_vals, high=max_vals, size=(num_centers, X_train.shape[1]))

    # 4. Distancias (train)
    distances_train = np.zeros((X_train.shape[0], num_centers))
    for i in range(X_train.shape[0]):
        for j in range(num_centers):
            distances_train[i, j] = np.linalg.norm(X_train[i, :] - centers[j, :])

    # 5. Matriz A_train
    A_train = np.hstack((np.ones((X_train.shape[0], 1)), rbf_activation(distances_train)))

    # 6. W por pseudoinversa (A_train^+ * Y_train)
    try:
        W = np.linalg.pinv(A_train).dot(Y_train)
    except Exception as e:
        # Devolvemos un error en lugar de crashear
        raise ValueError(f"Fallo al calcular pesos W: {str(e)}")

    # 7. Predicciones
    Y_train_pred = predict_rbf(X_train, centers, W)
    Y_test_pred = predict_rbf(X_test, centers, W)

    # 8. Métricas
    metrics_train = calculate_metrics(Y_train, Y_train_pred, optimal_error)
    metrics_test = calculate_metrics(Y_test, Y_test_pred, optimal_error)

    # --- FIN DE LA LÓGICA DE ENTRENAMIENTO FALTANTE ---


    # 1. Estructura de resultados (dict)
    results_data = {
        "config": {
            "num_centers": int(num_centers),
            "optimal_error": float(optimal_error),
            "patterns_total": int(len(X)),
            "patterns_train": int(len(X_train)),
            "patterns_test": int(len(X_test)),
        },
        "weights": {
            # Aseguramos que W[0] sea un float simple
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

    # 2. Artefacto/Bundle del modelo (dict)
    #    Contiene solo lo necesario para predecir.
    model_bundle = {
        "centers": centers,
        "W": W,
        "scaler": scaler,
    }

    # Devuelve ambos
    return model_bundle, results_data