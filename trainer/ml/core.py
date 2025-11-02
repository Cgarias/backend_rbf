# trainer/ml/core.py

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

# NO DEBE HABER MÁS "from ." IMPORTS AQUÍ

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
    """
    Calcula las métricas de error.
    """
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
    """
    num_patterns = X_scaled.shape[0]
    num_centers = centers.shape[0]
    distances = np.zeros((num_patterns, num_centers))

    # Calcula la matriz de distancias
    for i in range(num_patterns):
        for j in range(num_centers):
            distances[i, j] = np.linalg.norm(X_scaled[i, :] - centers[j, :])
    
    # Calcula la matriz de activación
    FA_matrix = rbf_activation(distances)
    
    # Añade la columna de bias (unos)
    A = np.hstack((np.ones((num_patterns, 1)), FA_matrix))
    
    # Realiza la predicción (producto punto)
    # A.dot(W) funciona para W shape (n_centers+1, 1) o (n_centers+1,)
    Y_r = A.dot(W)
    
    return Y_r