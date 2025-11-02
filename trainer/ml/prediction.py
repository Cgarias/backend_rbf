# En trainer/ml/prediction.py
import os
import joblib
import numpy as np
import pandas as pd
from .core import predict_rbf # La función de predicción pura

def predict_from_file(model_path, new_data):
    """
    Predicción manual usando un archivo de modelo específico.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError("No se encontró el archivo del modelo.")

    model_bundle = joblib.load(model_path)
    
    return predict_from_bundle(model_bundle, new_data)


def predict_from_bundle(model_bundle, new_data_list):
    """
    Predice usando el bundle cargado y los nuevos datos de entrada.
    Maneja la codificación y decodificación.
    """
    
    # 1. Cargar artefactos del modelo
    centers = np.array(model_bundle["centers"])
    W = np.array(model_bundle["W"])
    scaler = model_bundle["scaler"]
    
    # 2. Cargar los mapas de codificación
    if "data_mappings" not in model_bundle:
        raise ValueError("Model bundle está corrupto, no contiene 'data_mappings'.")
        
    mappings = model_bundle["data_mappings"]
    original_cols = mappings.get("original_input_columns")
    processed_cols = mappings.get("processed_input_columns")
    output_map = mappings.get("output_map")

    if not original_cols or not processed_cols:
        raise ValueError("Data mappings está incompleto (faltan columnas de entrada).")

    # 3. Convertir la entrada (lista de listas) a DataFrame
    #    con las columnas *originales*
    try:
        X_new_df = pd.DataFrame(new_data_list, columns=original_cols)
    except ValueError as e:
        msg = f"Error al crear DataFrame. ¿El número de entradas ({len(new_data_list[0])}) coincide con las del entrenamiento ({len(original_cols)})?"
        raise ValueError(msg)

    # 4. Aplicar One-Hot Encoding
    X_processed_df = pd.get_dummies(X_new_df, dtype=float)
    
    # 5. Reindexar para que coincida con las columnas EXACTAS de entrenamiento
    #    Esto añade columnas faltantes (ej. 'Raza_Gyr') con 0s
    #    y elimina columnas nuevas que no estaban en el entrenamiento.
    X_reindexed_df = X_processed_df.reindex(columns=processed_cols, fill_value=0.0)

    # 6. Escalar los datos
    X_scaled = scaler.transform(X_reindexed_df)

    # 7. Predecir (esto dará una salida numérica)
    Y_pred_numeric = predict_rbf(X_scaled, centers, W)

    # 8. Decodificar la salida
    final_predictions = []
    if output_map:
        # Es Clasificación: redondear y buscar en el mapa
        for p in Y_pred_numeric.flatten():
            # Redondear al entero más cercano y convertir a string
            pred_code = str(int(np.round(p)))
            # Buscar en el mapa
            pred_label = output_map.get(pred_code, f"Pred. Numérica: {p:.2f} (Código '{pred_code}' no encontrado)")
            final_predictions.append(pred_label)
    else:
        # Es Regresión: devolver el número tal cual
        final_predictions = [float(p) for p in Y_pred_numeric.flatten()]
        
    return {"input": new_data_list, "predicted_output": final_predictions}