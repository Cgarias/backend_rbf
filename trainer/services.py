import os
import uuid
import numpy as np
import pandas as pd
import gdown
from django.core.files.base import ContentFile
from sklearn.model_selection import train_test_split
from .models import Dataset, TrainedModel
from .ml import training, prediction # Módulo ML refactorizado
from django.conf import settings
import joblib

def download_from_gdrive(google_drive_url):
    """
    Extrae el file_id de una URL de Google Drive y descarga el archivo.
    """
    file_id = None
    if "id=" in google_drive_url:
        file_id = google_drive_url.split("id=")[1].split("&")[0]
    elif "drive.google.com/file/d/" in google_drive_url:
        file_id = google_drive_url.split("/d/")[1].split("/")[0]

    if not file_id:
        raise ValueError("URL de Drive no válida")
        
    output_path = os.path.join(settings.MEDIA_ROOT, 'temp_gdrive_download.csv')
    # Asegúrate de que el directorio temporal exista (si MEDIA_ROOT es 'media/')
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=True)
    return output_path

def create_dataset_from_upload(file=None, google_drive_url=None):
    """
    Crea un objeto Dataset en la BBDD a partir de un archivo o URL.
    Devuelve el objeto Dataset y un dict con su info.
    """
    if file:
        dataset = Dataset(name=file.name)
        # Django maneja la creación de carpetas (datasets/) aquí
        dataset.file.save(file.name, file, save=True) 
        df = pd.read_csv(dataset.file.path)
    
    elif google_drive_url:
        temp_path = download_from_gdrive(google_drive_url)
        dataset = Dataset(name=os.path.basename(temp_path).replace('temp_gdrive_download', 'gdrive_dataset'))
        with open(temp_path, 'rb') as f:
            dataset.file.save('gdrive_dataset.csv', ContentFile(f.read()), save=True)
        os.remove(temp_path) # Limpia el archivo temporal
        df = pd.read_csv(dataset.file.path)
    else:
        # Esto no debería pasar si el serializer funciona, pero es buena defensa
        raise ValueError("No se proporcionó ni archivo ni URL.")

    # --- LÓGICA DE DF FALTANTE ---
    num_patterns = int(df.shape[0])
    columns = df.columns.tolist()
    detected_target = columns[-1] if len(columns) > 1 else None
    input_columns = columns[:-1] if len(columns) > 1 else columns
    output_columns = [detected_target] if detected_target else []

    # Proveer 70% train sample (opcional, pero estaba en tu código original)
    df_train, _ = train_test_split(df, test_size=0.3, random_state=42)
    train_data = {"columns": df_train.columns.tolist(), "values": df_train.values.tolist()}
    # ---------------------------

    dataset_info = {
        "dataset_id": dataset.id,
        "patterns_total": num_patterns,
        "inputs_total": len(input_columns),
        "outputs_total": len(output_columns),
        "input_columns": input_columns,
        "output_columns": output_columns,
        "detected_target": detected_target,
        "saved_path": dataset.file.url, # Devuelve la URL (ej. /media/datasets/...)
        "format": "CSV",
        "train_data": train_data, # Devuelve la muestra
    }
    return dataset, dataset_info

def train_model_from_dataset(dataset, num_centers, optimal_error):
    """
    Orquesta el entrenamiento, AHORA CON CODIFICACIÓN AUTOMÁTICA.
    """
    df = pd.read_csv(dataset.file.path)
    
    if df.shape[1] < 2:
        raise ValueError("El dataset debe tener al menos 1 entrada y 1 salida.")

    # --- INICIO DE LÓGICA DE CODIFICACIÓN ---
    
    # 1. Separar Entradas (X) y Salida (Y)
    X_df = df.iloc[:, :-1]
    Y_df = df.iloc[:, -1]
    
    data_mappings = {} # Mapa para guardar

    # 2. Guardar nombres de columnas de entrada originales
    data_mappings["original_input_columns"] = X_df.columns.tolist()

    # 3. Codificar Entradas X (One-Hot Encoding para texto)
    # pd.get_dummies convierte 'Raza' en 'Raza_Gyr' y 'Raza_Girolando'
    X_processed_df = pd.get_dummies(X_df, dtype=float)
    data_mappings["processed_input_columns"] = X_processed_df.columns.tolist()
    
    # Convertir a array de NumPy para el modelo
    X_data = X_processed_df.values

    # 4. Codificar Salida Y (Label Encoding para texto)
    if pd.api.types.is_object_dtype(Y_df):
        # Es texto (Clasificación). Usamos factorize.
        # "Crítico" -> 0, "Alerta" -> 1, "Estable" -> 2
        Y_codes, Y_labels = pd.factorize(Y_df)
        Y_data = Y_codes.reshape(-1, 1)
        # Guardamos el mapa para decodificar: {"0": "Crítico", "1": "Alerta", ...}
        # Usamos str(i) como llave para compatibilidad JSON
        data_mappings["output_map"] = {str(i): label for i, label in enumerate(Y_labels)}
    else:
        # Es numérico (Regresión). Lo usamos tal cual.
        Y_data = Y_df.values.reshape(-1, 1)
        data_mappings["output_map"] = None # No hay mapa

    # --- FIN DE LÓGICA DE CODIFICACIÓN ---

    # 5. Llamar a la lógica de ML (con los datos ya procesados)
    #    train_rbf (en ml/training.py) ya se encarga de escalar X_data
    model_bundle, results_data = training.train_rbf(
        X=X_data,      # <-- Usamos X procesado
        Y=Y_data,      # <-- Usamos Y procesado
        num_centers=num_centers, 
        optimal_error=optimal_error
    )
    
    # 6. Añadir los mapas al 'model_bundle' para que se guarden en el .pkl
    #    Esto es VITAL para que la predicción funcione.
    model_bundle["data_mappings"] = data_mappings

    # 7. Guardar artefacto del modelo
    model_filename = f"rbf_model_{dataset.id}_{uuid.uuid4().hex[:6]}.pkl"
    model_path = os.path.join(settings.MEDIA_ROOT, 'models', model_filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_bundle, model_path)

    # 8. Crear el objeto en BBDD
    trained_model = TrainedModel.objects.create(
        dataset=dataset,
        model_file=f"models/{model_filename}", # Ruta relativa
        metrics=results_data.get("metrics", {}),
        config=results_data.get("config", {}),
        data_mappings=data_mappings  # <-- Guardamos los mapas en la BBDD
    )
    return trained_model