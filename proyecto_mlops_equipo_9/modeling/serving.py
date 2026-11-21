"""Modelo y datos que usa la API, cargados una sola vez por proceso.

Si existe models/modelo_servicio.json (la exportación de `export.py`, que es lo que va dentro
de la imagen de Docker), el modelo se carga de models/ y no se necesita servidor de MLflow. Si
no existe, se usa el Model Registry, como en desarrollo con `mlflow ui`.

Variables de entorno (configuración del servidor, no del cliente):
    MODELO_DIR           carpeta de la exportación (por defecto, models/)
    DATASET_SERVICIO     Parquet de variables a usar en lugar del predeterminado
    MODELO_URI           modelo del Model Registry cuando no hay exportación
    MLFLOW_TRACKING_URI  servidor de MLflow cuando no hay exportación
"""

from functools import lru_cache
import os
from pathlib import Path

import pandas as pd

from proyecto_mlops_equipo_9.config import MODELS_DIR, PROCESSED_DATA_DIR
from proyecto_mlops_equipo_9.modeling.export import ARCHIVO_METADATOS, cargar_modelo_local
from proyecto_mlops_equipo_9.modeling.predict import predecir, seleccionar_filas
from proyecto_mlops_equipo_9.modeling.train import ConfiguracionEntrenamiento

DATASET_COMPLETO = PROCESSED_DATA_DIR / "datos_feature_engineered_zona_hora.parquet"


def cargar_desde_mlflow():
    """Modelo del Model Registry y su URI; requiere el servidor de MLflow."""
    import mlflow

    config = ConfiguracionEntrenamiento()
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    uri = os.environ.get("MODELO_URI", f"models:/{config.nombre_registro}@{config.alias_registro}")
    return mlflow.sklearn.load_model(uri), uri


@lru_cache(maxsize=1)
def cargar_servicio():
    """Devuelve (modelo, datos, origen del modelo); se calcula en la primera petición."""
    modelos_dir = Path(os.environ.get("MODELO_DIR", MODELS_DIR))
    if (modelos_dir / ARCHIVO_METADATOS).is_file():
        modelo, metadatos = cargar_modelo_local(modelos_dir)
        origen = metadatos["uri_registro"]
        # El Parquet completo solo existe donde se corrió el pipeline; en el contenedor se usan
        # los días recientes que se exportaron junto con el modelo.
        dataset = DATASET_COMPLETO
        if not dataset.is_file():
            dataset = modelos_dir / metadatos["datos"]["archivo"]
    else:
        modelo, origen = cargar_desde_mlflow()
        dataset = DATASET_COMPLETO
    dataset = Path(os.environ.get("DATASET_SERVICIO", dataset))
    return modelo, pd.read_parquet(dataset), origen


def generar_pronostico(desde=None, hasta=None):
    """Pronóstico del periodo pedido (sin periodo, la última hora disponible) y su modelo."""
    modelo, datos, origen = cargar_servicio()
    return predecir(modelo, seleccionar_filas(datos, desde, hasta)), origen
