from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

import mlflow
import glob
import os

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from proyecto_mlops_equipo_9.config import (PROCESSED_DATA_DIR, CLEAN_DATA_DIR)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Data Partitioning Experiment")

# Datos cronológicos default para particionar los archivos.
SPLITS = {
    "train": {
        "start": "2025-01-01",
        "end": "2026-04-01"
    },
    "validation": {
        "start": "2026-04-01",
        "end": "2026-06-01"
    },
    "drift": {
        "start": "2026-06-01",
        "end": "2026-10-01"
    }
}

TIMESTAMP_COL = "inicio_intervalo"

def creacion_particiones(df):
    partitions = {}

    for name, dates in SPLITS.items():

        start = pd.Timestamp(dates["start"])
        end = pd.Timestamp(dates["end"])

        mask = (
            (df[TIMESTAMP_COL] >= start) &
            (df[TIMESTAMP_COL] < end)
        )

        partition = df.loc[mask].copy()

        if partition.empty:
            raise ValueError(f"{name} partition is empty.")

        partitions[name] = partition

    return partitions

with mlflow.start_run():
    #Asignación de variales de rutas de entrada y salida.
    input_path: Path = PROCESSED_DATA_DIR / "datos_feature_engineered.parquet"
    output_path: Path = PROCESSED_DATA_DIR / "partitions"

    df = pd.read_parquet(input_path)
    print('Datos obtenidos correctamente desde ruta\n')

    # Verificación rápida de que la columna de fecha exista.
    if TIMESTAMP_COL not in df.columns:
        raise ValueError(
            f"Timestamp column '{TIMESTAMP_COL}' not found."
        )

    # Ordenar los valores deacuerdo a su fecha previo a la partición.
    df = df.sort_values(TIMESTAMP_COL).reset_index(drop=True)

    partitions = creacion_particiones(df)

    # Verificación que folder destino exista, en caso de que sea falso crear.
    output_path.mkdir(parents=True, exist_ok=True)

    # LLevar a cabo el guardado de cada dataset dentro de su localización.
    for name, partition in partitions.items():
        print(name)
        output_file = output_path / f"{name}.parquet"
        print(output_file)

        mlflow_dataset = mlflow.data.from_pandas(
            partition , name=f"datos_{name}",
        )
        # Agregar datos para el control de versiones de mlflow
        mlflow.log_input(mlflow_dataset, context=f"particionamiento de {name}")

        partition.to_parquet(
                output_file,
                index=False
            )