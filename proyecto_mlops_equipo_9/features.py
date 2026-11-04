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
mlflow.set_experiment("Feature Engineering Experiment")

app = typer.Typer()

columnas_features = ['inicio', 'fin',
       'origen_id', 'destino_id', 'duracion_min',
       'viaje_corto', 'corto_distinta_estacion',
       'velocidad_kmh']

def obtener_datos_limpios(input_path):
    #A partir de la localización en donde se tienen los datos limpios se realiza
    #una lectura, y despues de guardan en un dataframe
    archivos_limpios = glob.glob(os.path.join(input_path, "*.parquet"))
    df = pd.concat(
        [pd.read_parquet(archivo) for archivo in archivos_limpios],
        ignore_index=True
    )

    return df

def eliminar_columnas(df, columnas):
    #Las columnas que no sean necesarias serán eliminadas.
    df_out = df[columnas].copy()
    return df_out

def cambiar_tipo_dato(df, columnas):
    #Debido a estar trabajando con series de tiempo se debe verificar que 
    #cumplan con su respectivo tipo de dato
    for columna in columnas:
        df[columna] = pd.to_datetime(df[columna])

    return df

def agregar_salidas(df):
    agg_df = (
        df.groupby([
            pd.Grouper(key="inicio", freq="15min"),
            "origen_id"
        ])
        .agg(
            salidas=("origen_id", "size"),
            duracion_prom_salidas=("duracion_min", "mean"),
            velocidad_prom_salidas=("velocidad_kmh", "mean"),
            proporcion_viajes_cortos_salidas=("viaje_corto", "mean"),
            proporcion_cortos_distinta_estacion_salidas=(
                "corto_distinta_estacion", "mean"
            )
        )
        .reset_index()
        .rename(columns={
            "inicio": "inicio_intervalo",
            "origen_id": "area_id"
        })
    )
    return agg_df

def agregar_llegadas(df):
    agg_df = (
        df.groupby([
            pd.Grouper(key="fin", freq="15min"),
            "destino_id"
        ])
        .agg(
            llegadas=("origen_id", "size"),
            duracion_prom_llegadas=("duracion_min", "mean"),
            velocidad_prom_llegadas=("velocidad_kmh", "mean"),
            proporcion_viajes_cortos_llegadas=("viaje_corto", "mean"),
            proporcion_cortos_distinta_estacion_llegadas=(
                "corto_distinta_estacion", "mean"
            )
        )
        .reset_index()
        .rename(columns={
            "fin": "inicio_intervalo",
            "destino_id": "area_id"
        })
    )
    return agg_df

def procesar_agregado(df):
    #Se realiza el proceso de agregado sobre intervalos de tiempo
    #y ids de areas.
    salidas = agregar_salidas(df)
    llegadas = agregar_llegadas(df)

    agg_df = pd.merge(
        salidas,
        llegadas,
        on=["inicio_intervalo", "area_id"],
        how="outer"
    )

    agg_df[["salidas", "llegadas"]] = (
        agg_df[["salidas", "llegadas"]]
        .fillna(0)
        .astype(int)
    )

    agg_df = agg_df.sort_values(
        ["area_id","inicio_intervalo"]
    ).reset_index(drop=True)

    #En caso de que al agregar por tiempo se creen valores vacios
    #estos se cambiarán por 0, ya que siginfica que no hay en ese intervalo
    agg_df["salidas"] = agg_df["salidas"].fillna(0)
    agg_df["llegadas"] = agg_df["llegadas"].fillna(0)
    
    return agg_df

def agregar_variables_objetivo(df):
    #Para cada una de las variables donde tenemos la cantidad de viajes
    #se calculara la cantidad de la siguiente hora, lo cual es lo que se
    #desea predecir en el modelo final.
    for col in ["salidas", "llegadas"]:
        df[f"{col}_siguiente_hora"] = (
            df.groupby("area_id")[col]
            .transform(
                lambda s: s.shift(-1)
                .rolling(4, min_periods=4)
                .sum()
            )
        )

    df["salidas_siguiente_hora"] = df["salidas_siguiente_hora"].fillna(0)
    df["llegadas_siguiente_hora"] = df["llegadas_siguiente_hora"].fillna(0)

    return df

def agregar_caracteristicas_temporales(df):
    # Asegurar que la columna de fecha tenga formato datetime
    df["inicio_intervalo"] = pd.to_datetime(
        df["inicio_intervalo"]
    )

    # Características temporales básicas
    df["hora"] = df["inicio_intervalo"].dt.hour
    df["dia_semana"] = df["inicio_intervalo"].dt.dayofweek
    df["dia_mes"] = df["inicio_intervalo"].dt.day
    df["mes"] = df["inicio_intervalo"].dt.month
    df["dia_anio"] = df["inicio_intervalo"].dt.dayofyear
    df["anio"] = df["inicio_intervalo"].dt.year

    # Variable binaria: fin de semana
    df["es_fin_semana"] = (
        df["dia_semana"] >= 5
    ).astype(int)

    # Hora pico (ajustable según los patrones de tu ciudad)
    df["es_hora_pico"] = (
        df["hora"].between(7, 9)
        | df["hora"].between(16, 19)
    ).astype(int)

    # Codificación cíclica de la hora (ciclo de 24 horas)
    hora_decimal = (
        df["inicio_intervalo"].dt.hour
        + df["inicio_intervalo"].dt.minute / 60
    )

    df["hora_sin"] = np.sin(
        2 * np.pi * hora_decimal / 24
    )
    df["hora_cos"] = np.cos(
        2 * np.pi * hora_decimal / 24
    )

    # Codificación cíclica del día de la semana (ciclo de 7 días)
    df["dia_semana_sin"] = np.sin(
        2 * np.pi * df["dia_semana"] / 7
    )
    df["dia_semana_cos"] = np.cos(
        2 * np.pi * df["dia_semana"] / 7
    )

    return df

def ordenar_columnas(df, objetivos):
    df = df[
        [col for col in df.columns if col not in objetivos]
        + objetivos
    ]
    return df

def crear_pipeline_features():
    #Creación de los pasos del pipeline de acuerdo a las funciones creadas anteriormente
    pipeline = Pipeline(steps=[
        (
            "eliminar_columnas",
            FunctionTransformer(
                lambda df: eliminar_columnas( df, columnas_features ),
                validate=False
            )
        ),
        (
            "cambiar_tipo_dato",
            FunctionTransformer(
                lambda df: cambiar_tipo_dato( df, ["inicio", "fin"] ),
                validate=False
            )
        ),
        (
            "procesar_agregado",
            FunctionTransformer(
                procesar_agregado,
                validate=False
            )
        ),
        (
            "agregar_variables_objetivo",
            FunctionTransformer(
                agregar_variables_objetivo,
                validate=False
            )
        ),
        (
            "agregar_caracteristicas_temporales",
            FunctionTransformer(
                agregar_caracteristicas_temporales,
                validate=False
            )
        ),
        (
            "ordenar_columnas",
            FunctionTransformer(
                lambda df: ordenar_columnas( df, [ "salidas_siguiente_hora", "llegadas_siguiente_hora" ] ),
                validate=False
            )
        )
    ],
    verbose=True
    )

    return pipeline

with mlflow.start_run():
    #Asignación de variales de rutas de entrada y salida.
    input_path: Path = CLEAN_DATA_DIR
    output_path: Path = PROCESSED_DATA_DIR / "datos_feature_engineered.parquet"

    #Creación y ejecución del pipeline creado para realizar feature engineering.
    print('\n')

    df = obtener_datos_limpios(input_path)
    print('Datos obtenidos correctamente desde ruta\n')

    
    pipeline_features = crear_pipeline_features()

    try:
        df = pipeline_features.fit_transform(df)
        print('Pipeline finalizado correctamente')
    except Exception as e:
        print('Error durante la ejecución del Pipeline:')
        print(f'Detalle de error: {e}')
        raise
   
    #Guardado de el dataframe terminado en la ruta de salida
    df.to_parquet(
        output_path,
        index=False
    )
    print('\nDatos guardados correctamente en '+str(output_path))

    #Guardado y registro en mlflow para su versionamiento.
    print('\nComenzando proceso de registro en mlflow')
    #Se crea un objeto tipo dataset de mlflow
    mlflow_dataset = mlflow.data.from_pandas(
        df , name="datos_transformados",
    )

    # Agregar datos para el control de versiones de mlflow
    mlflow.log_input(mlflow_dataset, context="feature engineering")

    # Log metadata de version
    mlflow.log_params({
        "data_size": len(df),
        "features_count": len(df.columns) - 2,
        "salidas_distribution": df["salidas_siguiente_hora"].value_counts().to_dict(),
        "llegadas_distribution": df["llegadas_siguiente_hora"].value_counts().to_dict(),
    })

    # Log data quality metrics
    mlflow.log_metrics({
        "missing_values_pct": (df.isnull().sum().sum() / df.size) * 100,
        "duplicate_rows": df.duplicated().sum(),
        "salidas_balance": df["salidas_siguiente_hora"].std(),
        "llegadas_balance": df["llegadas_siguiente_hora"].std(),
    })

    

    # Access dataset metadata
    print(f"\n\nDataset name: {mlflow_dataset.name}")  # Defaults to "dataset" if not specified
    print(f"Dataset digest: {mlflow_dataset.digest}")  # Unique hash identifier (computed automatically)
    print(f"Dataset source: {mlflow_dataset.source}")  # DatasetSource object
    print(f"Dataset profile: {mlflow_dataset.profile}")  # Optional: implementation-specific statistics
    print(f"Dataset schema: {mlflow_dataset.schema}")  # Optional: implementation-specific schema
    