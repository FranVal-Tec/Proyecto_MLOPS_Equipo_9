"""Pronóstico por zona con el modelo registrado, ordenado por prioridad de atención.

Uso: python -m proyecto_mlops_equipo_9.modeling.predict --desde "2026-09-30 18:00"
Sin fechas, usa la última hora objetivo disponible en el dataset.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from proyecto_mlops_equipo_9.config import PROCESSED_DATA_DIR
from proyecto_mlops_equipo_9.modeling.train import ConfiguracionEntrenamiento


def predecir(modelo, filas):
    """Salidas, llegadas y flujo neto; las zonas con mayor flujo absoluto van primero."""
    columnas = list(modelo.feature_names_in_)
    entrada = filas[columnas].copy()
    numericas = [c for c in columnas if c != "zona_id"]
    entrada[numericas] = entrada[numericas].astype("float64")
    entrada["zona_id"] = entrada.zona_id.astype(str)
    pronostico = np.clip(modelo.predict(entrada), 0, None)
    out = filas[["zona_id", "inicio_hora_objetivo"]].copy()
    out["salidas_pronosticadas"] = pronostico[:, 0]
    out["llegadas_pronosticadas"] = pronostico[:, 1]
    out["flujo_neto_pronosticado"] = pronostico[:, 1] - pronostico[:, 0]
    return out.sort_values(
        ["inicio_hora_objetivo", "flujo_neto_pronosticado"],
        key=lambda c: c.abs() if c.name == "flujo_neto_pronosticado" else c,
        ascending=[True, False],
    ).reset_index(drop=True)


def seleccionar_filas(df, desde=None, hasta=None):
    """Filas del periodo pedido; sin periodo, la última hora objetivo disponible."""
    if desde is None:
        return df[df.inicio_hora_objetivo.eq(df.inicio_hora_objetivo.max())]
    inicio = pd.Timestamp(desde)
    fin = pd.Timestamp(hasta) if hasta else inicio + pd.Timedelta(hours=1)
    filas = df[df.inicio_hora_objetivo.ge(inicio) & df.inicio_hora_objetivo.lt(fin)]
    if filas.empty:
        raise ValueError("No hay filas en el periodo solicitado.")
    return filas


def main():
    config = ConfiguracionEntrenamiento()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROCESSED_DATA_DIR / "datos_feature_engineered_zona_hora.parquet",
    )
    parser.add_argument("--desde", default=None)
    parser.add_argument("--hasta", default=None)
    parser.add_argument("--salida", type=Path, default=PROCESSED_DATA_DIR / "predicciones.csv")
    parser.add_argument(
        "--modelo-uri", default=f"models:/{config.nombre_registro}@{config.alias_registro}"
    )
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = parser.parse_args()

    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    modelo = mlflow.sklearn.load_model(args.modelo_uri)
    filas = seleccionar_filas(pd.read_parquet(args.dataset), args.desde, args.hasta)
    resultado = predecir(modelo, filas)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(args.salida, index=False)
    print(resultado.head(10).to_string(index=False))
    print(f"Pronósticos guardados en {args.salida}")


if __name__ == "__main__":
    main()
