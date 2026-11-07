"""Ventanas temporales, línea base y métricas del pronóstico por zona y hora."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

OBJETIVOS = ["salidas_siguiente_hora", "llegadas_siguiente_hora"]
# La fila t predice t+1: el rezago de 167 h es la misma hora de la semana pasada.
LINEA_BASE = ["salidas_lag_167h", "llegadas_lag_167h"]


@dataclass(frozen=True)
class Ventana:
    """Entrena con objetivos terminados antes de `corte` y evalúa en [corte, fin)."""

    nombre: str
    corte: pd.Timestamp
    fin: pd.Timestamp


@dataclass(frozen=True)
class CalendarioValidacion:
    """Periodos acordados: selección, prueba única y escenario de drift."""

    seleccion_desde: str = "2025-10"
    seleccion_hasta: str = "2026-03"
    prueba_desde: str = "2026-04"
    prueba_hasta: str = "2026-05"
    drift_desde: str = "2026-06"
    drift_hasta: str = "2026-09"


def ventanas_mensuales(desde, hasta):
    """Una ventana por mes; cada corte usa todo lo anterior para entrenar."""
    return [
        Ventana(str(mes), mes.start_time, (mes + 1).start_time)
        for mes in pd.period_range(desde, hasta, freq="M")
    ]


def ventana_periodo(nombre, desde, hasta):
    """Una sola ventana que cubre varios meses, como la prueba previa al drift."""
    return Ventana(
        nombre, pd.Period(desde, "M").start_time, (pd.Period(hasta, "M") + 1).start_time
    )


def indices_ventana(df, ventana):
    """Índices posicionales de entrenamiento y evaluación según la hora objetivo."""
    entrenamiento = df.fin_hora_objetivo.le(ventana.corte).to_numpy()
    evaluacion = (
        df.inicio_hora_objetivo.ge(ventana.corte) & df.inicio_hora_objetivo.lt(ventana.fin)
    ).to_numpy()
    idx_entrenamiento, idx_evaluacion = np.flatnonzero(entrenamiento), np.flatnonzero(evaluacion)
    if not len(idx_entrenamiento) or not len(idx_evaluacion):
        raise ValueError(f"La ventana {ventana.nombre} no tiene datos suficientes.")
    return idx_entrenamiento, idx_evaluacion


def metricas(y_real, y_pred):
    """Errores en viajes; los conteos negativos pronosticados se recortan a cero."""
    real = np.asarray(y_real, dtype=float)
    pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    # Flujo neto = llegadas - salidas, la señal que usa el operador para priorizar.
    flujo_real, flujo_pred = real[:, 1] - real[:, 0], pred[:, 1] - pred[:, 0]
    return {
        "mae_salidas": float(np.abs(real[:, 0] - pred[:, 0]).mean()),
        "mae_llegadas": float(np.abs(real[:, 1] - pred[:, 1]).mean()),
        "mae": float(np.abs(real - pred).mean()),
        "mae_flujo_neto": float(np.abs(flujo_real - flujo_pred).mean()),
        "wape": float(np.abs(real - pred).sum() / max(real.sum(), 1)),
    }


def scorer_metricas(estimador, X, y):
    """Scorer de varias métricas para la búsqueda de hiperparámetros."""
    return metricas(y, estimador.predict(X))


def prediccion_base(df):
    """Pronóstico ingenuo: la misma hora de la semana pasada."""
    faltantes = [c for c in LINEA_BASE if c not in df]
    if faltantes:
        raise ValueError(
            f"Faltan {faltantes}: regenera features.py con rezagos_h que incluyan 167."
        )
    return df[LINEA_BASE].to_numpy(dtype=float)


def mejora_contra_base(mae_modelo, mae_base):
    """Fracción de error reducida frente a la línea base."""
    return float(1 - mae_modelo / mae_base)


def metricas_por_volumen(evaluacion, y_pred, entrenamiento):
    """MAE por tercil de volumen en entrenamiento; separa zonas sin historial."""
    medias = entrenamiento.groupby("zona_id")[OBJETIVOS[0]].mean()
    grupos = pd.qcut(medias.rank(method="first"), 3, labels=["bajo", "medio", "alto"])
    grupo = evaluacion.zona_id.map(grupos.astype(object)).fillna("sin_historial").to_numpy()
    real = evaluacion[OBJETIVOS].to_numpy(dtype=float)
    pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        f"mae_volumen_{nombre}": float(
            np.abs(real[grupo == nombre] - pred[grupo == nombre]).mean()
        )
        for nombre in pd.unique(grupo)
    }
