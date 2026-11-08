"""Pruebas temporales y de cobertura de la ingeniería de variables."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from proyecto_mlops_equipo_9.features import (
    ConfiguracionFeatures,
    agregar_historia,
    agregar_variables_objetivo,
    construir_panel,
    crear_pipeline_features,
)


def test_objetivo_hora_siguiente_sin_cruzar_zonas():
    frame = pd.DataFrame(
        {
            "zona_id": ["A"] * 3 + ["B"] * 3,
            "inicio_hora": list(pd.date_range("2026-01-01", periods=3, freq="h")) * 2,
            "salidas": [1, 2, 3, 10, 20, 30],
            "llegadas": [4, 5, 6, 40, 50, 60],
        }
    )
    original = frame.copy(deep=True)
    result = agregar_variables_objetivo(frame)
    assert result.salidas_siguiente_hora.iloc[[0, 1, 3, 4]].tolist() == [2, 3, 20, 30]
    assert result.groupby("zona_id").tail(1).salidas_siguiente_hora.isna().all()
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError, match="continuo"):
        agregar_variables_objetivo(frame.drop(index=1))


def test_ceros_activacion_y_falla_global():
    horas = pd.date_range("2026-01-01", periods=5, freq="h")
    counts = pd.DataFrame(
        {
            "zona_id": ["A", "A", "B", "A"],
            "inicio_hora": [horas[0], horas[1], horas[1], horas[3]],
            "salidas": [1.0, 1.0, 1.0, 1.0],
            "llegadas": [2.0, 2.0, 2.0, 2.0],
        }
    ).set_index(["zona_id", "inicio_hora"])
    result = construir_panel(counts, ["A", "B"], horas).set_index(["zona_id", "inicio_hora"])
    assert pd.isna(result.loc[("B", horas[0]), "salidas"])
    assert result.loc[("B", horas[3]), "salidas"] == 0
    assert result.loc[("B", horas[3]), "salidas_cero_asumido"]
    assert result.loc[(slice(None), horas[2]), "salidas"].isna().all()
    failure = construir_panel(
        counts, ["A", "B"], horas, periodos_sin_cobertura=[(horas[1], horas[2])]
    )
    assert failure.loc[failure.inicio_hora.eq(horas[1]), "salidas"].isna().all()


def test_historia_cobertura_y_perturbacion_futura():
    frame = pd.DataFrame(
        {
            "zona_id": ["A"] * 200,
            "inicio_hora": pd.date_range("2026-01-01", periods=200, freq="h"),
            "salidas": np.arange(200, dtype=float),
            "llegadas": np.arange(200, dtype=float),
        }
    )
    frame.loc[176:181, "salidas"] = np.nan
    result = agregar_historia(frame)
    assert result.loc[24, "salidas_media_previa_24h"] == 11.5
    assert result.loc[:23, "salidas_media_previa_24h"].isna().all()
    assert result.loc[199, "salidas_cobertura_previa_24h"] == 0.75
    assert result.loc[199, "salidas_media_previa_24h"] == frame.loc[175:198, "salidas"].mean()
    changed = frame.copy()
    changed.loc[182, "salidas"] = np.nan
    assert pd.isna(agregar_historia(changed).loc[199, "salidas_media_previa_24h"])
    changed = frame.copy()
    changed.loc[180, ["salidas", "llegadas"]] = 99999
    cols = [c for c in result if "_lag_" in c or "_previa_" in c]
    pd.testing.assert_frame_equal(
        result.loc[:179, cols], agregar_historia(changed).loc[:179, cols]
    )


def test_pipeline_sklearn_sin_objetivos_y_borde_inicial():
    horas = pd.date_range("2026-01-01", periods=8, freq="h")
    counts = pd.DataFrame(
        {"zona_id": ["A"] * 8, "inicio_hora": horas, "salidas": [1.0] * 8, "llegadas": [1.0] * 8}
    )
    counts = counts.set_index(["zona_id", "inicio_hora"])
    config = ConfiguracionFeatures(rezagos_h=(1,), ventanas_h=(2,))
    pipeline = crear_pipeline_features(["A"], horas, horas[0], pd.Timedelta(hours=3), config)
    assert isinstance(pipeline, Pipeline)
    assert all(isinstance(transformer, FunctionTransformer) for _, transformer in pipeline.steps)
    result = pipeline.fit_transform(counts)
    assert not any(c.endswith("_siguiente_hora") for c in result)
    assert result.loc[:2, "llegadas"].isna().all()
    assert result.loc[0, "inicio_hora_objetivo"] == horas[1]
    assert result.loc[0, "fin_hora_objetivo"] == horas[2]
    assert result.loc[0, "hora_objetivo"] == 1
    assert result.loc[0, "llegadas_borde_inicial"]


def test_cierre_nocturno_en_variables_y_no_como_objetivo():
    horas = pd.date_range("2026-01-01", periods=10, freq="h")
    activas = [0, 1, 2, 5, 6, 7, 9]
    counts = pd.DataFrame(
        {"zona_id": "A", "inicio_hora": horas[activas], "salidas": 1.0, "llegadas": 1.0}
    ).set_index(["zona_id", "inicio_hora"])
    panel = construir_panel(counts, ["A"], horas, horas_cierre=(0, 1, 2, 3, 4, 5))
    assert panel.loc[[3, 4], "salidas"].eq(0).all()
    assert panel.loc[[3, 4], "cierre_nocturno"].all()
    assert pd.isna(panel.loc[8, "salidas"])
    objetivos = agregar_variables_objetivo(panel)
    assert objetivos.loc[[2, 3], "salidas_siguiente_hora"].isna().all()
    assert objetivos.loc[4, "salidas_siguiente_hora"] == 1
    assert pd.isna(objetivos.loc[7, "salidas_siguiente_hora"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_cobertura_media": 0},
        {"min_cobertura_media": 1.1},
        {"celda_m": 0},
        {"rezagos_h": (0,)},
        {"ventanas_h": ()},
        {"rezagos_h": (1, 1)},
        {"periodos_sin_cobertura": (("2026-02-01", "2026-01-01"),)},
        {"horas_cierre": (24,)},
    ],
)
def test_configuracion_invalida(kwargs):
    with pytest.raises(ValueError):
        ConfiguracionFeatures(**kwargs)
