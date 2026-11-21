"""Pruebas de ventanas, línea base, métricas y pipelines de modelado."""

import numpy as np
import pandas as pd
import pytest

from proyecto_mlops_equipo_9.modeling.predict import predecir
from proyecto_mlops_equipo_9.modeling.train import (
    ESPACIOS,
    ConfiguracionEntrenamiento,
    buscar_hiperparametros,
    crear_pipeline,
    preparar_dataset,
    tipos_confiables,
)
from proyecto_mlops_equipo_9.modeling.validation import (
    OBJETIVOS,
    indices_ventana,
    metricas,
    metricas_por_volumen,
    prediccion_base,
    ventana_periodo,
    ventanas_mensuales,
    scorer_metricas,
)


@pytest.fixture
def panel():
    horas = pd.date_range("2025-01-01", "2025-07-31 23:00", freq="h")
    rng = np.random.default_rng(9)
    partes = []
    for zona, escala in (("A", 2), ("B", 5), ("C", 9)):
        media = escala * (1.5 + np.sin(2 * np.pi * horas.hour / 24))
        partes.append(
            pd.DataFrame(
                {
                    "zona_id": zona,
                    "inicio_hora": horas,
                    "inicio_hora_objetivo": horas + pd.Timedelta(hours=1),
                    "fin_hora_objetivo": horas + pd.Timedelta(hours=2),
                    "hora_objetivo": (horas + pd.Timedelta(hours=1)).hour,
                    "dia_semana_objetivo": horas.day_of_week,
                    "anio_objetivo": 2025,
                    "salidas_lag_167h": rng.poisson(media),
                    "llegadas_lag_167h": rng.poisson(media),
                    OBJETIVOS[0]: rng.poisson(media),
                    OBJETIVOS[1]: rng.poisson(media),
                }
            )
        )
    return pd.concat(partes, ignore_index=True).sample(frac=1, random_state=9)


FEATURES = [
    "zona_id",
    "hora_objetivo",
    "dia_semana_objetivo",
    "anio_objetivo",
    "salidas_lag_167h",
    "llegadas_lag_167h",
]


def test_ventana_no_usa_objetivos_posteriores_al_corte(panel):
    ventana = ventanas_mensuales("2025-05", "2025-05")[0]
    entrenamiento, evaluacion = indices_ventana(panel, ventana)
    assert panel.iloc[entrenamiento].fin_hora_objetivo.max() <= ventana.corte
    assert panel.iloc[evaluacion].inicio_hora_objetivo.min() >= ventana.corte
    assert panel.iloc[evaluacion].inicio_hora_objetivo.max() < ventana.fin
    assert not set(entrenamiento) & set(evaluacion)


def test_ventanas_avanzan_y_periodo_cubre_meses():
    ventanas = ventanas_mensuales("2025-10", "2026-03")
    assert [v.nombre for v in ventanas][0::5] == ["2025-10", "2026-03"]
    assert all(a.fin == b.corte for a, b in zip(ventanas, ventanas[1:]))
    prueba = ventana_periodo("prueba", "2026-04", "2026-05")
    assert prueba.corte == pd.Timestamp("2026-04-01")
    assert prueba.fin == pd.Timestamp("2026-06-01")


def test_metricas_recortan_negativos_y_calculan_flujo():
    resultado = metricas([[1, 3], [2, 2]], [[-1, 3], [2, 4]])
    assert resultado["mae_salidas"] == 0.5
    assert resultado["mae_llegadas"] == 1
    assert resultado["mae"] == 0.75
    assert resultado["mae_flujo_neto"] == 1.5


def test_scorer_prioriza_menor_error():
    """Un menor MAE debe producir un score mayor sin alterar las métricas originales."""

    class EstimadorFalso:
        def __init__(self, predicciones):
            self.predicciones = predicciones

        def predict(self, X):
            return self.predicciones

    y_real = np.array([[2, 4], [3, 5]], dtype=float)
    X = np.zeros((2, 1))

    perfecto = scorer_metricas(EstimadorFalso(y_real), X, y_real)
    impreciso = scorer_metricas(EstimadorFalso(y_real + 2), X, y_real)

    assert perfecto["mae"] == 0.0
    assert impreciso["mae"] == -2.0
    assert perfecto["mae"] > impreciso["mae"]

    for nombre, valor in metricas(y_real, y_real + 2).items():
        assert impreciso[nombre] == -valor


def test_linea_base_exige_rezago_de_misma_hora():
    with pytest.raises(ValueError, match="167"):
        prediccion_base(pd.DataFrame({"salidas_lag_168h": [1], "llegadas_lag_168h": [1]}))


def test_zona_sin_historial_se_reporta_aparte():
    entrenamiento = pd.DataFrame({"zona_id": list("ABC"), OBJETIVOS[0]: [1.0, 2.0, 3.0]})
    evaluacion = pd.DataFrame(
        {"zona_id": ["C", "N"], OBJETIVOS[0]: [3.0, 1.0], OBJETIVOS[1]: [3.0, 1.0]}
    )
    resultado = metricas_por_volumen(evaluacion, [[3, 3], [2, 2]], entrenamiento)
    assert resultado == {"mae_volumen_alto": 0.0, "mae_volumen_sin_historial": 1.0}


@pytest.mark.parametrize("nombre", sorted(ESPACIOS))
def test_busqueda_y_prediccion_por_modelo(panel, nombre):
    config = ConfiguracionEntrenamiento(n_iter=1, max_muestras_bosque=0.5)
    df, columnas = preparar_dataset(panel, FEATURES, config)
    assert "anio_objetivo" not in columnas
    ventanas = ventanas_mensuales("2025-06", "2025-07")
    parametros, por_ventana = buscar_hiperparametros(nombre, df, columnas, ventanas, config)
    assert len(por_ventana) == 2 and por_ventana[0]["mae"] > 0
    modelo = crear_pipeline(nombre, columnas, config).set_params(**parametros)
    modelo.fit(df[columnas], df[OBJETIVOS])
    filas = df[df.inicio_hora_objetivo.eq(df.inicio_hora_objetivo.max())]
    filas = pd.concat([filas, filas.assign(zona_id="NUEVA").head(1)])
    resultado = predecir(modelo, filas)
    assert len(resultado) == 4
    assert (resultado[["salidas_pronosticadas", "llegadas_pronosticadas"]] >= 0).all().all()
    flujo = resultado.flujo_neto_pronosticado.abs()
    assert flujo.is_monotonic_decreasing


def test_modelos_se_guardan_y_cargan_con_mlflow(panel, tmp_path):
    mlflow_sklearn = pytest.importorskip("mlflow.sklearn")
    pytest.importorskip("skops")
    config = ConfiguracionEntrenamiento(max_muestras_bosque=0.5)
    df, columnas = preparar_dataset(panel, FEATURES, config)
    muestra = df[columnas].head(500)
    for nombre in sorted(ESPACIOS):
        modelo = crear_pipeline(nombre, columnas, config).fit(df[columnas], df[OBJETIVOS])
        ruta = str(tmp_path / nombre)
        mlflow_sklearn.save_model(modelo, ruta, skops_trusted_types=tipos_confiables(modelo))
        cargado = mlflow_sklearn.load_model(ruta)
        np.testing.assert_allclose(cargado.predict(muestra), modelo.predict(muestra))



def test_integracion_pipeline_carga_prediccion_y_metricas(panel, tmp_path):
    ruta_dataset = tmp_path / "panel_integracion.parquet"
    panel.to_parquet(ruta_dataset, index=False)
    datos = pd.read_parquet(ruta_dataset)

    assert not datos.empty

    config = ConfiguracionEntrenamiento()
    df, columnas = preparar_dataset(datos, FEATURES, config)

    assert "anio_objetivo" not in columnas
    assert all(columna in df.columns for columna in columnas)

    ventana = ventanas_mensuales("2025-05", "2025-05")[0]
    entrenamiento, evaluacion = indices_ventana(df, ventana)

    datos_entrenamiento = df.iloc[entrenamiento]
    datos_evaluacion = df.iloc[evaluacion]

    modelo = crear_pipeline("ridge", columnas, config)
    modelo.fit(
        datos_entrenamiento[columnas],
        datos_entrenamiento[OBJETIVOS],
    )

    predicciones = predecir(modelo, datos_evaluacion)

    assert not predicciones.empty
    assert len(predicciones) == len(datos_evaluacion)

    claves = ["zona_id", "inicio_hora_objetivo"]
    comparacion = predicciones.merge(
        datos_evaluacion[claves + OBJETIVOS],
        on=claves,
        validate="one_to_one",
    )

    assert len(comparacion) == len(datos_evaluacion)

    y_real = comparacion[OBJETIVOS].to_numpy()
    y_pred = comparacion[
        ["salidas_pronosticadas", "llegadas_pronosticadas"]
    ].to_numpy()

    resultados = metricas(y_real, y_pred)

    assert set(resultados) == {
        "mae_salidas",
        "mae_llegadas",
        "mae",
        "mae_flujo_neto",
        "wape",
    }
    assert all(np.isfinite(valor) for valor in resultados.values())
    assert all(valor >= 0 for valor in resultados.values())
