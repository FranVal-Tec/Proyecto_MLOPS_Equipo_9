"""Fixtures compartidas por las pruebas de exportación, del servicio y de la API."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from proyecto_mlops_equipo_9.modeling import export as ex
from proyecto_mlops_equipo_9.modeling.train import (
    ConfiguracionEntrenamiento,
    crear_pipeline,
    preparar_dataset,
    tipos_confiables,
)
from proyecto_mlops_equipo_9.modeling.validation import OBJETIVOS

VARIABLES_EXPORTACION = [
    "zona_id",
    "hora_objetivo",
    "dia_semana_objetivo",
    "salidas_lag_167h",
    "llegadas_lag_167h",
]
VERSION_FALSA = "7"


@pytest.fixture(scope="session")
def panel_exportacion():
    """Tres zonas con objetivos de marzo al 31 de mayo de 2026 a las 23:00."""
    objetivo = pd.date_range("2026-03-01 01:00", "2026-05-31 23:00", freq="h")
    rng = np.random.default_rng(9)
    partes = []
    for zona, escala in ((1, 2), (2, 5), (3, 9)):
        media = escala * (1.5 + np.sin(2 * np.pi * objetivo.hour / 24))
        partes.append(
            pd.DataFrame(
                {
                    "zona_id": f"G{zona}",  # como en los datos reales, texto
                    "inicio_hora_objetivo": objetivo,
                    "fin_hora_objetivo": objetivo + pd.Timedelta(hours=1),
                    "hora_objetivo": objetivo.hour,
                    "dia_semana_objetivo": objetivo.day_of_week,
                    "salidas_lag_167h": rng.poisson(media),
                    "llegadas_lag_167h": rng.poisson(media),
                    OBJETIVOS[0]: rng.poisson(media),
                    OBJETIVOS[1]: rng.poisson(media),
                }
            )
        )
    return pd.concat(partes, ignore_index=True)


@pytest.fixture(scope="session")
def modelo_exportacion(panel_exportacion):
    config = ConfiguracionEntrenamiento()
    df, columnas = preparar_dataset(panel_exportacion, VARIABLES_EXPORTACION, config)
    pipeline = crear_pipeline("hist_gradient_boosting", columnas, config)
    return pipeline.fit(df[columnas], df[OBJETIVOS])


@pytest.fixture
def descarga_falsa(modelo_exportacion, monkeypatch):
    """Sustituye al Model Registry: guarda el modelo como lo haría la descarga."""
    mlflow_sklearn = pytest.importorskip("mlflow.sklearn")
    pytest.importorskip("skops")

    def descargar(nombre, alias, carpeta, tracking_uri):
        ruta = carpeta / "modelo"
        mlflow_sklearn.save_model(
            modelo_exportacion,
            str(ruta),
            skops_trusted_types=tipos_confiables(modelo_exportacion),
        )
        return ruta, {
            "nombre": nombre,
            "version": VERSION_FALSA,
            "alias": alias,
            "uri_registro": f"models:/{nombre}/{VERSION_FALSA}",
            "run_id": "run-prueba",
            "etiquetas": {},
        }

    monkeypatch.setattr(ex, "descargar_version", descargar)


@pytest.fixture
def requirements_actuales(tmp_path):
    """requirements.txt con las versiones instaladas, como lo dejaría el equipo."""
    ruta = tmp_path / "requirements.txt"
    versiones = ex.versiones_entorno()
    lineas = [f"{p}=={v}" for p, v in versiones.items() if p != "python" and v]
    ruta.write_text("\n".join(lineas + ["-e ."]) + "\n", encoding="utf-8")
    return ruta


@pytest.fixture
def exportacion(panel_exportacion, descarga_falsa, requirements_actuales, tmp_path):
    """Exportación completa en una carpeta temporal, sin servidor de MLflow."""
    pytest.importorskip("pyarrow")
    dataset = tmp_path / "features.parquet"
    panel_exportacion.to_parquet(dataset, index=False)
    modelos = tmp_path / "models"
    metadatos = ex.exportar(dataset, modelos, "sin-servidor", requirements=requirements_actuales)
    return SimpleNamespace(modelos=modelos, metadatos=metadatos, dataset=dataset)
