"""Pruebas de cómo el servicio elige el modelo y los datos."""

import pandas as pd
import pytest

from proyecto_mlops_equipo_9.modeling import serving


@pytest.fixture(autouse=True)
def servicio_limpio(monkeypatch):
    """Cada prueba carga el servicio desde cero y sin variables de entorno heredadas."""
    for variable in ("MODELO_DIR", "DATASET_SERVICIO", "MODELO_URI", "MLFLOW_TRACKING_URI"):
        monkeypatch.delenv(variable, raising=False)
    serving.cargar_servicio.cache_clear()
    yield
    serving.cargar_servicio.cache_clear()


def test_usa_la_exportacion_sin_mlflow(exportacion, monkeypatch, tmp_path):
    monkeypatch.setenv("MODELO_DIR", str(exportacion.modelos))
    # Como en el contenedor: no existe el Parquet completo de data/processed.
    monkeypatch.setattr(serving, "DATASET_COMPLETO", tmp_path / "no-existe.parquet")

    _, datos, origen = serving.cargar_servicio()
    assert origen == "models:/mibici-demanda-zona-hora/7"
    assert len(datos) == exportacion.metadatos["datos"]["filas"]

    ultima, _ = serving.generar_pronostico()
    assert set(ultima.inicio_hora_objetivo) == {pd.Timestamp("2026-05-31 23:00")}
    assert len(ultima) == 3

    periodo, _ = serving.generar_pronostico("2026-05-31 20:00", "2026-05-31 23:00")
    assert len(periodo) == 3 * 3

    with pytest.raises(ValueError, match="No hay filas"):
        serving.generar_pronostico("2030-01-01 00:00")


def test_prefiere_el_dataset_completo_y_respeta_la_variable(exportacion, monkeypatch):
    monkeypatch.setenv("MODELO_DIR", str(exportacion.modelos))
    monkeypatch.setattr(serving, "DATASET_COMPLETO", exportacion.dataset)
    assert len(serving.cargar_servicio()[1]) == len(pd.read_parquet(exportacion.dataset))

    serving.cargar_servicio.cache_clear()
    recientes = exportacion.modelos / exportacion.metadatos["datos"]["archivo"]
    monkeypatch.setenv("DATASET_SERVICIO", str(recientes))
    assert len(serving.cargar_servicio()[1]) == exportacion.metadatos["datos"]["filas"]


def test_sin_exportacion_usa_el_model_registry(exportacion, monkeypatch, tmp_path):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("mlflow.sklearn")
    llamadas = {}
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda uri: llamadas.update(tracking=uri))
    monkeypatch.setattr(
        mlflow.sklearn, "load_model", lambda uri: llamadas.update(modelo=uri) or "modelo"
    )
    monkeypatch.setenv("MODELO_DIR", str(tmp_path / "vacio"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    monkeypatch.setenv("DATASET_SERVICIO", str(exportacion.dataset))

    modelo, _, origen = serving.cargar_servicio()
    assert modelo == "modelo"
    assert origen == "models:/mibici-demanda-zona-hora@candidato"
    assert llamadas == {"tracking": "http://mlflow:5000", "modelo": origen}

    serving.cargar_servicio.cache_clear()
    monkeypatch.setenv("MODELO_URI", "models:/mibici-demanda-zona-hora/2")
    assert serving.cargar_servicio()[2] == "models:/mibici-demanda-zona-hora/2"
