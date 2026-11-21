"""Pruebas de la API con el modelo exportado, como se sirve dentro del contenedor."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from proyecto_mlops_equipo_9 import fastAPI
from proyecto_mlops_equipo_9.modeling import serving


@pytest.fixture
def cliente(exportacion, monkeypatch, tmp_path):
    monkeypatch.setenv("MODELO_DIR", str(exportacion.modelos))
    monkeypatch.delenv("DATASET_SERVICIO", raising=False)
    monkeypatch.setattr(serving, "DATASET_COMPLETO", tmp_path / "no-existe.parquet")
    serving.cargar_servicio.cache_clear()
    yield TestClient(fastAPI.app)
    serving.cargar_servicio.cache_clear()


def test_inicio_responde(cliente):
    assert cliente.get("/").status_code == 200


def test_predict_sin_cuerpo_usa_la_ultima_hora(cliente):
    respuesta = cliente.post("/predict")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["metadata"]["fecha_prediccion"] == "2026-05-31 23:00:00"
    assert cuerpo["metadata"]["modelo_utilizado"] == "models:/mibici-demanda-zona-hora/7"
    assert len(cuerpo["predicciones"]) == 3
    assert {p["tendencia"] for p in cuerpo["predicciones"]} <= {
        "positive",
        "negative",
        "balanced",
    }


def test_predict_con_periodo(cliente):
    respuesta = cliente.post(
        "/predict", json={"desde": "2026-05-31T20:00:00", "hasta": "2026-05-31T23:00:00"}
    )
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert len(cuerpo["predicciones"]) == 3 * 3
    assert cuerpo["metadata"]["zonas_totales"] == 3


@pytest.mark.parametrize(
    "cuerpo",
    [
        {"dataset": "/etc/passwd"},
        {"hasta": "2026-05-31T23:00:00"},
        {"desde": "2026-05-31T23:00:00", "hasta": "2026-05-31T20:00:00"},
        {"desde": "no es fecha"},
        {"desde": "2030-01-01T00:00:00"},
    ],
)
def test_predict_rechaza_entradas_invalidas(cliente, cuerpo):
    assert cliente.post("/predict", json=cuerpo).status_code == 422
