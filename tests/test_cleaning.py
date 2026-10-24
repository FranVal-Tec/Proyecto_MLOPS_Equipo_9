"""Pruebas de integridad, límites, controles de versión y publicación silver."""

import json

import numpy as np
import pandas as pd
import pytest

from proyecto_mlops_equipo_9 import cleaning as c
from proyecto_mlops_equipo_9.dataset import COLUMNAS


@pytest.fixture
def coords():
    return pd.DataFrame(
        {"latitude": [20.6840, 20.6726, np.nan], "longitude": [-103.3555, -103.3535, np.nan]},
        index=[51, 60, 70],
    )


@pytest.fixture
def raw():
    filas = []

    def fila(
        i,
        segundos,
        origen=51,
        destino=60,
        nacimiento=2000,
        genero="F",
        inicio="2026-09-01 10:00:00",
    ):
        inicio = pd.Timestamp(inicio)
        filas.append(
            [
                i,
                10,
                genero,
                nacimiento,
                inicio,
                inicio + pd.Timedelta(seconds=segundos),
                origen,
                destino,
            ]
        )

    fila(1, 600)
    fila(1, 600)  # copia exacta
    fila(2, 600, nacimiento=None, genero=None)
    fila(3, 30)  # velocidad alta
    fila(4, 10801, destino=51)
    fila(5, 59, destino=51)
    fila(6, 60, destino=51)  # límite inclusivo aceptado
    fila(7, 600, nacimiento=1800, genero="X")
    fila(8, 600, destino=999)
    fila(9, 600)
    fila(9, 601)  # mismo ID, valores distintos
    fila(10, 0)
    fila(11, 600, inicio="2026-08-31 23:59:00")
    fila(12, 600, destino=70)  # catálogo sin coordenadas
    fila(13, 10800, destino=51)
    df = pd.DataFrame(filas, columns=COLUMNAS)
    df["genero"] = df["genero"].astype("category")
    df["duracion_min"] = (df["fin"] - df["inicio"]).dt.total_seconds() / 60
    return df


def test_reglas_limites_conflictos_y_balance(raw, coords):
    anotados, validos, bitacora = c.limpiar_mes(raw, "2026_09", coords)
    assert set(validos["viaje_id"]) == {1, 2, 6, 7, 13}
    assert len(anotados) == len(raw) == len(validos) + bitacora["filas_excluidas"].sum()
    assert anotados.loc[anotados["viaje_id"].eq(9), "motivo_exclusion"].eq("id_conflictivo").all()
    assert (
        anotados.loc[anotados["viaje_id"].eq(11), "motivo_exclusion"].eq("inicio_fuera_mes").all()
    )
    assert (
        anotados.loc[anotados["viaje_id"].eq(12), "motivo_exclusion"].eq("sin_coordenadas").all()
    )


def test_no_modifica_raw_y_preserva_demografia(raw, coords):
    original = raw.copy(deep=True)
    _, validos, _ = c.limpiar_mes(raw, "2026_09", coords)
    pd.testing.assert_frame_equal(raw, original)
    corregido = validos.loc[validos["viaje_id"].eq(7)]
    assert corregido["anio_nacimiento"].isna().all()
    assert corregido["genero"].isna().all()
    assert corregido["anio_nacimiento_original"].eq(1800).all()
    assert corregido["genero_original"].eq("X").all()
    assert validos.loc[validos["viaje_id"].eq(2), "genero"].isna().all()


def test_id_entre_archivos_se_pone_en_cuarentena(raw, coords):
    anotados, validos, _ = c.limpiar_mes(raw, "2026_09", coords, ids_entre_archivos={1})
    assert 1 not in set(validos["viaje_id"])
    assert anotados.loc[anotados["viaje_id"].eq(1), "flag_id_entre_archivos"].all()


def test_politica_flexible_no_elimina_integridad(raw, coords):
    p = dict(
        c.POLITICA_DEFAULT,
        excluir_misma_estacion_corta=False,
        excluir_velocidad=False,
        excluir_duracion_larga=False,
        excluir_estacion_sin_catalogo=False,
        excluir_sin_coordenadas=False,
    )
    _, validos, _ = c.limpiar_mes(raw, "2026_09", coords, p)
    assert {3, 4, 5, 8, 12}.issubset(set(validos["viaje_id"]))
    assert {9, 10, 11}.isdisjoint(set(validos["viaje_id"]))


def test_validador_rechaza_resultado_alterado(raw, coords):
    anotados, _, bitacora = c.limpiar_mes(raw, "2026_09", coords)
    anotados.loc[anotados["viaje_id"].eq(9), "es_valido"] = True
    with pytest.raises(ValueError):
        c.validar_resultado(raw, anotados, bitacora, "2026_09", c.POLITICA_DEFAULT)


def test_control_no_confunde_usuario_de_proceso_con_usuario_viaje(raw):
    instante = pd.Timestamp("2026-10-24T02:00:00Z")
    controlado = c.agregar_control(
        raw,
        contexto={"usuario": "operador", "updt_dt": instante},
        entrada={"archivo": "data/raw/mes.csv", "sha256": "abc"},
    )
    assert controlado["usuario_id"].eq(10).all()
    assert controlado["usuario"].eq("operador").all()
    assert str(controlado["updt_dt"].dt.tz) == "UTC"
    with pytest.raises(ValueError):
        c.agregar_control(
            controlado, contexto={"usuario": "otro"}, entrada={"archivo": "otro", "sha256": "xyz"}
        )


def test_version_existente_no_se_sobrescribe(tmp_path):
    version = tmp_path / "version_001"
    version.mkdir()
    prueba = version / "preservar.txt"
    prueba.write_text("original")
    with pytest.raises(FileExistsError):
        c.generar_version(salida_base=tmp_path)
    assert prueba.read_text() == "original"


def test_version_no_admite_rutas(tmp_path):
    with pytest.raises(ValueError):
        c.generar_version("../otra", salida_base=tmp_path)


def test_materializacion_parquet_y_sello(tmp_path, monkeypatch, raw):
    entrada = tmp_path / "datos_abiertos_2026_09.csv"
    raw.to_csv(entrada, index=False)
    huella_raw = c.sha256_archivo(entrada)
    catalogo = tmp_path / "nomenclatura.csv"
    pd.DataFrame(
        {
            "id": [51, 60, 70],
            "name": ["A", "B", "C"],
            "latitude": [20.684, 20.6726, np.nan],
            "longitude": [-103.3555, -103.3535, np.nan],
        }
    ).to_csv(catalogo, index=False)
    monkeypatch.setattr(c, "ruta_viajes", lambda mes: entrada)
    monkeypatch.setattr(c, "columnas_mes", lambda mes: COLUMNAS)
    monkeypatch.setattr(c, "cargar_mes", lambda mes: raw.copy(deep=True))
    salida = c.generar_version(
        meses=["2026_09"],
        catalogo=catalogo,
        salida_base=tmp_path / "silver",
        usuario="operador_prueba",
    )
    resultado = c.verificar_version(salida)
    assert resultado["viajes_validos"] == 5
    assert resultado["viajes_excluidos"] == len(raw) - 5
    assert resultado["parquet_releidos"] == 2
    assert c.sha256_archivo(entrada) == huella_raw
    manifest = json.loads((salida / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mlflow_al_crear"]["run_id"] is None
    df = pd.read_parquet(salida / "viajes/viajes_2026_09.parquet")
    assert df["version_datos"].eq("version_001").all()
    assert df["usuario"].eq("operador_prueba").all()
    assert df["insrt_dt"].eq(df["updt_dt"]).all()
    with pytest.raises(FileExistsError):
        c.generar_version(salida_base=tmp_path / "silver")
    # Detecta cambios posteriores tanto en artefactos como en el manifiesto firmado por el sello.
    ruta = salida / "layout.md"
    ruta.write_text("alterado", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        c.verificar_version(salida)
