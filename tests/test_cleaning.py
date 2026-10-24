"""Pruebas del pipeline de limpieza (cleaning.py).

Basadas en la prueba original de limpiar_viajes (duplicados, nulos demográficos, viajes cortos entre
estaciones distintas y viajes largos), adaptadas a las funciones de limpieza actuales.
"""

import pandas as pd
import pytest

from proyecto_mlops_equipo_9.cleaning import limpiar_mes, validar

# Dos estaciones a ~1.3 km en línea recta
COORDS = pd.DataFrame(
    {"latitude": [20.6840, 20.6726], "longitude": [-103.3555, -103.3535]},
    index=pd.Index([51, 60], name="id"),
)


def viajes(filas):
    """DataFrame con el formato que entrega dataset.cargar_mes."""
    df = pd.DataFrame(filas, columns=["viaje_id", "usuario_id", "genero", "anio_nacimiento",
                                      "inicio", "fin", "origen_id", "destino_id"])
    df["inicio"] = pd.to_datetime(df["inicio"])
    df["fin"] = pd.to_datetime(df["fin"])
    df["genero"] = df["genero"].astype("category")
    df["duracion_min"] = (df["fin"] - df["inicio"]).dt.total_seconds() / 60
    return df


@pytest.fixture
def datos():
    return viajes([
        [1, 10, "F", 2000, "2026-09-01 10:00:00", "2026-09-01 10:10:00", 51, 60],  # válido
        [1, 10, "F", 2000, "2026-09-01 10:00:00", "2026-09-01 10:10:00", 51, 60],  # copia exacta
        [2, 20, None, None, "2026-09-01 11:00:00", "2026-09-01 11:15:00", 60, 51],  # válido, sin datos demográficos
        [3, 30, "M", 1995, "2026-09-01 11:30:00", "2026-09-01 11:30:30", 51, 60],  # 30 s entre estaciones a ~1.3 km
        [4, 40, "M", 1995, "2026-09-01 12:00:00", "2026-09-01 16:00:00", 51, 51],  # cuatro horas
        [5, 50, "F", 1990, "2026-09-01 13:00:00", "2026-09-01 13:00:20", 60, 60],  # desanclaje fallido
        [6, 60, "F", 1990, "2026-09-01 14:00:00", "2026-09-01 14:12:00", 60, 60],  # paseo real a la misma estación
        [7, 70, "M", 1800, "2026-09-01 15:00:00", "2026-09-01 15:08:00", 60, 51],  # edad imposible
        [8, 80, "M", 1990, "2026-09-01 16:00:00", "2026-09-01 16:05:00", 60, 999],  # estación fuera del catálogo
    ])


def test_elimina_solo_viajes_invalidos(datos):
    limpio, _ = limpiar_mes(datos, "2026_09", COORDS)
    assert sorted(limpio["viaje_id"]) == [1, 2, 6, 7]


def test_conserva_nulos_demograficos(datos):
    limpio, _ = limpiar_mes(datos, "2026_09", COORDS)
    assert limpio.loc[limpio["viaje_id"] == 2, "genero"].isna().all()


def test_edad_imposible_se_vuelve_faltante_sin_eliminar_el_viaje(datos):
    limpio, bitacora = limpiar_mes(datos, "2026_09", COORDS)
    assert limpio.loc[limpio["viaje_id"] == 7, "anio_nacimiento"].isna().all()
    assert bitacora[-1]["celdas_modificadas"] == 1


def test_bitacora_cuenta_cada_paso(datos):
    _, bitacora = limpiar_mes(datos, "2026_09", COORDS)
    eliminadas = {fila["paso"]: fila["filas_eliminadas"] for fila in bitacora}
    assert eliminadas["2 eliminar_duplicados"] == 1
    assert eliminadas["4 eliminar_desanclajes_fallidos"] == 1
    assert eliminadas["5 eliminar_estaciones_fuera_de_catalogo"] == 1
    assert eliminadas["6 eliminar_velocidad_imposible"] == 1
    assert eliminadas["7 eliminar_duracion_excesiva"] == 1
    assert bitacora[0]["filas_antes"] - bitacora[-1]["filas_despues"] == 5


def test_no_modifica_el_original(datos):
    copia = datos.copy()
    limpiar_mes(datos, "2026_09", COORDS)
    pd.testing.assert_frame_equal(datos, copia)


def test_validar_acepta_limpio_y_rechaza_crudo(datos):
    limpio, _ = limpiar_mes(datos, "2026_09", COORDS)
    validar(limpio, "2026_09", COORDS)
    with pytest.raises(ValueError):
        validar(datos, "2026_09", COORDS)
