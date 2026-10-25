"""Carga estándar de los datos crudos de MiBici.

Todas las etapas (notebooks, cleaning.py, features.py) leen los datos con estas funciones, así que
el manejo de encoding, nombres de columna y tipos vive en un solo lugar.

Uso:
    from proyecto_mlops_equipo_9.dataset import cargar_mes, cargar_estaciones, meses_disponibles

    viajes = cargar_mes("2026-05")
    estaciones = cargar_estaciones()
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

from proyecto_mlops_equipo_9.config import RAW_DATA_DIR

VIAJES_DIR = RAW_DATA_DIR / "viajes"
ESTACIONES_DIR = RAW_DATA_DIR / "estaciones"

# Columnas estándar, en orden. Los archivos originales usan nombres con acentos y mayúsculas.
COLUMNAS = ["viaje_id", "usuario_id", "genero", "anio_nacimiento", "inicio", "fin", "origen_id", "destino_id"]

_PREFIJOS = {
    "viaje": "viaje_id",
    "usuario": "usuario_id",
    "genero": "genero",
    "inicio": "inicio",
    "fin": "fin",
    "origen": "origen_id",
    "destino": "destino_id",
}
_FORMATO_FECHA = "%Y-%m-%d %H:%M:%S"


def nombre_estandar(columna: str) -> str:
    """'Año_de_nacimiento' -> 'anio_nacimiento', 'Inicio_del_viaje' -> 'inicio', etc."""
    limpio = unicodedata.normalize("NFKD", columna).encode("ascii", "ignore").decode()
    limpio = re.sub(r"\W+", "_", limpio.strip().lower()).strip("_")
    if "nacimiento" in limpio:
        return "anio_nacimiento"
    for prefijo, nombre in _PREFIJOS.items():
        if limpio.startswith(prefijo):
            return nombre
    return limpio  # columna desconocida: se conserva para que se note


def ruta_viajes(periodo: str) -> Path:
    """'2026-05' o '2026_05' -> data/raw/viajes/datos_abiertos_2026_05.csv"""
    return VIAJES_DIR / f"datos_abiertos_{periodo.replace('-', '_')}.csv"


def meses_disponibles() -> list[str]:
    """Meses descargados en data/raw/viajes, como 'AAAA_MM', en orden."""
    return sorted(p.stem.removeprefix("datos_abiertos_") for p in VIAJES_DIR.glob("datos_abiertos_*.csv"))


def _leer_csv(ruta: Path, **kwargs) -> pd.DataFrame:
    """Lee con utf-8 y, si falla, con latin-1 (julio 2025, por ejemplo, viene en latin-1)."""
    try:
        return pd.read_csv(ruta, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(ruta, encoding="latin-1", **kwargs)


def columnas_mes(periodo: str) -> list[str]:
    """Columnas estándar de un mes, leyendo solo el encabezado (rápido, para revisar el esquema)."""
    return [nombre_estandar(c) for c in _leer_csv(ruta_viajes(periodo), nrows=0).columns]


def _a_fecha(serie: pd.Series) -> pd.Series:
    fechas = pd.to_datetime(serie, format=_FORMATO_FECHA, errors="coerce")
    if fechas.isna().mean() > 0.5:  # otro formato de fecha, posible en años anteriores
        fechas = pd.to_datetime(serie, errors="coerce", dayfirst=True)
    return fechas


def cargar_mes(periodo: str) -> pd.DataFrame:
    """Carga un mes de viajes con columnas estándar, fechas convertidas y la duración en minutos.

    `NULL` (sin comillas) en el CSV se lee como faltante. No se elimina ninguna fila: la limpieza
    es responsabilidad de cleaning.py.
    """
    ruta = ruta_viajes(periodo)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Desde la raíz del repo corre: make data")

    df = _leer_csv(ruta, na_values=["NULL", ""], low_memory=False)
    df = df.rename(columns=nombre_estandar)
    df["inicio"] = _a_fecha(df["inicio"])
    df["fin"] = _a_fecha(df["fin"])
    df["genero"] = df["genero"].astype("category")
    df["duracion_min"] = (df["fin"] - df["inicio"]).dt.total_seconds() / 60
    return df


def cargar_estaciones(ruta: Path | None = None) -> pd.DataFrame:
    """Catálogo de estaciones (el más reciente si no se indica la ruta).

    Agrega `dada_de_baja` (nombre con prefijo 'ex-') y `codigo` (p. ej. 'GDL-020'). La llave para unir
    con los viajes es `id`: algunos códigos se reutilizaron para estaciones nuevas.
    """
    if ruta is None:
        catalogos = sorted(ESTACIONES_DIR.glob("nomenclatura_*.csv"))
        if not catalogos:
            raise FileNotFoundError(f"No hay catálogo en {ESTACIONES_DIR}. Corre: make data")
        ruta = catalogos[-1]
    estaciones = _leer_csv(ruta)
    estaciones.columns = [c.strip().lower() for c in estaciones.columns]
    estaciones["dada_de_baja"] = estaciones["name"].str.startswith("ex-")
    estaciones["codigo"] = estaciones["name"].str.extract(r"((?:GDL|ZPN|TLQ)-\d{3})", expand=False)
    return estaciones
