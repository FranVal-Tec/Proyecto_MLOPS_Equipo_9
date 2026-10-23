"""Limpieza de los viajes de MiBici: data/raw/viajes -> data/interim/viajes_limpios.

Uso (desde la raíz del repo, con el entorno activado):

    python -m proyecto_mlops_equipo_9.cleaning                       # todos los meses descargados
    python -m proyecto_mlops_equipo_9.cleaning --meses 2026_05 --meses 2026_06
    python -m proyecto_mlops_equipo_9.cleaning --mlflow              # además registra la versión en MLflow

Las reglas y sus umbrales vienen del notebook 1.1-LV-reglas-limpieza, donde cada una está justificada
con datos. Si se cambia un umbral, hay que cambiarlo aquí y documentarlo allá.

Salidas:
    data/interim/viajes_limpios/viajes_limpios_AAAA_MM.parquet   un archivo por mes
    data/interim/viajes_limpios/bitacora_limpieza.csv            filas eliminadas por mes y por paso
    data/interim/viajes_limpios/manifest_limpieza.csv            tamaño y SHA-256 de cada archivo limpio
"""

import numpy as np
import pandas as pd
import typer
from loguru import logger

from proyecto_mlops_equipo_9.config import INTERIM_DATA_DIR
from proyecto_mlops_equipo_9.dataset import COLUMNAS, cargar_estaciones, cargar_mes, meses_disponibles
from proyecto_mlops_equipo_9.download import sha256

SALIDA_DIR = INTERIM_DATA_DIR / "viajes_limpios"
BITACORA = SALIDA_DIR / "bitacora_limpieza.csv"
MANIFEST = SALIDA_DIR / "manifest_limpieza.csv"
EXPERIMENTO_MLFLOW = "mibici-datos"

# ------ Contrato de datos (justificación en el notebook 1.1) ----------
GENEROS_VALIDOS = {"M", "F"}
DURACION_MIN_MISMA_ESTACION_S = 60  # T5: desanclaje fallido
VELOCIDAD_MAX_KMH = 30              # T6: velocidad en línea recta imposible
DURACION_MAX_MIN = 180              # T7: préstamo demasiado largo
EDAD_MIN, EDAD_MAX = 16, 90         # T9: edad plausible

app = typer.Typer()


def velocidad_kmh(df: pd.DataFrame, coords: pd.DataFrame) -> pd.Series:
    """Velocidad en línea recta (km/h) con la fórmula del haversine. NaN si alguna estación no está en el catálogo."""
    origen = coords.reindex(df["origen_id"]).to_numpy()
    destino = coords.reindex(df["destino_id"]).to_numpy()
    lat1, lon1, lat2, lon2 = (np.radians(a) for a in (origen[:, 0], origen[:, 1], destino[:, 0], destino[:, 1]))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    distancia_km = 2 * 6371 * np.arcsin(np.sqrt(a))
    with np.errstate(divide="ignore", invalid="ignore"):
        velocidad = distancia_km / (df["duracion_min"].to_numpy() / 60)
    return pd.Series(velocidad, index=df.index)


# ------ Pasos de limpieza: funciones puras (reciben un DataFrame y regresan uno nuevo) ----------

def eliminar_fechas_invalidas(df):
    return df[df["inicio"].notna() & df["fin"].notna()]


def eliminar_duplicados(df):
    return df.drop_duplicates(subset="viaje_id")


def eliminar_duracion_no_positiva(df):
    return df[df["duracion_min"] > 0]


def eliminar_desanclajes_fallidos(df):
    fallido = (df["origen_id"] == df["destino_id"]) & (df["duracion_min"] * 60 < DURACION_MIN_MISMA_ESTACION_S)
    return df[~fallido]


def eliminar_estaciones_fuera_de_catalogo(df, coords):
    return df[df["origen_id"].isin(coords.index) & df["destino_id"].isin(coords.index)]


def eliminar_velocidad_imposible(df, coords):
    return df[velocidad_kmh(df, coords) <= VELOCIDAD_MAX_KMH]


def eliminar_duracion_excesiva(df):
    return df[df["duracion_min"] <= DURACION_MAX_MIN]


def corregir_edad_imposible(df):
    edad = df["inicio"].dt.year - df["anio_nacimiento"]
    return df.assign(anio_nacimiento=df["anio_nacimiento"].where(edad.between(EDAD_MIN, EDAD_MAX)))


def limpiar_mes(df: pd.DataFrame, mes: str, coords: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Aplica los pasos en orden (el 5 va antes del 6: la velocidad necesita la ubicación). Regresa el mes y su bitácora."""
    pasos = [
        ("1 eliminar_fechas_invalidas", eliminar_fechas_invalidas),
        ("2 eliminar_duplicados", eliminar_duplicados),
        ("3 eliminar_duracion_no_positiva", eliminar_duracion_no_positiva),
        ("4 eliminar_desanclajes_fallidos", eliminar_desanclajes_fallidos),
        ("5 eliminar_estaciones_fuera_de_catalogo", lambda d: eliminar_estaciones_fuera_de_catalogo(d, coords)),
        ("6 eliminar_velocidad_imposible", lambda d: eliminar_velocidad_imposible(d, coords)),
        ("7 eliminar_duracion_excesiva", eliminar_duracion_excesiva),
        ("8 corregir_edad_imposible", corregir_edad_imposible),
    ]
    bitacora = []
    for paso, funcion in pasos:
        antes, faltantes_antes = len(df), int(df["anio_nacimiento"].isna().sum())
        df = funcion(df)
        # Las celdas modificadas solo se cuentan en pasos que no eliminan filas (paso 8)
        celdas = int(df["anio_nacimiento"].isna().sum()) - faltantes_antes if len(df) == antes else 0
        bitacora.append({
            "mes": mes, "paso": paso, "filas_antes": antes, "filas_despues": len(df),
            "filas_eliminadas": antes - len(df), "celdas_modificadas": celdas,
        })
    return df.reset_index(drop=True), bitacora


def validar(df: pd.DataFrame, mes: str, coords: pd.DataFrame) -> None:
    """Comprueba que un mes limpio cumple el contrato; si algo falla, se detiene con un mensaje claro."""
    anio, num_mes = map(int, mes.split("_"))
    misma = df["origen_id"] == df["destino_id"]
    edad = df["inicio"].dt.year - df["anio_nacimiento"]
    reglas = {
        "columnas del contrato": list(df.columns[: len(COLUMNAS)]) == COLUMNAS,
        "fechas válidas": bool(df["inicio"].notna().all() and df["fin"].notna().all()),
        "inicio dentro del mes": bool(((df["inicio"].dt.year == anio) & (df["inicio"].dt.month == num_mes)).all()),
        "sin viajes repetidos": df["viaje_id"].is_unique,
        "duración en (0, 180] min": bool(df["duracion_min"].between(0, DURACION_MAX_MIN, inclusive="right").all()),
        "sin desanclajes fallidos": not (misma & (df["duracion_min"] * 60 < DURACION_MIN_MISMA_ESTACION_S)).any(),
        "estaciones en catálogo": bool(df["origen_id"].isin(coords.index).all() and df["destino_id"].isin(coords.index).all()),
        "velocidad posible": bool((velocidad_kmh(df, coords) <= VELOCIDAD_MAX_KMH).all()),
        "género M/F o faltante": bool((df["genero"].isin(GENEROS_VALIDOS) | df["genero"].isna()).all()),
        "edad 16-90 o faltante": bool((edad.between(EDAD_MIN, EDAD_MAX) | edad.isna()).all()),
    }
    fallas = [regla for regla, cumple in reglas.items() if not cumple]
    if fallas:
        raise ValueError(f"{mes} no cumple el contrato: {', '.join(fallas)}")


def guardar_bitacora(nueva: pd.DataFrame) -> pd.DataFrame:
    """Actualiza la bitácora: reemplaza los meses procesados en esta corrida y conserva los demás."""
    if BITACORA.exists():
        anterior = pd.read_csv(BITACORA, dtype={"mes": str})
        nueva = pd.concat([anterior[~anterior["mes"].isin(nueva["mes"])], nueva])
    nueva = nueva.sort_values(["mes", "paso"]).reset_index(drop=True)
    nueva.to_csv(BITACORA, index=False)
    return nueva


def escribir_manifest() -> pd.DataFrame:
    archivos = sorted(SALIDA_DIR.glob("viajes_limpios_*.parquet"))
    manifest = pd.DataFrame([
        {"archivo": ruta.name, "bytes": ruta.stat().st_size, "sha256": sha256(ruta)} for ruta in archivos
    ])
    manifest.to_csv(MANIFEST, index=False)
    return manifest


def registrar_en_mlflow(bitacora: pd.DataFrame, meses: list[str]) -> None:
    """Registra esta versión de los datos limpios: umbrales, conteos y archivos de trazabilidad."""
    import mlflow  # solo se importa si se pide --mlflow

    crudos = int(bitacora.groupby("mes")["filas_antes"].first().sum())
    limpios = int(bitacora.groupby("mes")["filas_despues"].last().sum())
    por_paso = bitacora.groupby("paso")[["filas_eliminadas", "celdas_modificadas"]].sum()

    mlflow.set_experiment(EXPERIMENTO_MLFLOW)
    with mlflow.start_run(run_name="limpieza"):
        mlflow.set_tag("etapa", "limpieza")
        mlflow.log_params({
            "meses": f"{meses[0]}..{meses[-1]}", "n_meses": len(meses),
            "duracion_min_misma_estacion_s": DURACION_MIN_MISMA_ESTACION_S,
            "velocidad_max_kmh": VELOCIDAD_MAX_KMH, "duracion_max_min": DURACION_MAX_MIN,
            "edad_min": EDAD_MIN, "edad_max": EDAD_MAX,
        })
        mlflow.log_metrics({
            "viajes_crudos": crudos, "viajes_limpios": limpios,
            "pct_eliminado": round(100 * (1 - limpios / crudos), 3),
        })
        for paso, fila in por_paso.iterrows():
            nombre = paso.replace(" ", "_")
            mlflow.log_metric(f"eliminados/{nombre}", int(fila["filas_eliminadas"]))
            if fila["celdas_modificadas"]:
                mlflow.log_metric(f"celdas/{nombre}", int(fila["celdas_modificadas"]))
        mlflow.log_artifact(str(BITACORA))
        mlflow.log_artifact(str(MANIFEST))
    logger.success(f"Versión registrada en MLflow (experimento '{EXPERIMENTO_MLFLOW}')")


@app.command()
def main(
    meses: list[str] = typer.Option(None, help="Meses a limpiar (AAAA_MM). Si se omite, todos los descargados."),
    mlflow: bool = typer.Option(False, "--mlflow", help="Registra la versión de los datos limpios en MLflow."),
):
    """Limpia los viajes de data/raw/viajes y guarda un Parquet por mes en data/interim/viajes_limpios."""
    meses = [m.replace("-", "_") for m in meses] if meses else meses_disponibles()
    if not meses:
        raise typer.BadParameter("No hay meses descargados. Corre primero: make data")

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)
    coords = cargar_estaciones().set_index("id")[["latitude", "longitude"]]

    bitacora = []
    for mes in meses:
        limpio, bitacora_mes = limpiar_mes(cargar_mes(mes), mes, coords)
        validar(limpio, mes, coords)
        limpio.to_parquet(SALIDA_DIR / f"viajes_limpios_{mes}.parquet", index=False)
        bitacora.extend(bitacora_mes)
        crudos = bitacora_mes[0]["filas_antes"]
        logger.info(f"{mes}: {crudos:,} → {len(limpio):,} viajes ({1 - len(limpio) / crudos:.1%} eliminados)")

    bitacora = guardar_bitacora(pd.DataFrame(bitacora))
    escribir_manifest()
    total_crudos = bitacora.groupby("mes")["filas_antes"].first().sum()
    total_limpios = bitacora.groupby("mes")["filas_despues"].last().sum()
    etiqueta = "mes limpio y validado" if len(meses) == 1 else "meses limpios y validados"
    logger.success(f"{len(meses)} {etiqueta}. Bitácora completa: {total_crudos:,} → {total_limpios:,} viajes")

    if mlflow:
        registrar_en_mlflow(bitacora[bitacora["mes"].isin(meses)], meses)


if __name__ == "__main__":
    app()
