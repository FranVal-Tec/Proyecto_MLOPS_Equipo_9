import pandas as pd


def limpiar_viajes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y valida registros individuales de viajes MiBici.

    Conserva los registros con datos demográficos faltantes
    y marca duraciones sospechosas sin eliminarlas.
    """

    # Trabajar sobre una copia para no modificar el original
    limpio = df.copy()

    # Eliminar duplicados exactos
    limpio = limpio.drop_duplicates().copy()

    # Estandarizar nombres de columnas
    limpio = limpio.rename(columns={
        "Viaje_Id": "viaje_id",
        "Usuario_Id": "usuario_id",
        "Genero": "genero",
        "Año_de_nacimiento": "anio_nacimiento",
        "Inicio_del_viaje": "inicio",
        "Fin_del_viaje": "fin",
        "Origen_Id": "origen_id",
        "Destino_Id": "destino_id"
    })

    # Convertir fechas
    limpio["inicio"] = pd.to_datetime(
        limpio["inicio"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    limpio["fin"] = pd.to_datetime(
        limpio["fin"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    # Calcular duración del viaje
    limpio["duracion_min"] = (
        limpio["fin"] - limpio["inicio"]
    ).dt.total_seconds() / 60

    # Identificar registros sospechosos
    limpio["viaje_corto"] = (
        limpio["duracion_min"] >= 0
    ) & (
        limpio["duracion_min"] < 1
    )

    limpio["corto_distinta_estacion"] = (
        limpio["viaje_corto"]
        & (limpio["origen_id"] != limpio["destino_id"])
    )

    limpio["duracion_larga"] = limpio["duracion_min"] > 180

    return limpio