"""Limpieza reproducible de viajes MiBici y materialización de versiones silver.

Ejecutar desde la raíz: python -m proyecto_mlops_equipo_9.cleaning --version version_001
MLflow se registra en una sesión asistida separada; este módulo no crea runs.
Las reglas se explican y diagnostican en el notebook final, que importa estas funciones.
"""

import argparse
import gc
import getpass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import uuid

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from proyecto_mlops_equipo_9.config import PROJ_ROOT
from proyecto_mlops_equipo_9.dataset import (
    COLUMNAS,
    cargar_estaciones,
    cargar_mes,
    columnas_mes,
    meses_disponibles,
    ruta_viajes,
)

POLITICA_DEFAULT = {
    "version_metodologia": "1.0.0",
    "min_misma_estacion_s": 60,
    "max_velocidad_kmh": 30,
    "max_duracion_min": 180,
    "edad_min": 16,
    "edad_max": 90,
    "umbral_dia_bajo": 0.6,
    "excluir_misma_estacion_corta": True,
    "excluir_estacion_sin_catalogo": True,
    "excluir_sin_coordenadas": True,
    "excluir_velocidad": True,
    "excluir_duracion_larga": True,
}


def velocidad_linea_recta(df, coords):
    origen = coords.reindex(df["origen_id"]).to_numpy(dtype=float)
    destino = coords.reindex(df["destino_id"]).to_numpy(dtype=float)
    lat1, lon1, lat2, lon2 = map(
        np.radians, [origen[:, 0], origen[:, 1], destino[:, 0], destino[:, 1]]
    )
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    km = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    minutos = df["duracion_min"].to_numpy(dtype=float)
    v = np.full(len(df), np.nan)
    np.divide(km * 60, minutos, out=v, where=np.isfinite(minutos) & (minutos > 0))
    return pd.Series(v, index=df.index)


def anotar_viajes(raw, mes, coords, politica, ids_entre_archivos=frozenset()):
    df = raw.copy(deep=True).reset_index(drop=True)
    df["fila_csv"] = np.arange(len(df)) + 2  # encabezado en línea 1
    df["mes_archivo"] = mes
    unicos = df.drop_duplicates(subset=COLUMNAS)
    ids_conflicto = set(unicos.loc[unicos["viaje_id"].duplicated(keep=False), "viaje_id"].dropna())
    anio, numero = map(int, mes.split("_"))
    df["flag_duplicado_exacto"] = df.duplicated(subset=COLUMNAS)
    df["flag_id_conflictivo"] = df["viaje_id"].isin(ids_conflicto)
    df["flag_id_entre_archivos"] = df["viaje_id"].isin(ids_entre_archivos)
    df["flag_id_esencial_nulo"] = df[["viaje_id", "origen_id", "destino_id"]].isna().any(axis=1)
    df["flag_fecha_invalida"] = df[["inicio", "fin"]].isna().any(axis=1)
    df["flag_fuera_mes"] = df["inicio"].notna() & (
        (df["inicio"].dt.year != anio) | (df["inicio"].dt.month != numero)
    )
    df["flag_duracion_no_positiva"] = df["duracion_min"].le(0)
    df["viaje_corto"] = df["duracion_min"].ge(0) & df["duracion_min"].lt(
        politica["min_misma_estacion_s"] / 60
    )
    misma = df["origen_id"].eq(df["destino_id"])
    df["corto_distinta_estacion"] = df["viaje_corto"] & ~misma
    df["flag_misma_estacion_corta"] = df["viaje_corto"] & misma
    df["duracion_larga"] = df["duracion_min"].gt(politica["max_duracion_min"])
    df["flag_estacion_sin_catalogo"] = ~df["origen_id"].isin(coords.index) | ~df[
        "destino_id"
    ].isin(coords.index)
    ids_con_coords = coords.dropna().index
    df["flag_sin_coordenadas"] = ~df["flag_estacion_sin_catalogo"] & (
        ~df["origen_id"].isin(ids_con_coords) | ~df["destino_id"].isin(ids_con_coords)
    )
    df["velocidad_kmh"] = velocidad_linea_recta(df, coords)
    df["flag_velocidad_imposible"] = df["velocidad_kmh"].gt(politica["max_velocidad_kmh"])
    df["anio_nacimiento_original"] = df["anio_nacimiento"]
    edad = df["inicio"].dt.year - pd.to_numeric(df["anio_nacimiento"], errors="coerce")
    df["flag_edad_imposible"] = edad.notna() & ~edad.between(
        politica["edad_min"], politica["edad_max"]
    )
    df["flag_nacimiento_no_numerico"] = (
        df["anio_nacimiento"].notna()
        & pd.to_numeric(df["anio_nacimiento"], errors="coerce").isna()
    )
    df["anio_nacimiento"] = pd.to_numeric(df["anio_nacimiento"], errors="coerce").mask(
        df["flag_edad_imposible"]
    )
    df["edad"] = edad.mask(df["flag_edad_imposible"])
    df["genero_original"] = df["genero"]
    df["flag_genero_invalido"] = df["genero"].notna() & ~df["genero"].isin(["M", "F"])
    df["genero"] = df["genero"].mask(df["flag_genero_invalido"])
    return df


def reglas_seleccion(p):
    return [
        ("id_esencial_nulo", "flag_id_esencial_nulo", True),
        ("fecha_invalida", "flag_fecha_invalida", True),
        ("inicio_fuera_mes", "flag_fuera_mes", True),
        ("id_conflictivo", "flag_id_conflictivo", True),
        ("id_entre_archivos", "flag_id_entre_archivos", True),
        ("duplicado_exacto", "flag_duplicado_exacto", True),
        ("duracion_no_positiva", "flag_duracion_no_positiva", True),
        ("misma_estacion_corta", "flag_misma_estacion_corta", p["excluir_misma_estacion_corta"]),
        (
            "estacion_sin_catalogo",
            "flag_estacion_sin_catalogo",
            p["excluir_estacion_sin_catalogo"],
        ),
        ("sin_coordenadas", "flag_sin_coordenadas", p["excluir_sin_coordenadas"]),
        ("velocidad_imposible", "flag_velocidad_imposible", p["excluir_velocidad"]),
        ("duracion_larga", "duracion_larga", p["excluir_duracion_larga"]),
    ]


def seleccionar_viajes(anotado, politica):
    df = anotado.copy()
    df["motivo_exclusion"] = pd.Series(pd.NA, index=df.index, dtype="string")
    disponible = pd.Series(True, index=df.index)
    bitacora = []
    for orden, (nombre, bandera, activa) in enumerate(reglas_seleccion(politica), 1):
        antes = int(disponible.sum())
        rechazadas = (
            disponible & df[bandera].fillna(False) if activa else pd.Series(False, index=df.index)
        )
        df.loc[rechazadas, "motivo_exclusion"] = nombre
        disponible &= ~rechazadas
        bitacora.append(
            {
                "mes": df["mes_archivo"].iloc[0] if len(df) else "",
                "orden": orden,
                "paso": nombre,
                "activa": activa,
                "filas_antes": antes,
                "filas_despues": int(disponible.sum()),
                "filas_excluidas": int(rechazadas.sum()),
            }
        )
    df["es_valido"] = disponible
    return df, pd.DataFrame(bitacora)


def validar_resultado(raw, anotado, bitacora, mes, politica):
    valido = anotado.loc[anotado["es_valido"]]
    # ValueError permanece activo incluso si Python se ejecuta con optimizaciones.
    fallas = []
    if len(raw) != len(anotado):
        fallas.append("se perdieron filas de la vista anotada")
    if len(raw) != len(valido) + int(bitacora["filas_excluidas"].sum()):
        fallas.append("balance de filas")
    if not valido["viaje_id"].is_unique:
        fallas.append("IDs repetidos")
    if valido["motivo_exclusion"].notna().any():
        fallas.append("motivo en fila válida")
    if anotado.loc[~anotado["es_valido"], "motivo_exclusion"].isna().any():
        fallas.append("rechazo sin motivo")
    for nombre, bandera, activa in reglas_seleccion(politica):
        if activa and valido[bandera].any():
            fallas.append(nombre)
    if not (valido["genero"].isna() | valido["genero"].isin(["M", "F"])).all():
        fallas.append("género")
    edad = valido["inicio"].dt.year - valido["anio_nacimiento"]
    if not (edad.isna() | edad.between(politica["edad_min"], politica["edad_max"])).all():
        fallas.append("edad")
    if fallas:
        raise ValueError(f"{mes}: " + ", ".join(fallas))
    return True


def limpiar_mes(raw, mes, coords, politica=None, ids_entre_archivos=frozenset()):
    """Devuelve (anotados, validos, bitacora), sin modificar los datos de entrada."""
    politica = dict(POLITICA_DEFAULT if politica is None else politica)
    anotados = anotar_viajes(raw, mes, coords, politica, ids_entre_archivos)
    anotados, bitacora = seleccionar_viajes(anotados, politica)
    bitacora["mes"] = mes
    validar_resultado(raw, anotados, bitacora, mes, politica)
    return anotados, anotados.loc[anotados["es_valido"]].copy(), bitacora


CONTROL_DESCRIPCIONES = {
    "usuario": "Cuenta que ejecutó la limpieza; no es el usuario de MiBici (usuario_id).",
    "insrt_dt": "Fecha de materialización de la versión en UTC, con zona horaria.",
    "updt_dt": "Fecha de última escritura UTC; igual a insrt_dt en una versión inmutable.",
    "version_datos": "Etiqueta de la entrega, por ejemplo version_001.",
    "version_limpieza": "Versión semántica de la política de limpieza.",
    "dataset_id": "SHA-256 de insumos, código, política y entorno; no incluye fecha ni usuario.",
    "lote_id": "UUID de la ejecución que materializó los archivos.",
    "archivo_origen": "Ruta relativa al proyecto del CSV original.",
    "sha256_origen": "SHA-256 de todos los bytes del CSV original.",
    "catalogo_version": "Nombre del catálogo de estaciones fijado para esta versión.",
    "catalogo_sha256": "SHA-256 del catálogo utilizado.",
    "codigo_limpieza_sha256": "SHA-256 del archivo cleaning.py ejecutado.",
    "fila_csv": "Número lógico de registro en el CSV contando el encabezado como 1.",
    "mes_archivo": "Periodo del archivo original, AAAA_MM.",
    "es_valido": "Resultado de la política; True en viajes/ y False en revision/.",
    "motivo_exclusion": "Primer motivo de rechazo secuencial; faltante para viajes válidos.",
    "anio_nacimiento_original": "Valor declarado antes de corregir la demografía.",
    "genero_original": "Categoría declarada antes de corregir la demografía.",
}


def sha256_archivo(ruta):
    h = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def guardar_json(ruta, contenido):
    Path(ruta).write_text(
        json.dumps(contenido, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def agregar_control(df, *, contexto, entrada):
    """Añade trazabilidad a cada fila sin alterar los campos originales de negocio."""
    colisiones = (set(contexto) | {"archivo_origen", "sha256_origen"}) & set(df.columns)
    if colisiones:
        raise ValueError(f"Columnas de control ya presentes: {sorted(colisiones)}")
    resultado = df.copy()
    for columna, valor in contexto.items():
        resultado[columna] = valor
    resultado["archivo_origen"] = entrada["archivo"]
    resultado["sha256_origen"] = entrada["sha256"]
    return resultado


def _escribir_parquet_verificado(frame, ruta, carpeta_version):
    frame.to_parquet(ruta, engine="pyarrow", compression="zstd", index=False)
    recuperado = pd.read_parquet(ruta, engine="pyarrow")
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), recuperado, check_dtype=True)
    return {
        "archivo": ruta.relative_to(carpeta_version).as_posix(),
        "filas": len(recuperado),
        "bytes": ruta.stat().st_size,
        "sha256": sha256_archivo(ruta),
        "tipo": "viajes_validos" if ruta.parent.name == "viajes" else "viajes_excluidos",
    }


def _evidencia_git():
    git = shutil.which("git")
    if not git:
        return {"commit": None, "estado": "git_no_disponible; usar snapshots y SHA-256"}
    try:
        commit = subprocess.run(
            [git, "-C", str(PROJ_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
        estado = subprocess.run(
            [git, "-C", str(PROJ_ROOT), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.splitlines()
        return {
            "commit_base": commit,
            "arbol_con_cambios": bool(estado),
            "archivos_con_cambios": estado,
            "nota": "El código ejecutado está preservado en fuentes/codigo; el commit base puede no incluirlo.",
        }
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "estado": "no_consultable; usar snapshots y SHA-256"}


def _layout_markdown(schema, control):
    lineas = [
        "# Layout de viajes silver",
        "",
        "Ambas carpetas usan el mismo esquema: viajes/ contiene filas válidas y revision/ las excluidas.",
        "Las columnas de negocio se mantienen; las de control identifican el proceso y la versión.",
        "",
        "| Campo | Tipo Arrow / Parquet | Definición |",
        "|---|---|---|",
    ]
    negocio = {
        "viaje_id": "Identificador del viaje.",
        "usuario_id": "Identificador del usuario de MiBici.",
        "inicio": "Fecha y hora original de inicio; zona del servicio no declarada en el CSV.",
        "fin": "Fecha y hora original de fin; no se convierte automáticamente a UTC.",
        "origen_id": "Estación de origen.",
        "destino_id": "Estación de destino.",
        "genero": "Categoría M/F o faltante tras corrección.",
        "anio_nacimiento": "Año de nacimiento numérico o faltante tras corrección.",
        "duracion_min": "Diferencia entre fin e inicio en minutos.",
        "velocidad_kmh": "Distancia Haversine / duración; faltante si no es calculable.",
        "edad": "Año del viaje menos nacimiento; faltante si está fuera de política.",
        "viaje_corto": "Duración no negativa y menor al umbral configurado.",
        "corto_distinta_estacion": "Viaje corto con distinto origen y destino.",
        "duracion_larga": "Duración superior al límite configurado.",
    }
    for campo in schema:
        descripcion = control.get(
            campo.name, negocio.get(campo.name, "Indicador de calidad calculado antes de filtrar.")
        )
        lineas.append(f"| `{campo.name}` | `{campo.type}` | {descripcion} |")
    lineas.extend(
        [
            "",
            "Los tiempos insrt_dt y updt_dt son UTC del proceso; inicio y fin mantienen el tiempo de negocio.",
            "No se actualizan archivos dentro de una versión completa: los cambios producen otra versión.",
            "Las banderas originales pueden solaparse; motivo_exclusion corresponde al primer paso que rechazó la fila.",
        ]
    )
    return "\n".join(lineas) + "\n"


def verificar_version(carpeta, *, completa=True):
    """Relee todos los Parquet y verifica hashes, layout, control y balance contra el manifiesto."""
    carpeta = Path(carpeta)
    manifest = json.loads((carpeta / "manifest.json").read_text(encoding="utf-8"))
    if completa and not (carpeta / "_SUCCESS.json").is_file():
        raise ValueError("La versión no tiene marcador de publicación completa.")
    if completa:
        sello = json.loads((carpeta / "_SUCCESS.json").read_text(encoding="utf-8"))
        if (
            sello["manifest_sha256"] != sha256_archivo(carpeta / "manifest.json")
            or sello["verificacion_sha256"] != sha256_archivo(carpeta / "verificacion.json")
            or sello["dataset_id"] != manifest["dataset_id"]
        ):
            raise ValueError("La versión publicada fue modificada o su sello no coincide.")
    totales = {"viajes_validos": 0, "viajes_excluidos": 0}
    entradas = {e["mes"]: e for e in manifest["entradas"] if e["tipo"] == "viajes"}
    columnas_control = set(manifest["campos_control_constantes"])
    for salida in manifest["salidas"]:
        ruta = (carpeta / salida["archivo"]).resolve()
        if not ruta.is_relative_to(carpeta.resolve()):
            raise ValueError("Ruta de salida fuera de la versión.")
        if sha256_archivo(ruta) != salida["sha256"]:
            raise ValueError(f"SHA-256 incompatible: {salida['archivo']}")
        if salida["tipo"] not in totales:
            continue
        frame = pd.read_parquet(ruta)
        if len(frame) != salida["filas"]:
            raise ValueError(f"Conteo incompatible: {ruta.name}")
        if not columnas_control.issubset(frame.columns):
            raise ValueError("Faltan campos de control.")
        for columna, esperado in manifest["campos_control_constantes"].items():
            esperado = pd.Timestamp(esperado) if columna in {"insrt_dt", "updt_dt"} else esperado
            if not frame[columna].eq(esperado).all():
                raise ValueError(f"Control incompatible: {columna}")
        for mes, entrada in entradas.items():
            filas = frame["mes_archivo"].eq(mes)
            if not frame.loc[filas, "sha256_origen"].eq(entrada["sha256"]).all():
                raise ValueError("Huella de origen incompatible.")
        if not frame["mes_archivo"].isin(entradas).all():
            raise ValueError("Mes de origen desconocido.")
        es_valido = salida["tipo"] == "viajes_validos"
        if not frame["es_valido"].eq(es_valido).all():
            raise ValueError("Estado de fila incompatible con su carpeta.")
        if es_valido and (
            not frame["viaje_id"].is_unique or frame["motivo_exclusion"].notna().any()
        ):
            raise ValueError("Filas válidas con ID repetido o motivo de rechazo.")
        if not es_valido and frame["motivo_exclusion"].isna().any():
            raise ValueError("Filas excluidas sin motivo.")
        totales[salida["tipo"]] += len(frame)
    if totales["viajes_validos"] != manifest["resumen"]["validos"]:
        raise ValueError("Total válido incompatible.")
    if totales["viajes_excluidos"] != manifest["resumen"]["excluidos"]:
        raise ValueError("Total excluido incompatible.")
    if sum(totales.values()) != manifest["resumen"]["originales"]:
        raise ValueError("Balance de filas incompatible.")
    return {
        "estado": "CORRECTO",
        "parquet_releidos": sum(s["tipo"] in totales for s in manifest["salidas"]),
        "archivos_con_hash_verificado": len(manifest["salidas"]),
        **totales,
    }


def generar_version(
    version="version_001", *, usuario=None, meses=None, catalogo=None, salida_base=None
):
    """Materializa una versión silver completa y verificada. No registra nada en MLflow."""
    if not re.fullmatch(r"version_[0-9]{3,}", version):
        raise ValueError("La versión debe tener formato version_001, version_002, etc.")
    usuario = (usuario or getpass.getuser()).strip()
    if not usuario:
        raise ValueError("Se necesita un usuario de ejecución no vacío.")
    politica = dict(POLITICA_DEFAULT)
    salida_base = Path(salida_base) if salida_base else PROJ_ROOT / "data/silver"
    destino = salida_base / version
    if destino.exists():
        raise FileExistsError(f"La versión ya existe y no se sobrescribe: {destino}")
    meses = sorted({m.replace("-", "_") for m in (meses or meses_disponibles())})
    if not meses or any(not re.fullmatch(r"[0-9]{4}_(0[1-9]|1[0-2])", m) for m in meses):
        raise ValueError("Se requieren meses disponibles en formato AAAA_MM.")
    catalogo = (
        Path(catalogo) if catalogo else PROJ_ROOT / "data/raw/estaciones/nomenclatura_2026_09.csv"
    )
    for mes in meses:
        if columnas_mes(mes) != COLUMNAS:
            raise ValueError(f"Contrato de columnas incompatible: {mes}")
    estaciones = cargar_estaciones(catalogo)
    if estaciones["id"].isna().any() or estaciones["id"].duplicated().any():
        raise ValueError("El catálogo requiere IDs únicos y no nulos.")
    coords = estaciones.set_index("id")[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="coerce"
    )
    coord_valida = (
        np.isfinite(coords).all(axis=1)
        & coords["latitude"].between(-90, 90)
        & coords["longitude"].between(-180, 180)
    )
    coords.loc[~coord_valida] = np.nan
    entorno = {
        "python": platform.python_version(),
        **{
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "pyarrow", "loguru", "python-dotenv"]
        },
    }
    fuentes = [
        Path(__file__).resolve(),
        PROJ_ROOT / "proyecto_mlops_equipo_9/dataset.py",
        PROJ_ROOT / "proyecto_mlops_equipo_9/config.py",
        PROJ_ROOT / "proyecto_mlops_equipo_9/__init__.py",
    ]
    hashes_codigo = {p.name: sha256_archivo(p) for p in fuentes}
    hash_catalogo = sha256_archivo(catalogo)
    entradas, vectores = [], []
    for mes in meses:
        ruta = ruta_viajes(mes)
        huella = sha256_archivo(ruta)
        raw = cargar_mes(mes)
        vectores.append(raw["viaje_id"].dropna().drop_duplicates().to_numpy())
        entradas.append(
            {
                "tipo": "viajes",
                "mes": mes,
                "archivo": ruta.relative_to(PROJ_ROOT).as_posix()
                if ruta.is_relative_to(PROJ_ROOT)
                else str(ruta),
                "bytes": ruta.stat().st_size,
                "filas": len(raw),
                "sha256": huella,
            }
        )
        del raw
        print(f"Insumo identificado: {mes}", flush=True)
    todos_ids = np.concatenate(vectores)
    ids, frecuencias = np.unique(todos_ids, return_counts=True)
    ids_entre_archivos = set(ids[frecuencias > 1].tolist())
    del vectores, todos_ids, ids, frecuencias
    entradas.append(
        {
            "tipo": "catalogo",
            "mes": "",
            "archivo": catalogo.name,
            "bytes": catalogo.stat().st_size,
            "filas": len(estaciones),
            "sha256": hash_catalogo,
        }
    )
    identidad = {
        "entradas": [{"archivo": e["archivo"], "sha256": e["sha256"]} for e in entradas],
        "politica": politica,
        "codigo": hashes_codigo,
        "entorno": entorno,
    }
    dataset_id = hashlib.sha256(
        json.dumps(identidad, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    instante = pd.Timestamp.now(tz="UTC")
    lote_id = str(uuid.uuid4())
    contexto = {
        "usuario": usuario,
        "insrt_dt": instante,
        "updt_dt": instante,
        "version_datos": version,
        "version_limpieza": politica["version_metodologia"],
        "dataset_id": dataset_id,
        "lote_id": lote_id,
        "catalogo_version": catalogo.name,
        "catalogo_sha256": hash_catalogo,
        "codigo_limpieza_sha256": hashes_codigo["cleaning.py"],
    }
    control_json = {
        k: v.isoformat() if isinstance(v, pd.Timestamp) else v for k, v in contexto.items()
    }
    salida_base.mkdir(parents=True, exist_ok=True)
    lock = salida_base / f".{version}.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(lote_id)
    trabajo = salida_base / f".{version}.en_proceso_{lote_id}"
    try:
        if destino.exists():
            raise FileExistsError(f"La versión ya existe: {destino}")
        trabajo.mkdir(exist_ok=False)
        for carpeta in ["viajes", "revision", "fuentes/codigo", "metadatos"]:
            (trabajo / carpeta).mkdir(parents=True)
        for fuente in fuentes:
            shutil.copy2(fuente, trabajo / "fuentes/codigo" / fuente.name)
        shutil.copy2(catalogo, trabajo / "fuentes" / catalogo.name)
        notebook = PROJ_ROOT / "notebooks/1.2-integracion-limpieza-final.ipynb"
        notebook_sha = None
        if notebook.exists():
            shutil.copy2(notebook, trabajo / "fuentes/notebook_limpieza.ipynb")
            notebook_sha = sha256_archivo(trabajo / "fuentes/notebook_limpieza.ipynb")
        resumenes, bitacoras, correcciones, indicadores, salidas = [], [], [], [], []
        for mes in meses:
            entrada = next(e for e in entradas if e["mes"] == mes)
            if sha256_archivo(ruta_viajes(mes)) != entrada["sha256"]:
                raise ValueError(f"El insumo cambió entre pasadas: {mes}")
            raw = cargar_mes(mes)
            anotados, validos, bitacora = limpiar_mes(
                raw, mes, coords, politica, ids_entre_archivos
            )
            excluidos = anotados.loc[~anotados["es_valido"]].copy()
            unico = anotados.loc[~anotados["flag_duplicado_exacto"]]
            for bandera in [c for c in anotados if c.startswith("flag_")] + [
                "viaje_corto",
                "corto_distinta_estacion",
                "duracion_larga",
            ]:
                indicadores.append(
                    {
                        "mes": mes,
                        "indicador": bandera,
                        "filas": int(unico[bandera].sum()),
                        "denominador": len(unico),
                        "porcentaje": float(100 * unico[bandera].mean()) if len(unico) else None,
                    }
                )
            for columna, bandera in [
                ("anio_nacimiento", "flag_edad_imposible"),
                ("anio_nacimiento", "flag_nacimiento_no_numerico"),
                ("genero", "flag_genero_invalido"),
            ]:
                correcciones.append(
                    {
                        "mes": mes,
                        "columna": columna,
                        "motivo": bandera,
                        "celdas_en_anotados": int(anotados[bandera].sum()),
                        "celdas_en_validos": int(validos[bandera].sum()),
                        "accion": "convertir_a_faltante; original_preservado",
                    }
                )
            resumenes.append(
                {
                    "mes": mes,
                    "originales": len(raw),
                    "validos": len(validos),
                    "excluidos": len(excluidos),
                    "duplicados_exactos": int(anotados["flag_duplicado_exacto"].sum()),
                    "estaciones_origen_validas": int(validos["origen_id"].nunique()),
                }
            )
            bitacoras.append(bitacora)
            for carpeta, frame in [("viajes", validos), ("revision", excluidos)]:
                controlado = agregar_control(frame, contexto=contexto, entrada=entrada)
                salida = trabajo / carpeta / f"viajes_{mes}.parquet"
                salidas.append(_escribir_parquet_verificado(controlado, salida, trabajo))
            print(
                f"{mes}: {len(raw):,} originales -> {len(validos):,} válidos + {len(excluidos):,} excluidos; Parquet verificados",
                flush=True,
            )
            del raw, anotados, validos, excluidos, unico, controlado, frame
            gc.collect()
        resumen = pd.DataFrame(resumenes)
        bitacora = pd.concat(bitacoras, ignore_index=True)
        totales = {c: int(resumen[c].sum()) for c in ["originales", "validos", "excluidos"]}
        if totales["excluidos"] != int(bitacora["filas_excluidas"].sum()):
            raise ValueError("Balance de bitácora incompatible.")
        for nombre, frame in [
            ("resumen_mensual", resumen),
            ("bitacora", bitacora),
            ("correcciones", pd.DataFrame(correcciones)),
            ("indicadores_calidad", pd.DataFrame(indicadores)),
            ("manifest_entradas", pd.DataFrame(entradas)),
        ]:
            frame.to_csv(trabajo / "metadatos" / f"{nombre}.csv", index=False, encoding="utf-8")
        guardar_json(trabajo / "politica.json", politica)
        guardar_json(trabajo / "metadatos/entorno.json", entorno)
        esquema = pq.read_schema(trabajo / salidas[0]["archivo"])
        guardar_json(
            trabajo / "metadatos/layout.json",
            [
                {"campo": campo.name, "tipo": str(campo.type), "nullable": campo.nullable}
                for campo in esquema
            ],
        )
        (trabajo / "layout.md").write_text(
            _layout_markdown(esquema, CONTROL_DESCRIPCIONES), encoding="utf-8"
        )
        (trabajo / "README.md").write_text(
            f"# Silver {version}\n\nEstado: versión completa después de _SUCCESS.json.\n\n"
            f"- Originales: {totales['originales']:,}\n- Válidos: {totales['validos']:,}\n- Excluidos: {totales['excluidos']:,}\n"
            f"- Usuario de proceso: {usuario}\n- Fecha UTC: {instante.isoformat()}\n- dataset_id: {dataset_id}\n\n"
            "viajes/ conserva las filas válidas y revision/ las excluidas, ambas con indicadores y control.\n"
            "fuentes/ preserva el catálogo, el código ejecutado y el notebook de referencia.\n"
            "metadatos/ contiene conteos, correcciones, indicadores, esquema y manifiestos.\n\n"
            "No sobrescribir esta carpeta. Los originales siguen en data/raw y sus huellas están en el manifiesto.\n"
            "El identificador de contenido excluye usuario, lote y fecha; las huellas de los Parquet sí identifican sus bytes exactos.\n\n"
            "MLflow: pendiente de registro asistido al crear esta versión. No se inició ningún run.\n"
            "La relación con un futuro run se guardará fuera de la carpeta inmutable.\n",
            encoding="utf-8",
        )
        # Las fuentes no deben cambiar mientras se construye esta versión.
        if any(sha256_archivo(p) != hashes_codigo[p.name] for p in fuentes):
            raise ValueError("El código cambió durante la limpieza.")
        if sha256_archivo(catalogo) != hash_catalogo:
            raise ValueError("El catálogo cambió durante la limpieza.")
        for entrada in entradas:
            if (
                entrada["tipo"] == "viajes"
                and sha256_archivo(ruta_viajes(entrada["mes"])) != entrada["sha256"]
            ):
                raise ValueError("Un archivo raw cambió durante la limpieza.")
        # Incluye hashes de todas las salidas estáticas, además de los Parquet.
        for ruta in sorted(trabajo.rglob("*")):
            if ruta.is_file() and ruta.suffix != ".parquet":
                salidas.append(
                    {
                        "archivo": ruta.relative_to(trabajo).as_posix(),
                        "tipo": "metadatos_o_fuente",
                        "bytes": ruta.stat().st_size,
                        "sha256": sha256_archivo(ruta),
                    }
                )
        manifest = {
            "version_datos": version,
            "dataset_id": dataset_id,
            "lote_id": lote_id,
            "estado": "COMPLETA",
            "creado_utc": instante.isoformat(),
            "usuario": usuario,
            "version_limpieza": politica["version_metodologia"],
            "politica": politica,
            "resumen": totales,
            "meses": meses,
            "ids_entre_archivos": len(ids_entre_archivos),
            "entradas": entradas,
            "salidas": salidas,
            "codigo_sha256": hashes_codigo,
            "notebook_referencia_sha256": notebook_sha,
            "entorno": entorno,
            "git_al_ejecutar": _evidencia_git(),
            "campos_control_constantes": control_json,
            "mlflow_al_crear": {"estado": "PENDIENTE_REGISTRO_ASISTIDO", "run_id": None},
        }
        guardar_json(trabajo / "manifest.json", manifest)
        verificacion = verificar_version(trabajo, completa=False)
        guardar_json(trabajo / "verificacion.json", verificacion)
        guardar_json(
            trabajo / "_SUCCESS.json",
            {
                "estado": "COMPLETA",
                "dataset_id": dataset_id,
                "manifest_sha256": sha256_archivo(trabajo / "manifest.json"),
                "verificacion_sha256": sha256_archivo(trabajo / "verificacion.json"),
            },
        )
        trabajo.rename(destino)
        print(
            f"Versión publicada: {destino}\n{json.dumps(totales)}\nMLflow: pendiente; no se inició ningún run.",
            flush=True,
        )
        return destino
    except Exception as error:
        if trabajo.exists():
            guardar_json(trabajo / "_FAILED.json", {"estado": "INCOMPLETA", "error": str(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Raw -> silver versionado; registro MLflow asistido por separado."
    )
    parser.add_argument("--version", default="version_001")
    parser.add_argument(
        "--usuario",
        default=None,
        help="Cuenta responsable de la ejecución; por defecto usuario del sistema.",
    )
    parser.add_argument(
        "--meses", action="append", help="AAAA_MM; repetir para varios meses. Por defecto todos."
    )
    parser.add_argument("--catalogo", type=Path, default=None)
    parser.add_argument(
        "--verificar", type=Path, help="Verifica una carpeta de versión sin generar otra."
    )
    args = parser.parse_args()
    if args.verificar:
        print(json.dumps(verificar_version(args.verificar), ensure_ascii=False, indent=2))
    else:
        generar_version(
            args.version, usuario=args.usuario, meses=args.meses, catalogo=args.catalogo
        )


if __name__ == "__main__":
    main()
