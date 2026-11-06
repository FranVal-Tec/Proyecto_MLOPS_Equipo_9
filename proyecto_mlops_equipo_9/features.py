"""Ingeniería de variables por zona/hora y registro explícito en MLflow.

Uso: python -m proyecto_mlops_equipo_9.features
Importar este módulo no lee datasets, escribe archivos ni conecta con MLflow.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Mantiene el comando de ejecución directa definido en MLproject.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proyecto_mlops_equipo_9.config import CLEAN_DATA_DIR, PROCESSED_DATA_DIR
from proyecto_mlops_equipo_9.dataset import cargar_estaciones
from proyecto_mlops_equipo_9.eda import zonificar

OBJETIVOS = ["salidas_siguiente_hora", "llegadas_siguiente_hora"]
AUDITORIA = [
    "inicio_hora",
    "instante_emision",
    "inicio_hora_objetivo",
    "fin_hora_objetivo",
    "primera_hora_observada_zona",
    "zona_observada",
    "cobertura_horaria_asumida",
    "salidas_sin_registros",
    "llegadas_sin_registros",
    "salidas_cero_asumido",
    "llegadas_cero_asumido",
    "llegadas_borde_inicial",
]
CALENDARIO = [
    "hora_objetivo",
    "dia_semana_objetivo",
    "dia_mes_objetivo",
    "mes_objetivo",
    "dia_anio_objetivo",
    "anio_objetivo",
    "es_fin_semana",
    "es_hora_pico",
    "hora_sin",
    "hora_cos",
    "dia_semana_sin",
    "dia_semana_cos",
]


@dataclass(frozen=True)
class ConfiguracionFeatures:
    """Parámetros explícitos de la metodología aprobada en el notebook."""

    celda_m: int = 1000
    rezagos_h: tuple[int, ...] = (1, 24, 168)
    ventanas_h: tuple[int, ...] = (24, 168)
    min_cobertura_media: float = 0.75
    asumir_cobertura_completa: bool = True
    periodos_sin_cobertura: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        if self.celda_m <= 0 or not 0 < self.min_cobertura_media <= 1:
            raise ValueError("Celda y fracción mínima de cobertura fuera de rango.")
        for valores in (self.rezagos_h, self.ventanas_h):
            if not valores or len(set(valores)) != len(valores):
                raise ValueError("Los rezagos y ventanas deben ser únicos y no vacíos.")
            if any(not isinstance(v, int) or v <= 0 for v in valores):
                raise ValueError("Los rezagos y ventanas deben ser enteros positivos.")
        for inicio, fin in self.periodos_sin_cobertura:
            if pd.Timestamp(inicio) >= pd.Timestamp(fin):
                raise ValueError("Cada periodo sin cobertura debe cumplir inicio < fin.")


@dataclass
class ResultadoFeatures:
    """Datos y rutas producidos por una ejecución; run_id solo si se registró."""

    dataset: pd.DataFrame
    metadata: dict
    output_path: Path
    metadata_path: Path
    run_id: str | None = None
    experiment_id: str | None = None


def exigir(condicion, mensaje):
    """Validación de contrato activa incluso cuando Python se ejecuta con -O."""
    if not condicion:
        raise ValueError(mensaje)


def sha256_archivo(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for bloque in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def preparar_registros(frame, mes):
    """Valida viajes aceptados de una partición mensual sin alterar la entrada."""
    columnas = ["inicio", "fin", "origen_id", "destino_id", "es_valido"]
    exigir(set(columnas).issubset(frame.columns), "Faltan columnas del contrato silver.")
    out = frame[columnas].copy()
    exigir(not out.empty, f"Partición mensual vacía: {mes}.")
    exigir(out.es_valido.eq(True).all(), "Silver contiene viajes no aceptados.")
    for col in ("inicio", "fin"):
        out[col] = pd.to_datetime(out[col], errors="raise")
        exigir(out[col].notna().all(), f"Fechas nulas en {col}.")
        exigir(out[col].dt.tz is None, "Revisar el contrato de zona horaria.")
    exigir(out.inicio.dt.to_period("M").eq(mes).all(), "Inicio fuera del mes declarado.")
    exigir(out.fin.gt(out.inicio).all(), "Hay duraciones no positivas.")
    return out


def asignar_zonas(frame, mapa):
    """Relaciona estaciones y zonas con un catálogo de identificadores únicos."""
    exigir(mapa.index.is_unique, "El catálogo tiene identificadores duplicados.")
    out = frame.copy()
    out["zona_origen"] = out.origen_id.map(mapa)
    out["zona_destino"] = out.destino_id.map(mapa)
    return out


def contar_eventos(frame, fecha, zona):
    return (
        frame.assign(inicio_hora=frame[fecha].dt.floor("h"))
        .groupby([zona, "inicio_hora"])
        .size()
        .rename_axis(index=["zona_id", "inicio_hora"])
    )


def agregar_eventos(frame, inicio_periodo, fin_periodo):
    """Cuenta cada evento por su fecha, incluyendo cruces de mes y medianoche."""
    dentro_inicio = frame.inicio.ge(inicio_periodo) & frame.inicio.lt(fin_periodo)
    dentro_fin = frame.fin.ge(inicio_periodo) & frame.fin.lt(fin_periodo)
    return {
        "salidas": contar_eventos(frame.loc[dentro_inicio], "inicio", "zona_origen"),
        "llegadas": contar_eventos(frame.loc[dentro_fin], "fin", "zona_destino"),
        "filas": len(frame),
        "duracion_max": (frame.fin - frame.inicio).max(),
        "salidas_sin_zona": int(frame.zona_origen.isna().sum()),
        "llegadas_sin_zona": int(frame.zona_destino.isna().sum()),
        "salidas_fuera": int((~dentro_inicio & frame.zona_origen.notna()).sum()),
        "llegadas_fuera": int((~dentro_fin & frame.zona_destino.notna()).sum()),
    }


def crear_pipeline_mes(mes, mapa, inicio_periodo, fin_periodo):
    """Pipeline de preparación, asignación territorial y agregación de un mes."""
    return Pipeline(
        [
            (
                "validar_registros",
                FunctionTransformer(preparar_registros, validate=False, kw_args={"mes": mes}),
            ),
            (
                "asignar_zonas",
                FunctionTransformer(asignar_zonas, validate=False, kw_args={"mapa": mapa}),
            ),
            (
                "agregar_eventos",
                FunctionTransformer(
                    agregar_eventos,
                    validate=False,
                    kw_args={"inicio_periodo": inicio_periodo, "fin_periodo": fin_periodo},
                ),
            ),
        ]
    )


def obtener_agregados(input_path, config):
    """Lee silver mes a mes y valida huellas, cobertura nominal y conservación."""
    input_path = Path(input_path)
    silver = input_path.parent
    manifest_path = silver / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    archivos = sorted(input_path.glob("viajes_????_??.parquet"))
    exigir(bool(archivos), f"No hay Parquet silver en {input_path}.")
    meses = pd.PeriodIndex(
        [p.stem.removeprefix("viajes_").replace("_", "-") for p in archivos], freq="M"
    )
    exigir(meses.is_unique, "Hay meses duplicados.")
    exigir(
        meses.equals(pd.period_range(meses.min(), meses.max(), freq="M")),
        "Faltan meses intermedios.",
    )
    inicio, fin = meses.min().start_time, (meses.max() + 1).start_time
    catalogo_path = silver / "fuentes" / manifest["campos_control_constantes"]["catalogo_version"]
    exigir(
        sha256_archivo(catalogo_path) == manifest["campos_control_constantes"]["catalogo_sha256"],
        "Cambió el catálogo.",
    )
    estaciones = cargar_estaciones(catalogo_path)
    exigir(
        estaciones.id.notna().all() and estaciones.id.is_unique,
        "Identificadores del catálogo inválidos.",
    )
    geografia = zonificar(estaciones, config.celda_m)
    mapa = geografia.set_index("id")["zona"]
    zonas = sorted(mapa.dropna().unique())
    exigir(bool(zonas), "No hay zonas con coordenadas válidas.")
    productos = {p["archivo"]: p for p in manifest["salidas"]}
    partes = []
    columnas = ["inicio", "fin", "origen_id", "destino_id", "es_valido"]
    for archivo, mes in zip(archivos, meses):
        registro = productos[archivo.relative_to(silver).as_posix()]
        exigir(sha256_archivo(archivo) == registro["sha256"], f"Cambió {archivo.name}.")
        exigir(
            set(columnas).issubset(pq.read_schema(archivo).names),
            f"Faltan columnas en {archivo.name}.",
        )
        frame = pd.read_parquet(archivo, columns=columnas)
        exigir(len(frame) == registro["filas"], f"Conteo incorrecto en {archivo.name}.")
        pipeline = crear_pipeline_mes(mes, mapa, inicio, fin)
        partes.append(pipeline.fit_transform(frame))
        del frame
    salidas = pd.concat([p["salidas"] for p in partes]).groupby(level=[0, 1]).sum()
    llegadas = pd.concat([p["llegadas"] for p in partes]).groupby(level=[0, 1]).sum()
    conteos = pd.concat([salidas.rename("salidas"), llegadas.rename("llegadas")], axis=1)
    totales = {
        k: sum(p[k] for p in partes)
        for k in (
            "filas",
            "salidas_sin_zona",
            "llegadas_sin_zona",
            "salidas_fuera",
            "llegadas_fuera",
        )
    }
    for variable in ("salidas", "llegadas"):
        exigir(
            int(conteos[variable].sum())
            + totales[f"{variable}_sin_zona"]
            + totales[f"{variable}_fuera"]
            == totales["filas"],
            f"No se conservaron las {variable}.",
        )
    return {
        "conteos": conteos.sort_index(),
        "zonas": zonas,
        "horas": pd.date_range(inicio, fin, freq="h", inclusive="left"),
        "inicio": inicio,
        "fin": fin,
        "duracion_max": max(p["duracion_max"] for p in partes),
        "manifest_path": manifest_path,
        "catalogo_path": catalogo_path,
        "dataset_id": manifest["dataset_id"],
        "meses": len(meses),
        **totales,
    }


def construir_panel(
    conteos_fuente,
    zonas_catalogo,
    horas_calendario,
    asumir_cobertura=True,
    periodos_sin_cobertura=(),
):
    """Calendario continuo; ceros solo en zonas observadas y horas cubiertas."""
    exigir(not conteos_fuente.index.has_duplicates, "Conteos zona/hora duplicados.")
    indice = pd.MultiIndex.from_product(
        [zonas_catalogo, horas_calendario], names=["zona_id", "inicio_hora"]
    )
    out = conteos_fuente.reindex(indice).reset_index()
    primera = (
        conteos_fuente.dropna(how="all").reset_index().groupby("zona_id")["inicio_hora"].min()
    )
    out["primera_hora_observada_zona"] = out.zona_id.map(primera)
    out["zona_observada"] = out.inicio_hora.ge(out.primera_hora_observada_zona)
    actividad_red = conteos_fuente.groupby(level="inicio_hora").sum(min_count=1).sum(axis=1)
    out["cobertura_horaria_asumida"] = out.inicio_hora.map(actividad_red).fillna(0).gt(0) & bool(
        asumir_cobertura
    )
    for inicio, fin in periodos_sin_cobertura:
        exigir(pd.Timestamp(inicio) < pd.Timestamp(fin), "Periodo de cobertura inválido.")
        fuera = out.inicio_hora.ge(pd.Timestamp(inicio)) & out.inicio_hora.lt(pd.Timestamp(fin))
        out.loc[fuera, "cobertura_horaria_asumida"] = False
    elegible = out.zona_observada & out.cobertura_horaria_asumida
    for variable in ("salidas", "llegadas"):
        out[f"{variable}_sin_registros"] = out[variable].isna()
        out[f"{variable}_cero_asumido"] = out[variable].isna() & elegible
        if asumir_cobertura:
            out.loc[elegible, variable] = out.loc[elegible, variable].fillna(0)
            out.loc[~elegible, variable] = np.nan
        out[variable] = out[variable].astype("Float64")
    return out


def preparar_horizonte(frame, inicio_periodo, duracion_max):
    """Marca llegadas incompletas al inicio y define fechas de emisión y objetivo."""
    out = frame.copy()
    fin_margen = (inicio_periodo + duracion_max).ceil("h")
    out["llegadas_borde_inicial"] = out.inicio_hora.lt(fin_margen)
    out.loc[out.llegadas_borde_inicial, "llegadas"] = pd.NA
    out.loc[out.llegadas_borde_inicial, "llegadas_cero_asumido"] = False
    out["instante_emision"] = out.inicio_hora + pd.Timedelta(hours=1)
    out["inicio_hora_objetivo"] = out.instante_emision
    out["fin_hora_objetivo"] = out.instante_emision + pd.Timedelta(hours=1)
    return out


def validar_calendario(frame):
    exigir(not frame.duplicated(["zona_id", "inicio_hora"]).any(), "Claves duplicadas.")
    exigir(
        frame.groupby("zona_id").inicio_hora.diff().dropna().eq(pd.Timedelta(hours=1)).all(),
        "El calendario no es horario continuo.",
    )


def agregar_historia(
    frame, rezagos_h=(1, 24, 168), ventanas_h=(24, 168), min_cobertura_media=0.75
):
    """Historia anterior por zona, con cobertura mínima y calentamiento completo."""
    out = frame.sort_values(["zona_id", "inicio_hora"]).copy()
    validar_calendario(out)
    for variable in ("salidas", "llegadas"):
        grupo = out.groupby("zona_id")[variable]
        for lag in rezagos_h:
            out[f"{variable}_lag_{lag}h"] = grupo.shift(lag)
        for ventana in ventanas_h:
            out[f"{variable}_media_previa_{ventana}h"] = grupo.transform(
                lambda s, ventana=ventana: (
                    s.shift(1)
                    .rolling(ventana, min_periods=int(np.ceil(min_cobertura_media * ventana)))
                    .mean()
                )
            )
            out[f"{variable}_cobertura_previa_{ventana}h"] = grupo.transform(
                lambda s, ventana=ventana: (
                    s.shift(1).rolling(ventana, min_periods=1).count() / ventana
                )
            )
            if "primera_hora_observada_zona" in out:
                completa = (out.inicio_hora - out.primera_hora_observada_zona).ge(
                    pd.Timedelta(hours=ventana)
                )
            else:
                completa = out.groupby("zona_id").cumcount().ge(ventana)
            out.loc[~completa, f"{variable}_media_previa_{ventana}h"] = np.nan
    return out


def agregar_caracteristicas_temporales(frame):
    """Calendario conocido de la hora objetivo y representaciones cíclicas."""
    out = frame.copy()
    fecha = out.inicio_hora_objetivo.dt
    for nombre, valores in (
        ("hora", fecha.hour),
        ("dia_semana", fecha.day_of_week),
        ("dia_mes", fecha.day),
        ("mes", fecha.month),
        ("dia_anio", fecha.day_of_year),
        ("anio", fecha.year),
    ):
        out[f"{nombre}_objetivo"] = valores
    out["es_fin_semana"] = fecha.day_of_week.ge(5).astype("int8")
    out["es_hora_pico"] = (fecha.hour.between(7, 9) | fecha.hour.between(16, 19)).astype("int8")
    out["hora_sin"] = np.sin(2 * np.pi * fecha.hour / 24)
    out["hora_cos"] = np.cos(2 * np.pi * fecha.hour / 24)
    out["dia_semana_sin"] = np.sin(2 * np.pi * fecha.day_of_week / 7)
    out["dia_semana_cos"] = np.cos(2 * np.pi * fecha.day_of_week / 7)
    return out


def crear_pipeline_features(zonas, horas, inicio_periodo, duracion_max, config=None):
    """Pipeline exclusivamente de variables; los objetivos se agregan después."""
    config = config if config is not None else ConfiguracionFeatures()
    return Pipeline(
        [
            (
                "calendario_cobertura",
                FunctionTransformer(
                    construir_panel,
                    validate=False,
                    kw_args={
                        "zonas_catalogo": zonas,
                        "horas_calendario": horas,
                        "asumir_cobertura": config.asumir_cobertura_completa,
                        "periodos_sin_cobertura": config.periodos_sin_cobertura,
                    },
                ),
            ),
            (
                "horizonte",
                FunctionTransformer(
                    preparar_horizonte,
                    validate=False,
                    kw_args={"inicio_periodo": inicio_periodo, "duracion_max": duracion_max},
                ),
            ),
            (
                "historia",
                FunctionTransformer(
                    agregar_historia,
                    validate=False,
                    kw_args={
                        "rezagos_h": config.rezagos_h,
                        "ventanas_h": config.ventanas_h,
                        "min_cobertura_media": config.min_cobertura_media,
                    },
                ),
            ),
            (
                "calendario_objetivo",
                FunctionTransformer(agregar_caracteristicas_temporales, validate=False),
            ),
        ]
    )


def agregar_variables_objetivo(frame):
    """Construcción retrospectiva de etiquetas; no forma parte de los predictores."""
    out = frame.sort_values(["zona_id", "inicio_hora"]).copy()
    validar_calendario(out)
    for variable in ("salidas", "llegadas"):
        out[f"{variable}_siguiente_hora"] = out.groupby("zona_id")[variable].shift(-1)
    return out


def seleccionar_dataset(panel, fin_periodo):
    """Selecciona filas completas, conserva ceros y devuelve su contrato y balance."""
    historia = ["salidas", "llegadas"] + [
        c
        for c in panel
        if any(patron in c for patron in ("_lag_", "_media_previa_", "_cobertura_previa_"))
    ]
    features = ["zona_id"] + historia + CALENDARIO
    exigir(not set(features) & set(OBJETIVOS), "Los objetivos aparecen en predictores.")
    motivo = np.select(
        [
            ~panel[OBJETIVOS].notna().all(axis=1),
            ~panel[features].notna().all(axis=1),
            ~panel.fin_hora_objetivo.le(fin_periodo),
        ],
        ["objetivo_desconocido", "historia_incompleta", "fuera_del_periodo"],
        default="incluida",
    )
    balance = pd.Series(motivo).value_counts()
    dataset = (
        panel.loc[motivo == "incluida", AUDITORIA + features + OBJETIVOS]
        .sort_values(["inicio_hora_objetivo", "zona_id"])
        .reset_index(drop=True)
    )
    exigir(not dataset.empty, "No hay filas supervisadas con la política indicada.")
    exigir(dataset[features + OBJETIVOS].notna().all().all(), "Hay faltantes en el resultado.")
    exigir(
        not dataset.duplicated(["zona_id", "inicio_hora_objetivo"]).any(),
        "Hay claves duplicadas en el resultado.",
    )
    exigir(
        np.isfinite(dataset[historia + CALENDARIO + OBJETIVOS].to_numpy(dtype=float)).all(),
        "Hay valores no finitos.",
    )
    for variable in ["salidas", "llegadas", *OBJETIVOS]:
        exigir(
            dataset[variable].ge(0).all() and dataset[variable].mod(1).eq(0).all(),
            f"Conteos inválidos en {variable}.",
        )
        dataset[variable] = dataset[variable].astype("int32")
    dataset["zona_id"] = dataset.zona_id.astype("string")
    return dataset, features, {str(k): int(v) for k, v in balance.items()}


def exportar_resultados(dataset, features, balance, contexto, config, output_path):
    """Guarda Parquet y contrato JSON y verifica su lectura completa."""
    output_path = Path(output_path).resolve()
    fuentes = [contexto["manifest_path"], contexto["catalogo_path"]]
    exigir(output_path not in [p.resolve() for p in fuentes], "La salida pisa una fuente.")
    exigir(
        not output_path.is_relative_to(contexto["manifest_path"].parent.resolve()),
        "La salida debe estar fuera de la versión silver.",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False, compression="snappy")
    pd.testing.assert_frame_equal(dataset, pd.read_parquet(output_path))
    metadata = {
        "version": "fc_zona_hora_v2",
        "archivo": output_path.name,
        "filas": len(dataset),
        "columnas": len(dataset.columns),
        "features": features,
        "targets": OBJETIVOS,
        "audit_columns": AUDITORIA,
        "dtypes": {c: str(t) for c, t in dataset.dtypes.items()},
        "zona_celda_m": config.celda_m,
        "frecuencia": "h",
        "horizonte_horas": 1,
        "rezagos_horas": list(config.rezagos_h),
        "ventanas_horas": list(config.ventanas_h),
        "min_fraccion_observada_medias": config.min_cobertura_media,
        "convencion_temporal": "Fila t resume [t,t+1h); emisión t+1h; objetivo [t+1h,t+2h).",
        "cobertura_completa_asumida": config.asumir_cobertura_completa,
        "regla_ceros": "Desde primera actividad zonal; solo horas con actividad global y fuera de fallas conocidas.",
        "periodos_sin_cobertura": [[str(a), str(b)] for a, b in config.periodos_sin_cobertura],
        "disponibilidad": "Dataset offline retrospectivo; validar latencia de aceptación silver antes de producción.",
        "fuentes": {
            "manifest_silver_sha256": sha256_archivo(contexto["manifest_path"]),
            "catalogo_sha256": sha256_archivo(contexto["catalogo_path"]),
            "dataset_id_silver": contexto["dataset_id"],
            "desde": str(contexto["inicio"]),
            "hasta_exclusivo": str(contexto["fin"]),
        },
        "balance_filas": balance,
        "objetivos_cero": {c: int(dataset[c].eq(0).sum()) for c in OBJETIVOS},
        "parquet_sha256": sha256_archivo(output_path),
        "siguiente_etapa": "Particiones cronológicas globales; X=features, y=targets. Excluir auditoría de X.",
        "codigo_features_sha256": sha256_archivo(__file__),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return ResultadoFeatures(dataset, metadata, output_path, metadata_path)


def registrar_resultados_mlflow(
    resultado, contexto, config, tracking_uri, experiment_name, run_name
):
    """Registra solo bajo invocación explícita contra un servidor HTTP existente."""
    from urllib.parse import urlsplit

    import mlflow
    from mlflow.tracking import MlflowClient

    parsed = urlsplit(tracking_uri)
    exigir(
        parsed.scheme in {"http", "https"} and bool(parsed.netloc),
        "Se requiere un servidor HTTP de MLflow; no una base o carpeta local.",
    )
    exigir(mlflow.active_run() is None, "Finaliza la ejecución activa antes de iniciar otra.")
    client = MlflowClient(tracking_uri=tracking_uri)
    # Verifica conectividad antes de crear el experimento.
    client.search_experiments(max_results=1)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    metadata = resultado.metadata
    dataset = resultado.dataset
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "etapa": "feature_engineering",
                "unidad": "zona_hora",
                "version_metodologia": metadata["version"],
                "disponibilidad": "retrospectiva_offline",
            }
        )
        mlflow.log_params(
            {
                **asdict(config),
                "frecuencia": "h",
                "horizonte_horas": 1,
                "dataset_id_silver": contexto["dataset_id"],
                "dataset_path": str(resultado.output_path),
                "dataset_sha256": metadata["parquet_sha256"],
                "codigo_features_sha256": metadata["codigo_features_sha256"],
            }
        )
        mlflow.log_metrics(
            {
                "filas_finales": len(dataset),
                "filas_panel": sum(metadata["balance_filas"].values()),
                "viajes_silver": contexto["filas"],
                "zonas_finales": dataset.zona_id.nunique(),
                "predictores": len(metadata["features"]),
                "objetivos": len(OBJETIVOS),
                "columnas_auditoria": len(AUDITORIA),
                "missing_values_pct": float(dataset.isna().sum().sum() / dataset.size * 100),
                "duplicate_rows": int(dataset.duplicated().sum()),
                **{f"filas_{k}": v for k, v in metadata["balance_filas"].items()},
                **{f"ceros_{k}": v for k, v in metadata["objetivos_cero"].items()},
            }
        )
        # El Parquet permanece en disco. MLflow conserva su perfil, origen y digest.
        input_dataset = mlflow.data.from_pandas(
            dataset, source=resultado.output_path.as_uri(), name="mibici_zona_hora"
        )
        mlflow.log_input(input_dataset, context="feature_engineering")
        mlflow.log_artifact(str(resultado.metadata_path), artifact_path="contrato")
        mlflow.log_artifact(str(Path(__file__).resolve()), artifact_path="codigo")
        mlflow.log_dict(
            {"features": metadata["features"], "targets": OBJETIVOS, "audit_columns": AUDITORIA},
            "columnas.json",
        )
        resultado.run_id = run.info.run_id
        resultado.experiment_id = run.info.experiment_id
    exigir(
        client.get_run(resultado.run_id).info.status == "FINISHED",
        "La ejecución de MLflow no finalizó correctamente.",
    )
    return resultado


def ejecutar_features(
    input_path=CLEAN_DATA_DIR,
    output_path=None,
    config=None,
    registrar_mlflow=True,
    tracking_uri="http://127.0.0.1:5000",
    experiment_name="mibici-feature-engineering",
    run_name="zona_hora_v1",
):
    """Ejecuta featuring, exporta y opcionalmente registra en el servidor existente.

    No entrena modelos. Los objetivos se construyen fuera del pipeline de variables.
    Cada llamada con registrar_mlflow=True crea una nueva ejecución del experimento.
    """
    config = config if config is not None else ConfiguracionFeatures()
    exigir(
        config.asumir_cobertura_completa,
        "No exportar seleccionando solo horas positivas del modo conservador.",
    )
    contexto = obtener_agregados(input_path, config)
    pipeline = crear_pipeline_features(
        contexto["zonas"], contexto["horas"], contexto["inicio"], contexto["duracion_max"], config
    )
    panel = pipeline.fit_transform(contexto["conteos"])
    panel = agregar_variables_objetivo(panel)
    dataset, features, balance = seleccionar_dataset(panel, contexto["fin"])
    resultado = exportar_resultados(
        dataset,
        features,
        balance,
        contexto,
        config,
        output_path or PROCESSED_DATA_DIR / "datos_feature_engineered_zona_hora.parquet",
    )
    if registrar_mlflow:
        resultado = registrar_resultados_mlflow(
            resultado, contexto, config, tracking_uri, experiment_name, run_name
        )
    return resultado


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=CLEAN_DATA_DIR)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--experiment-name", default="mibici-feature-engineering")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--run-name", default="zona_hora_v1")
    parser.add_argument("--sin-mlflow", action="store_true")
    args = parser.parse_args()
    resultado = ejecutar_features(
        input_path=args.input_path,
        output_path=args.output_path,
        registrar_mlflow=not args.sin_mlflow,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    print(f"Dataset: {resultado.output_path} | filas: {len(resultado.dataset):,}")
    if resultado.run_id:
        print(f"MLflow run_id: {resultado.run_id}")


if __name__ == "__main__":
    main()
