"""Evalúa un mes de MiBici y, opcionalmente, registra la evidencia en MLflow."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from urllib.parse import urlsplit

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from proyecto_mlops_equipo_9.features import sha256_archivo
from proyecto_mlops_equipo_9.modeling.predict import predecir
from proyecto_mlops_equipo_9.modeling.validation import (
    OBJETIVOS,
    CalendarioValidacion,
    metricas,
    prediccion_base,
)

CLAVES = ["inicio_hora_objetivo", "zona_id"]
ESTRATOS = ["zona_id", "hora_objetivo", "dia_semana_objetivo"]
PRONOSTICOS = ["salidas_pronosticadas", "llegadas_pronosticadas"]
SENALES = ["salidas", "llegadas", "salidas_lag_23h", "llegadas_lag_23h"]
PRUEBA = "referencia_abril_mayo"
ESCENARIO_REAL = "mes_real"
REGISTRO = "mibici-demanda-zona-hora"
EXPERIMENTO = "mibici-monitoreo"


@dataclass(frozen=True)
class Umbrales:
    """Umbrales piloto de revisión, expresados como variación relativa."""

    entrada: float = 0.25
    mae: float = 0.20
    mejora_minima_naive: float = 0.10


def seleccionar_ventana(datos, desde, hasta=None):
    inicio = pd.Period(desde, freq="M").start_time
    fin = (pd.Period(hasta or desde, freq="M") + 1).start_time
    filas = datos.loc[datos.inicio_hora_objetivo.ge(inicio) & datos.inicio_hora_objetivo.lt(fin)]
    if filas.empty:
        raise ValueError(f"No hay etiquetas en la ventana {desde} a {hasta or desde}.")
    if filas.inicio_hora_objetivo.dt.to_period("M").nunique() != len(
        pd.period_range(desde, hasta or desde, freq="M")
    ):
        raise ValueError("La ventana solicitada tiene meses sin observaciones evaluables.")
    return filas.reset_index(drop=True)


def simular_sobreconteo(referencia, factor):
    """Multiplica la telemetría de salidas manteniendo las etiquetas originales."""
    columnas = [
        c
        for c in referencia
        if c == "salidas" or c.startswith(("salidas_lag_", "salidas_media_previa_"))
    ]
    alterado = referencia.copy()
    alterado[columnas] = alterado[columnas].astype(float) * factor
    pd.testing.assert_frame_equal(referencia[CLAVES + OBJETIVOS], alterado[CLAVES + OBJETIVOS])
    return alterado, columnas


def evaluar_modelos(modelo, filas, escenario, zonas_referencia):
    pred = predecir(modelo, filas).set_index(CLAVES)
    actual = pred.reindex(pd.MultiIndex.from_frame(filas[CLAVES]))[PRONOSTICOS].to_numpy()
    ingenuo = prediccion_base(filas)
    y = filas[OBJETIVOS].to_numpy(dtype=float)
    comparable = filas.zona_id.isin(zonas_referencia).to_numpy()
    segmentos = {
        "total": np.ones(len(filas), dtype=bool),
        "zonas_referencia": comparable,
        "otras_zonas": ~comparable,
    }
    filas_metricas = []
    for segmento, mascara in segmentos.items():
        if not mascara.any():
            continue
        validas = mascara & np.isfinite(actual).all(axis=1) & np.isfinite(ingenuo).all(axis=1)
        if not validas.any():
            continue
        for nombre, predicho in (
            ("modelo_fijo", actual),
            ("naive_estacional", ingenuo),
        ):
            filas_metricas.append(
                {
                    "escenario": escenario,
                    "segmento": segmento,
                    "modelo": nombre,
                    "filas_ventana": int(mascara.sum()),
                    "filas_pareadas": int(validas.sum()),
                    "zonas": int(filas.loc[validas, "zona_id"].nunique()),
                    "cobertura_modelo": float(np.isfinite(predicho[mascara]).all(axis=1).mean()),
                    **metricas(y[validas], predicho[validas]),
                }
            )
    return filas_metricas


def medir_drift(referencia, ventana, escenario):
    """Compara medias dando el mismo peso a cada estrato común de zona y calendario."""
    base = referencia.groupby(ESTRATOS, observed=True)[SENALES].mean()
    actual = ventana.groupby(ESTRATOS, observed=True)[SENALES].mean()
    comunes = base.index.intersection(actual.index)
    if comunes.empty:
        raise ValueError(f"Sin estratos comparables para medir drift en {escenario}.")
    media_base = base.loc[comunes].mean()
    media_actual = actual.loc[comunes].mean()
    return [
        {
            "escenario": escenario,
            "variable": variable,
            "estratos_comunes": len(comunes),
            "zonas_comunes": comunes.get_level_values("zona_id").nunique(),
            "media_referencia": float(media_base[variable]),
            "media_actual": float(media_actual[variable]),
            "cambio_relativo": cociente(media_actual[variable], media_base[variable]) - 1,
        }
        for variable in SENALES
    ]


def cociente(a, b):
    return float(a / b) if b else float("nan")


def generar_alertas(tabla, drift, umbrales):
    comparables = tabla[tabla.segmento.eq("zonas_referencia")]
    resumen = comparables.pivot(index="escenario", columns="modelo", values="mae")
    base = resumen.loc[PRUEBA, "modelo_fijo"]
    alertas = []
    for escenario, fila in resumen.drop(index=PRUEBA).iterrows():
        cambio_mae = cociente(fila.modelo_fijo, base) - 1
        mejora = 1 - cociente(fila.modelo_fijo, fila.naive_estacional)
        senales = drift[drift.escenario.eq(escenario)].set_index("variable").cambio_relativo
        for variable, cambio in senales.items():
            alertas.append(
                (
                    escenario,
                    f"entrada_{variable}",
                    abs(cambio),
                    umbrales.entrada,
                    abs(cambio) >= umbrales.entrada,
                    "Revisar fuente, cobertura y conteos antes de ajustar el modelo.",
                )
            )
        alertas.extend(
            [
                (
                    escenario,
                    "aumento_mae",
                    cambio_mae,
                    umbrales.mae,
                    cambio_mae >= umbrales.mae,
                    "Revisar error por zona y hora. Si persiste con datos válidos, evaluar reentrenamiento.",
                ),
                (
                    escenario,
                    "ventaja_naive",
                    mejora,
                    umbrales.mejora_minima_naive,
                    mejora < umbrales.mejora_minima_naive,
                    "Comparar el modelo con el naïve y evaluar un candidato con datos recientes.",
                ),
            ]
        )
    tabla_alertas = pd.DataFrame(
        alertas,
        columns=[
            "escenario",
            "regla",
            "valor",
            "umbral",
            "activada",
            "accion_propuesta",
        ],
    )
    tabla_alertas["evaluable"] = np.isfinite(tabla_alertas.valor)
    return tabla_alertas


def graficar(tabla, drift, alertas, mes, ruta):
    nombres = {
        PRUEBA: "Abr–may\nreferencia",
        ESCENARIO_REAL: f"{mes}\nreal",
        "sintetico_20": "Sintético\n+20%",
        "sintetico_40": "Sintético\n+40%",
    }
    orden = list(nombres)
    comparables = tabla[tabla.segmento.eq("zonas_referencia")]
    error = comparables.pivot(index="escenario", columns="modelo", values="mae").reindex(orden)
    cambios = drift.pivot(index="escenario", columns="variable", values="cambio_relativo")
    cambios = cambios.reindex(orden[1:])[["salidas", "llegadas"]] * 100
    figura = Figure(figsize=(11, 4.5), layout="constrained")
    FigureCanvasAgg(figura)
    izq, der = figura.subplots(1, 2)
    x = np.arange(len(orden))
    izq.bar(x - 0.18, error.modelo_fijo, 0.36, color="#08665d", label="Modelo fijo")
    izq.bar(
        x + 0.18,
        error.naive_estacional,
        0.36,
        color="#8495a5",
        label="Naïve estacional",
    )
    umbral_mae = alertas.loc[alertas.regla.eq("aumento_mae"), "umbral"].iloc[0]
    izq.axhline(
        error.iloc[0].modelo_fijo * (1 + umbral_mae),
        color="#b74624",
        ls="--",
        label="Umbral del modelo fijo",
    )
    izq.set(
        xticks=x,
        xticklabels=[nombres[n] for n in orden],
        ylabel="MAE (viajes por zona-hora)",
        title="Desempeño en zonas de referencia",
    )
    z = np.arange(len(cambios))
    der.bar(z - 0.18, cambios.salidas, 0.36, color="#08665d", label="Salidas")
    der.bar(z + 0.18, cambios.llegadas, 0.36, color="#8495a5", label="Llegadas")
    umbral = alertas.loc[alertas.regla.eq("entrada_salidas"), "umbral"].iloc[0] * 100
    der.axhline(umbral, color="#b74624", ls="--", label="Umbral de entrada")
    if cambios.min().min() < 0:
        der.axhline(-umbral, color="#b74624", ls="--")
    der.set(
        xticks=z,
        xticklabels=[nombres[n] for n in orden[1:]],
        ylabel="Cambio frente a referencia (%)",
        title="Entradas en estratos comparables",
    )
    for eje in (izq, der):
        eje.legend(frameon=False, fontsize=8)
        eje.grid(axis="y", alpha=0.23)
        eje.set_axisbelow(True)
    figura.savefig(ruta, dpi=160)


def escribir_json(ruta, datos):
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ejecutar_monitoreo(features, mes, modelo, identidad, salida, fuente=None, umbrales=None):
    """Evalúa referencia, mes real y estrés sintético con un modelo congelado."""
    features = Path(features)
    salida = Path(salida)
    umbrales = umbrales or Umbrales()
    calendario = CalendarioValidacion()
    if pd.Period(mes, freq="M") <= pd.Period(calendario.prueba_hasta, freq="M"):
        raise ValueError("El mes de monitoreo debe ser posterior al cierre de la referencia.")
    datos = pd.read_parquet(features)
    referencia = seleccionar_ventana(datos, calendario.prueba_desde, calendario.prueba_hasta)
    real = seleccionar_ventana(datos, mes)
    for filas in (referencia, real):
        if filas[CLAVES].duplicated().any() or not np.isfinite(filas[OBJETIVOS]).all().all():
            raise ValueError("La ventana contiene claves duplicadas o etiquetas no válidas.")
    zonas = set(referencia.zona_id) & set(real.zona_id)
    if not zonas:
        raise ValueError("El mes real y la referencia no tienen zonas comparables.")
    ventanas = {PRUEBA: referencia, ESCENARIO_REAL: real}
    simulaciones = {}
    for porcentaje in (20, 40):
        escenario = f"sintetico_{porcentaje}"
        ventanas[escenario], columnas = simular_sobreconteo(referencia, 1 + porcentaje / 100)
        simulaciones[escenario] = {
            "sobreconteo_pct": porcentaje,
            "columnas": columnas,
            "etiquetas_originales": True,
        }
    tabla = pd.DataFrame(
        [
            fila
            for nombre, datos_mes in ventanas.items()
            for fila in evaluar_modelos(modelo, datos_mes, nombre, zonas)
        ]
    )
    drift = pd.DataFrame(
        [
            fila
            for nombre, datos_mes in ventanas.items()
            if nombre != PRUEBA
            for fila in medir_drift(referencia, datos_mes, nombre)
        ]
    )
    alertas = generar_alertas(tabla, drift, umbrales)
    huella_features = sha256_archivo(features)
    if fuente and fuente.get("features_sha256") != huella_features:
        raise ValueError("La huella del dataset no coincide con la evidencia de procedencia.")
    evidencia = {
        "mes": mes,
        "modelo": identidad,
        "features_archivo": features.name,
        "features_sha256": huella_features,
        "umbral_piloto": True,
        "referencia": [calendario.prueba_desde, calendario.prueba_hasta],
        "mes_real": {
            "filas": len(real),
            "zonas": int(real.zona_id.nunique()),
            "zonas_comunes": len(zonas),
            "zonas_fuera_de_referencia": sorted(set(real.zona_id) - zonas),
            "horas_objetivo_evaluables": int(real.inicio_hora_objetivo.nunique()),
            "horas_calendario": int(pd.Period(mes, freq="M").days_in_month * 24),
            "primera_hora_objetivo": real.inicio_hora_objetivo.min().isoformat(),
            "ultima_hora_objetivo": real.inicio_hora_objetivo.max().isoformat(),
        },
        "fuente": fuente or {},
        "simulaciones": simulaciones,
        "umbrales": asdict(umbrales),
    }
    salida.mkdir(parents=True, exist_ok=True)
    for nombre, frame in [("metricas", tabla), ("drift", drift), ("alertas", alertas)]:
        frame.to_csv(salida / f"{nombre}.csv", index=False)
    escribir_json(salida / "evidencia.json", evidencia)
    graficar(tabla, drift, alertas, mes, salida / "monitoreo.png")
    return tabla, drift, alertas, evidencia


def cargar_modelo_local(ruta):
    import skops.io as sio

    ruta = Path(ruta)
    internos = sio.get_untrusted_types(file=ruta)
    aprobados = ("sklearn.", "numpy.", "scipy.")
    if any(tipo != "functools.partial" and not tipo.startswith(aprobados) for tipo in internos):
        raise ValueError(f"El modelo contiene tipos que requieren revisión: {internos}")
    return sio.load(ruta, trusted=internos)


def verificar_tracking(tracking_uri):
    """Exige que el servidor o archivo elegido ya contenga el registro del equipo."""
    if tracking_uri.startswith("sqlite:///"):
        archivo = Path(tracking_uri.removeprefix("sqlite:///")).resolve(strict=True)
        if archivo.name != "mlflow.db" or not archivo.is_file():
            raise ValueError("Selecciona el archivo mlflow.db existente del equipo.")
        import sqlite3

        with sqlite3.connect(archivo.as_uri() + "?mode=ro", uri=True) as db:
            if not db.execute(
                "select 1 from model_versions where name=? and version=2", (REGISTRO,)
            ).fetchone():
                raise ValueError("La base seleccionada no contiene la versión 2 de MiBici.")
    elif urlsplit(tracking_uri).scheme not in ("http", "https"):
        raise ValueError("Indica el servidor del equipo o su archivo SQLite existente.")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    cliente = mlflow.MlflowClient()
    if cliente.get_experiment_by_name("mibici-entrenamiento") is None:
        raise ValueError("El destino no contiene el experimento histórico del equipo.")
    cliente.get_model_version(REGISTRO, "2")
    return cliente


def registrar_evaluacion(tracking_uri, salida, resultados, artifact_location=None):
    import mlflow

    cliente = verificar_tracking(tracking_uri)
    experimento = cliente.get_experiment_by_name(EXPERIMENTO)
    if not experimento and tracking_uri.startswith("sqlite:///") and not artifact_location:
        raise ValueError("Indica --artifact-location para guardar los artefactos con SQLite.")
    identificador = (
        experimento.experiment_id
        if experimento
        else cliente.create_experiment(EXPERIMENTO, artifact_location=artifact_location)
    )
    tabla, drift, alertas, evidencia = resultados
    mes = evidencia["mes"]
    with mlflow.start_run(experiment_id=identificador, run_name=f"monitoreo_{mes}") as corrida:
        mlflow.log_params(
            {
                "mes": mes,
                "modelo": evidencia["modelo"]["id"],
                "features_sha256": evidencia["features_sha256"],
                **evidencia["umbrales"],
            }
        )
        mlflow.set_tags(
            {
                "origen_modelo": evidencia["modelo"]["origen"],
                "tipo": "monitoreo_mensual",
            }
        )
        if evidencia["fuente"]:
            mlflow.log_dict(evidencia["fuente"], "procedencia_datos.json")
            meses_fuente = evidencia["fuente"].get("fuentes_mensuales", [])
            fuente_actual = next((f for f in meses_fuente if f["periodo"] == mes), None)
            if fuente_actual:
                mlflow.log_params(
                    {
                        "csv_sha256": fuente_actual["sha256"],
                        "csv_url": fuente_actual.get("url") or "sin URL registrada",
                    }
                )
                mlflow.log_metrics(
                    {
                        key: fuente_actual[key]
                        for key in (
                            "filas_crudas",
                            "filas_aceptadas",
                            "filas_excluidas",
                        )
                    }
                )
        mlflow.log_metrics(
            {
                key: evidencia["mes_real"][key]
                for key in (
                    "zonas_comunes",
                    "horas_objetivo_evaluables",
                    "horas_calendario",
                )
            }
        )
        for nombre, grupo in tabla.groupby("escenario", sort=False):
            with mlflow.start_run(run_name=nombre, nested=True):
                mlflow.set_tag("escenario", nombre)
                if nombre in evidencia["simulaciones"]:
                    mlflow.log_param(
                        "sobreconteo_pct",
                        evidencia["simulaciones"][nombre]["sobreconteo_pct"],
                    )
                    mlflow.set_tag("etiquetas_originales", "true")
                for fila in grupo.itertuples(index=False):
                    prefijo = f"{fila.segmento}_{fila.modelo}"
                    valores = {
                        c: getattr(fila, c)
                        for c in (
                            "mae",
                            "mae_salidas",
                            "mae_llegadas",
                            "mae_flujo_neto",
                            "wape",
                            "filas_pareadas",
                            "filas_ventana",
                            "zonas",
                            "cobertura_modelo",
                        )
                    }
                    mlflow.log_metrics({f"{prefijo}_{k}": v for k, v in valores.items()})
                comparables = grupo[grupo.segmento.eq("zonas_referencia")].set_index("modelo")
                ventaja = 1 - cociente(
                    comparables.loc["modelo_fijo", "mae"],
                    comparables.loc["naive_estacional", "mae"],
                )
                if np.isfinite(ventaja):
                    mlflow.log_metric("ventaja_frente_naive", ventaja)
                for fila in drift[drift.escenario.eq(nombre)].itertuples(index=False):
                    if np.isfinite(fila.cambio_relativo):
                        mlflow.log_metric(f"cambio_relativo_{fila.variable}", fila.cambio_relativo)
                for fila in alertas[alertas.escenario.eq(nombre)].itertuples(index=False):
                    if fila.evaluable:
                        mlflow.log_metric(f"alerta_{fila.regla}", int(fila.activada))
        run_id = corrida.info.run_id
        evidencia["mlflow"] = {"experimento": EXPERIMENTO, "run_id": run_id}
        escribir_json(Path(salida) / "evidencia.json", evidencia)
        for archivo in (
            "metricas.csv",
            "drift.csv",
            "alertas.csv",
            "evidencia.json",
            "monitoreo.png",
        ):
            mlflow.log_artifact(str(Path(salida) / archivo), artifact_path="reporte")
    return run_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--mes", required=True)
    modelos = parser.add_mutually_exclusive_group(required=True)
    modelos.add_argument("--modelo-uri")
    modelos.add_argument("--modelo-local", type=Path)
    parser.add_argument(
        "--identidad-modelo", type=Path, help="JSON con id y origen del modelo local"
    )
    parser.add_argument("--fuente", type=Path, help="JSON de procedencia y validación del mes")
    parser.add_argument("--salida-dir", type=Path, default=Path("reports/drift/monthly"))
    parser.add_argument("--tracking-uri")
    parser.add_argument("--registrar-mlflow", action="store_true")
    parser.add_argument("--artifact-location", help="Ruta de artefactos para el experimento nuevo")
    args = parser.parse_args()
    if (args.registrar_mlflow or args.modelo_uri) and not args.tracking_uri:
        parser.error("Indica --tracking-uri para usar la instancia existente del equipo.")
    if args.modelo_uri:
        import mlflow

        verificar_tracking(args.tracking_uri)
        modelo = mlflow.sklearn.load_model(args.modelo_uri)
        identidad = {"id": args.modelo_uri, "origen": "registro_mlflow"}
    else:
        if not args.identidad_modelo:
            parser.error("Indica --identidad-modelo para documentar el artefacto local.")
        identidad = json.loads(args.identidad_modelo.read_text(encoding="utf-8"))
        huella_modelo = sha256_archivo(args.modelo_local)
        if identidad.get("artifact_sha256", huella_modelo) != huella_modelo:
            raise ValueError("La huella del modelo no coincide con su evidencia.")
        identidad["sha256_modelo_local"] = huella_modelo
        modelo = cargar_modelo_local(args.modelo_local)
    evidencia_fuente = args.fuente or args.features.with_suffix(".sources.json")
    fuente = (
        json.loads(evidencia_fuente.read_text(encoding="utf-8"))
        if args.fuente or evidencia_fuente.is_file()
        else None
    )
    salida = args.salida_dir / args.mes
    resultados = ejecutar_monitoreo(
        args.features, args.mes, modelo, identidad, salida, fuente=fuente
    )
    if args.registrar_mlflow:
        print(
            f"MLflow run: {registrar_evaluacion(args.tracking_uri, salida, resultados, args.artifact_location)}"
        )
    print(f"Reporte: {salida.resolve()}")


if __name__ == "__main__":
    main()
