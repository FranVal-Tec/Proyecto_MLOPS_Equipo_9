"""Selección, evaluación y registro de modelos de demanda por zona y hora.

Uso: python -m proyecto_mlops_equipo_9.modeling.train
Importar este módulo no lee datos ni conecta con MLflow.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from proyecto_mlops_equipo_9.config import PROCESSED_DATA_DIR, REPORTS_DIR
from proyecto_mlops_equipo_9.modeling.validation import (
    OBJETIVOS,
    CalendarioValidacion,
    indices_ventana,
    mejora_contra_base,
    metricas,
    metricas_por_volumen,
    prediccion_base,
    scorer_metricas,
    ventana_periodo,
    ventanas_mensuales,
)

CATEGORICAS_LINEAL = ["zona_id", "hora_objetivo", "dia_semana_objetivo"]

# Hiperparámetros candidatos; `modelo__estimator__` apunta al regresor de cada objetivo.
ESPACIOS = {
    "ridge": {"modelo__estimator__alpha": [0.1, 1.0, 10.0, 100.0]},
    "random_forest": {
        "modelo__estimator__n_estimators": [50, 100],
        "modelo__estimator__max_depth": [12, 18],
        "modelo__estimator__min_samples_leaf": [5, 20],
        "modelo__estimator__max_features": [0.33, 0.5],
    },
    "hist_gradient_boosting": {
        "modelo__estimator__loss": ["poisson", "squared_error"],
        "modelo__estimator__learning_rate": [0.05, 0.1],
        "modelo__estimator__max_iter": [200, 400],
        "modelo__estimator__max_leaf_nodes": [31, 63],
        "modelo__estimator__min_samples_leaf": [20, 50],
    },
}


# El modelo se entrena en este mismo proceso; solo se confía en tipos de estas librerías.
PREFIJOS_CONFIABLES = ("sklearn.", "numpy.", "scipy.")


@dataclass(frozen=True)
class ConfiguracionEntrenamiento:
    """Parámetros del experimento; las fechas viven en CalendarioValidacion."""

    semilla: int = 9
    n_iter: int = 5
    max_muestras_bosque: float = 0.2
    mejora_minima: float = 0.10
    # Con 21 meses, estas variables identifican el periodo en lugar de un patrón repetible.
    excluidas: tuple[str, ...] = ("anio_objetivo", "dia_anio_objetivo", "dia_mes_objetivo")
    nombre_registro: str = "mibici-demanda-zona-hora"
    alias_registro: str = "candidato"


def preparar_dataset(df, features, config):
    """Ordena por tiempo, fija tipos numéricos y devuelve las columnas de entrada."""
    columnas = [c for c in features if c not in config.excluidas]
    out = df.sort_values(["inicio_hora_objetivo", "zona_id"]).reset_index(drop=True)
    numericas = [c for c in columnas if c != "zona_id"]
    out[numericas] = out[numericas].astype("float64")
    out["zona_id"] = out.zona_id.astype(str)
    out[OBJETIVOS] = out[OBJETIVOS].astype("float64")
    return out, columnas


def crear_pipeline(nombre, columnas, config):
    """Preprocesamiento y regresor; se ajusta solo con datos de entrenamiento."""
    if nombre == "ridge":
        categoricas = [c for c in CATEGORICAS_LINEAL if c in columnas]
        numericas = [c for c in columnas if c not in categoricas]
        preprocesador = ColumnTransformer(
            [
                ("categoricas", OneHotEncoder(handle_unknown="ignore"), categoricas),
                ("numericas", StandardScaler(), numericas),
            ]
        )
        estimador = Ridge()
    else:
        # La zona queda en la primera columna; una zona nueva se trata como faltante (-1).
        codificador = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        preprocesador = ColumnTransformer(
            [
                ("zona", codificador, ["zona_id"]),
                ("numericas", "passthrough", [c for c in columnas if c != "zona_id"]),
            ]
        )
        if nombre == "random_forest":
            estimador = RandomForestRegressor(
                max_samples=config.max_muestras_bosque, n_jobs=-1, random_state=config.semilla
            )
        elif nombre == "hist_gradient_boosting":
            estimador = HistGradientBoostingRegressor(
                categorical_features=[0], early_stopping=False, random_state=config.semilla
            )
        else:
            raise ValueError(f"Modelo desconocido: {nombre}.")
    return Pipeline(
        [("preprocesamiento", preprocesador), ("modelo", MultiOutputRegressor(estimador))]
    )


def buscar_hiperparametros(nombre, df, columnas, ventanas, config):
    """Compara configuraciones en todas las ventanas; ningún dato se usa antes de su corte."""
    busqueda = RandomizedSearchCV(
        crear_pipeline(nombre, columnas, config),
        ESPACIOS[nombre],
        n_iter=config.n_iter,
        scoring=scorer_metricas,
        cv=[indices_ventana(df, ventana) for ventana in ventanas],
        refit=False,
        error_score="raise",
        random_state=config.semilla,
    )
    busqueda.fit(df[columnas], df[OBJETIVOS])
    resultados = pd.DataFrame(busqueda.cv_results_)
    mejor = int(resultados.mean_test_mae.idxmax())
    parametros = resultados.loc[mejor, "params"]
    por_ventana = [
    {
        metrica: -float(resultados.loc[mejor, f"split{i}_test_{metrica}"])
        for metrica in ("mae", "mae_salidas", "mae_llegadas", "mae_flujo_neto", "wape")
    }
    for i in range(len(ventanas))
    ]
    return parametros, por_ventana


def resumir(por_ventana):
    """Promedio y dispersión entre ventanas de cada métrica."""
    tabla = pd.DataFrame(por_ventana)
    return {
        **{f"{c}_promedio": float(tabla[c].mean()) for c in tabla},
        **{f"{c}_desviacion": float(tabla[c].std(ddof=0)) for c in tabla},
    }


def evaluar_base_ventanas(df, ventanas):
    """Métricas de la línea base con las mismas filas que evalúan los modelos."""
    return [
        metricas(
            df.iloc[evaluacion][OBJETIVOS], prediccion_base(df.iloc[evaluacion])
        )
        for _, evaluacion in (indices_ventana(df, v) for v in ventanas)
    ]


def ajustar_y_evaluar(nombre, parametros, df, columnas, ventana, config):
    """Ajusta con todo lo anterior al corte y evalúa la ventana una sola vez."""
    entrenamiento, evaluacion = indices_ventana(df, ventana)
    pipeline = crear_pipeline(nombre, columnas, config).set_params(**parametros)
    pipeline.fit(df.iloc[entrenamiento][columnas], df.iloc[entrenamiento][OBJETIVOS])
    return pipeline, evaluar_ajustado(pipeline, df, columnas, entrenamiento, evaluacion)


def evaluar_ajustado(pipeline, df, columnas, entrenamiento, evaluacion):
    """Métricas del modelo, de la línea base y por volumen de zona."""
    filas = df.iloc[evaluacion]
    predicho = pipeline.predict(filas[columnas])
    resultado = metricas(filas[OBJETIVOS], predicho)
    base = metricas(filas[OBJETIVOS], prediccion_base(filas))
    return {
        **resultado,
        **metricas_por_volumen(filas, predicho, df.iloc[entrenamiento]),
        "mae_linea_base": base["mae"],
        "mae_flujo_neto_linea_base": base["mae_flujo_neto"],
        "mejora_contra_base": mejora_contra_base(resultado["mae"], base["mae"]),
    }


def evaluar_drift(nombre, parametros, df, columnas, modelo_fijo, ventanas, config):
    """Compara, mes a mes, el modelo congelado contra uno reentrenado al corte."""
    filas = []
    for ventana in ventanas:
        entrenamiento, evaluacion = indices_ventana(df, ventana)
        fijo = evaluar_ajustado(modelo_fijo, df, columnas, entrenamiento, evaluacion)
        _, reentrenado = ajustar_y_evaluar(nombre, parametros, df, columnas, ventana, config)
        filas.append(
            {
                "mes": ventana.nombre,
                "mae_fijo": fijo["mae"],
                "mae_reentrenado": reentrenado["mae"],
                "mae_linea_base": fijo["mae_linea_base"],
                "mejora_fijo": fijo["mejora_contra_base"],
                "mejora_reentrenado": reentrenado["mejora_contra_base"],
            }
        )
    return filas


def registrar_ventanas(mlflow, por_ventana, ventanas):
    """Un run hijo por ventana, para compararlas en la interfaz de MLflow."""
    for ventana, valores in zip(ventanas, por_ventana):
        with mlflow.start_run(run_name=f"ventana_{ventana.nombre}", nested=True):
            mlflow.log_param("ventana", ventana.nombre)
            mlflow.log_metrics(valores)


def tipos_confiables(modelo):
    """Tipos internos que skops pide confirmar para guardar y cargar el modelo.

    Se aceptan clases de scikit-learn, NumPy y SciPy, y functools.partial; cualquier otro
    tipo detiene el registro para revisarlo.
    """
    import skops.io as sio

    tipos = sio.get_untrusted_types(data=sio.dumps(modelo))
    ajenos = [
        t for t in tipos if not t.startswith(PREFIJOS_CONFIABLES) and t != "functools.partial"
    ]
    if ajenos:
        raise ValueError(f"El modelo contiene tipos no revisados: {ajenos}")
    return tipos


def registrar_modelo(mlflow, modelo, df, columnas, nombre, parametros, prueba, config):
    """Publica el modelo en el Model Registry con versión, parámetros y métricas."""
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    ejemplo = df[columnas].head(5)
    info = mlflow.sklearn.log_model(
        modelo,
        name="modelo",
        registered_model_name=config.nombre_registro,
        skops_trusted_types=tipos_confiables(modelo),
        signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
        input_example=ejemplo,
    )
    version = info.registered_model_version
    cliente = MlflowClient()
    cliente.set_registered_model_alias(config.nombre_registro, config.alias_registro, version)
    etiquetas = {
        "algoritmo": nombre,
        "hiperparametros": json.dumps(parametros, sort_keys=True),
        "mae_prueba": prueba["mae"],
        "mae_flujo_neto_prueba": prueba["mae_flujo_neto"],
        "mejora_contra_base_prueba": prueba["mejora_contra_base"],
    }
    for clave, valor in etiquetas.items():
        cliente.set_model_version_tag(config.nombre_registro, version, clave, str(valor))
    return version


def guardar_resultados(salida_dir, comparacion, filas_drift, resumen):
    """Escribe la comparación de modelos, el drift mensual y el resultado de prueba."""
    salida_dir = Path(salida_dir)
    salida_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparacion).to_csv(salida_dir / "comparacion_modelos.csv", index=False)
    pd.DataFrame(filas_drift).to_csv(salida_dir / "drift_mensual.csv", index=False)
    (salida_dir / "resultado_prueba.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def ejecutar_entrenamiento(
    dataset_path,
    salida_dir,
    tracking_uri,
    experiment_name,
    modelos,
    config=None,
    calendario=None,
):
    """Selecciona con ventanas, prueba una vez, compara drift y registra al ganador."""
    import mlflow

    config = config or ConfiguracionEntrenamiento()
    calendario = calendario or CalendarioValidacion()
    dataset_path = Path(dataset_path)
    metadata = json.loads(dataset_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    df, columnas = preparar_dataset(pd.read_parquet(dataset_path), metadata["features"], config)
    ventanas = ventanas_mensuales(calendario.seleccion_desde, calendario.seleccion_hasta)
    prueba = ventana_periodo("prueba", calendario.prueba_desde, calendario.prueba_hasta)
    drift = ventanas_mensuales(calendario.drift_desde, calendario.drift_hasta)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="seleccion_modelos"):
        mlflow.log_params(
            {
                **{k: v for k, v in asdict(config).items() if k != "excluidas"},
                **asdict(calendario),
                "variables": len(columnas),
                "filas": len(df),
                "dataset_sha256": metadata.get("parquet_sha256"),
            }
        )
        mlflow.log_dict({"variables": columnas, "objetivos": OBJETIVOS}, "columnas.json")

        base = evaluar_base_ventanas(df, ventanas)
        comparacion = [{"modelo": "linea_base", **resumir(base)}]
        with mlflow.start_run(run_name="linea_base", nested=True):
            mlflow.log_param("modelo", "linea_base")
            mlflow.log_metrics(resumir(base))
            registrar_ventanas(mlflow, base, ventanas)

        mejores = {}
        for nombre in modelos:
            with mlflow.start_run(run_name=nombre, nested=True):
                parametros, por_ventana = buscar_hiperparametros(
                    nombre, df, columnas, ventanas, config
                )
                resumen = resumir(por_ventana)
                resumen["mejora_contra_base"] = mejora_contra_base(
                    resumen["mae_promedio"], comparacion[0]["mae_promedio"]
                )
                mlflow.log_param("modelo", nombre)
                mlflow.log_params(parametros)
                mlflow.log_metrics(resumen)
                registrar_ventanas(mlflow, por_ventana, ventanas)
            mejores[nombre] = (parametros, resumen["mae_promedio"])
            comparacion.append({"modelo": nombre, **resumen})

        ganador = min(mejores, key=lambda n: mejores[n][1])
        parametros = mejores[ganador][0]
        with mlflow.start_run(run_name=f"prueba_{ganador}", nested=True):
            modelo, resultado_prueba = ajustar_y_evaluar(
                ganador, parametros, df, columnas, prueba, config
            )
            filas_drift = evaluar_drift(ganador, parametros, df, columnas, modelo, drift, config)
            cumple = resultado_prueba["mejora_contra_base"] >= config.mejora_minima
            mlflow.log_param("modelo", ganador)
            mlflow.log_params(parametros)
            mlflow.log_metrics(resultado_prueba)
            mlflow.set_tag("cumple_criterio_despliegue", str(cumple))
            for paso, fila in enumerate(filas_drift):
                valores = {k: v for k, v in fila.items() if k != "mes"}
                mlflow.log_metrics({f"drift_{k}": v for k, v in valores.items()}, step=paso)
            resumen_final = {
                "ganador": ganador,
                "hiperparametros": parametros,
                "prueba": resultado_prueba,
                "cumple_criterio_despliegue": bool(cumple),
                "version_registro": None,
            }
            # Se guardan antes de registrar: si el registro falla, los resultados no se pierden.
            guardar_resultados(salida_dir, comparacion, filas_drift, resumen_final)
            resumen_final["version_registro"] = str(
                registrar_modelo(
                    mlflow, modelo, df, columnas, ganador, parametros, resultado_prueba, config
                )
            )
        guardar_resultados(salida_dir, comparacion, filas_drift, resumen_final)
        mlflow.log_artifacts(str(salida_dir), artifact_path="resultados")
    return resumen_final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROCESSED_DATA_DIR / "datos_feature_engineered_zona_hora.parquet",
    )
    parser.add_argument("--salida", type=Path, default=REPORTS_DIR / "modelado")
    parser.add_argument("--experiment-name", default="mibici-entrenamiento")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--modelos", nargs="+", choices=sorted(ESPACIOS), default=sorted(ESPACIOS))
    parser.add_argument("--n-iter", type=int, default=ConfiguracionEntrenamiento.n_iter)
    args = parser.parse_args()
    resultado = ejecutar_entrenamiento(
        args.dataset,
        args.salida,
        args.tracking_uri,
        args.experiment_name,
        args.modelos,
        ConfiguracionEntrenamiento(n_iter=args.n_iter),
    )
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
