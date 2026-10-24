"""Registro asistido de una versión silver existente; nunca modifica la versión.

python -m proyecto_mlops_equipo_9.register_data --paso preparar
python -m proyecto_mlops_equipo_9.register_data --paso registrar
python -m proyecto_mlops_equipo_9.register_data --paso verificar
"""

import argparse
from collections import Counter
from datetime import UTC, datetime
import gc
import hashlib
import json
from pathlib import Path
import re

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTO = "mibici-limpieza-integrada"


def sha256(ruta):
    h = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            h.update(bloque)
    return h.hexdigest()


def guardar_json(ruta, data):
    Path(ruta).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def contexto(version):
    if not re.fullmatch(r"version_[0-9]{3,}", version):
        raise ValueError("Formato de versión inválido.")
    carpeta = ROOT / "data/silver" / version
    manifest = json.loads((carpeta / "manifest.json").read_text(encoding="utf-8"))
    sello = json.loads((carpeta / "_SUCCESS.json").read_text(encoding="utf-8"))
    if sello["manifest_sha256"] != sha256(carpeta / "manifest.json"):
        raise ValueError("El manifiesto de la versión completa fue modificado.")
    uri = "sqlite:///" + (ROOT / "mlflow_limpieza_integrada.db").as_posix()
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    return carpeta, manifest, uri, client


def preparar(version):
    carpeta, manifest, uri, client = contexto(version)
    experimento = client.get_experiment_by_name(EXPERIMENTO)
    if experimento is None:
        experiment_id = client.create_experiment(
            EXPERIMENTO, artifact_location=(ROOT / "mlartifacts/limpieza_integrada").as_uri()
        )
    else:
        experiment_id = experimento.experiment_id
    print(
        json.dumps(
            {
                "paso": "preparar",
                "tracking_uri": uri,
                "experimento": EXPERIMENTO,
                "experiment_id": experiment_id,
                "version": version,
                "dataset_id": manifest["dataset_id"],
                "fuente_silver": str(carpeta),
                "runs_creados_en_este_paso": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return experiment_id


def plan_datasets(carpeta, manifest):
    plan = []
    for entrada in manifest["entradas"]:
        if entrada["tipo"] == "viajes":
            ruta = ROOT / entrada["archivo"]
            nombre = f"mibici_raw_{entrada['mes']}"
            grupo = "raw"
        else:
            ruta = carpeta / "fuentes" / Path(entrada["archivo"]).name
            nombre = f"mibici_catalogo_{Path(entrada['archivo']).stem}"
            grupo = "catalogo"
        plan.append(
            {
                "ruta": ruta,
                "nombre": nombre,
                "contexto": grupo,
                "sha256": entrada["sha256"],
                "filas": entrada["filas"],
            }
        )
    for salida in manifest["salidas"]:
        if salida["tipo"] not in {"viajes_validos", "viajes_excluidos"}:
            continue
        grupo = "silver" if salida["tipo"] == "viajes_validos" else "revision"
        mes = Path(salida["archivo"]).stem.removeprefix("viajes_")
        plan.append(
            {
                "ruta": carpeta / salida["archivo"],
                "nombre": f"mibici_{grupo}_{manifest['version_datos']}_{mes}",
                "contexto": grupo,
                "sha256": salida["sha256"],
                "filas": salida["filas"],
            }
        )
    return plan


def leer_dataset(item):
    ruta = item["ruta"]
    if sha256(ruta) != item["sha256"]:
        raise ValueError(f"El archivo cambió desde su versión: {ruta}")
    if ruta.suffix == ".parquet":
        frame = pd.read_parquet(ruta)
    else:
        try:
            frame = pd.read_csv(ruta, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            frame = pd.read_csv(ruta, encoding="latin-1", low_memory=False)
    if len(frame) != item["filas"]:
        raise ValueError(f"Conteo incompatible con el manifiesto: {ruta}")
    # Digest compacto de 128 bits; la huella SHA-256 completa se guarda en tags y manifiestos.
    dataset = mlflow.data.from_pandas(
        frame, source=ruta.resolve().as_uri(), name=item["nombre"], digest=item["sha256"][:32]
    )
    return frame, dataset


def registrar(version):
    carpeta, manifest, uri, client = contexto(version)
    experimento = client.get_experiment_by_name(EXPERIMENTO)
    if experimento is None:
        raise ValueError("Ejecuta primero --paso preparar.")
    recibo_dir = ROOT / "reports/versionado" / version
    recibo = recibo_dir / "registro_mlflow.json"
    if recibo.exists():
        registrado = json.loads(recibo.read_text(encoding="utf-8"))
        run = client.get_run(registrado["run_id"])
        if (
            run.info.status == "FINISHED"
            and run.data.tags.get("dataset_id") == manifest["dataset_id"]
        ):
            print(f"La versión ya tiene registro: {run.info.run_id}. No se creará otro run.")
            return
        raise ValueError("Existe un recibo que requiere revisión; no se sobrescribirá.")
    anteriores = client.search_runs(
        [experimento.experiment_id],
        filter_string=f"tags.dataset_id = '{manifest['dataset_id']}' AND tags.version_datos = '{version}' AND attributes.status = 'FINISHED'",
    )
    if anteriores:
        raise ValueError(
            f"Ya existe un run terminado ({anteriores[0].info.run_id}); recupera su recibo antes de repetir."
        )
    plan = plan_datasets(carpeta, manifest)
    # Comprobar serialización de tipos y fuente sin crear un run de prueba.
    prueba, dataset_prueba = leer_dataset(next(p for p in plan if p["contexto"] == "silver"))
    dataset_prueba.to_json()
    del prueba, dataset_prueba
    registro_datasets = []
    with mlflow.start_run(
        experiment_id=experimento.experiment_id, run_name=f"silver_{version}"
    ) as run:
        mlflow.set_tags(
            {
                "version_datos": version,
                "dataset_id": manifest["dataset_id"],
                "etapa": "limpieza",
                "capa": "silver",
                "usuario_proceso": manifest["usuario"],
                "codigo_limpieza_sha256": manifest["codigo_sha256"]["cleaning.py"],
                "manifest_sha256": sha256(carpeta / "manifest.json"),
                "almacenamiento_silver": carpeta.as_uri(),
                "tipo_registro": "version_datos; sin entrenamiento",
            }
        )
        mlflow.log_params(
            {
                **manifest["politica"],
                "catalogo": manifest["campos_control_constantes"]["catalogo_version"],
                "n_meses": len(manifest["meses"]),
                "formato": "parquet",
                "compresion": "zstd",
            }
        )
        mlflow.log_metrics(
            {
                **manifest["resumen"],
                "datasets_registrados": len(plan),
                "pct_excluido": 100
                * manifest["resumen"]["excluidos"]
                / manifest["resumen"]["originales"],
            }
        )
        for item in plan:
            frame, dataset = leer_dataset(item)
            mlflow.log_input(
                dataset,
                context=item["contexto"],
                tags={
                    "sha256_archivo": item["sha256"],
                    "version_datos": version,
                    "rol": "entrada" if item["contexto"] in {"raw", "catalogo"} else "salida",
                },
            )
            registro_datasets.append(
                {
                    "nombre": dataset.name,
                    "digest": dataset.digest,
                    "contexto": item["contexto"],
                    "filas": len(frame),
                    "source": item["ruta"].resolve().as_uri(),
                    "sha256": item["sha256"],
                }
            )
            print(
                f"Dataset registrado: {item['contexto']} | {item['nombre']} | {len(frame):,} filas",
                flush=True,
            )
            del frame, dataset
            gc.collect()
        for ruta in sorted(carpeta.iterdir()):
            if ruta.is_file():
                mlflow.log_artifact(str(ruta), artifact_path="version")
        mlflow.log_artifacts(str(carpeta / "metadatos"), artifact_path="metadatos")
        mlflow.log_artifacts(str(carpeta / "fuentes/codigo"), artifact_path="codigo_limpieza")
        mlflow.log_artifact(
            str(carpeta / "fuentes/notebook_limpieza.ipynb"), artifact_path="metodologia"
        )
        mlflow.log_artifact(__file__, artifact_path="codigo_registro")
        run_id = run.info.run_id
    recibo_dir.mkdir(parents=True, exist_ok=True)
    guardar_json(
        recibo,
        {
            "estado": "REGISTRADO_PENDIENTE_VERIFICACION",
            "tracking_uri": uri,
            "experimento": EXPERIMENTO,
            "experiment_id": experimento.experiment_id,
            "run_id": run_id,
            "version_datos": version,
            "dataset_id": manifest["dataset_id"],
            "manifest_sha256": sha256(carpeta / "manifest.json"),
            "mlflow_version": mlflow.__version__,
            "registro_utc": datetime.now(UTC).isoformat(),
            "datasets": registro_datasets,
        },
    )
    print(f"Run terminado: {run_id}. Siguiente paso: --paso verificar.", flush=True)


def verificar(version):
    carpeta, manifest, _uri, client = contexto(version)
    ruta_recibo = ROOT / "reports/versionado" / version / "registro_mlflow.json"
    recibo = json.loads(ruta_recibo.read_text(encoding="utf-8"))
    run = client.get_run(recibo["run_id"])
    plan = plan_datasets(carpeta, manifest)
    if run.info.status != "FINISHED" or run.data.tags["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("El run no terminó o no corresponde a este dataset.")
    esperados = Counter((p["nombre"], p["sha256"][:32], p["contexto"]) for p in plan)
    reales = Counter()
    for item in run.inputs.dataset_inputs:
        tags = {tag.key: tag.value for tag in item.tags}
        reales[(item.dataset.name, item.dataset.digest, tags["mlflow.data.context"])] += 1
        esperado = next(p for p in plan if p["nombre"] == item.dataset.name)
        if tags.get("sha256_archivo") != esperado["sha256"]:
            raise ValueError("La huella de archivo registrada no coincide.")
        profile = json.loads(item.dataset.profile)
        if profile["num_rows"] != esperado["filas"]:
            raise ValueError("El perfil del dataset no coincide con sus filas.")
    if reales != esperados:
        raise ValueError("Los datasets registrados no coinciden con el plan.")
    for nombre, valor in manifest["resumen"].items():
        if run.data.metrics[nombre] != valor:
            raise ValueError("Métrica de conteo incompatible.")
    if not client.list_artifacts(run.info.run_id, "version"):
        raise ValueError("No hay evidencia de versión en los artefactos.")
    recibo["estado"] = "VERIFICADO"
    recibo["verificacion"] = {
        "run_status": run.info.status,
        "datasets": len(plan),
        "por_contexto": dict(Counter(p["contexto"] for p in plan)),
        "resumen": manifest["resumen"],
        "verificado_utc": datetime.now(UTC).isoformat(),
    }
    guardar_json(ruta_recibo, recibo)
    client.log_artifact(run.info.run_id, str(ruta_recibo), artifact_path="registro")
    print(
        json.dumps(
            {k: v for k, v in recibo.items() if k != "datasets"}, ensure_ascii=False, indent=2
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="version_001")
    parser.add_argument("--paso", choices=["preparar", "registrar", "verificar"], required=True)
    args = parser.parse_args()
    {"preparar": preparar, "registrar": registrar, "verificar": verificar}[args.paso](args.version)


if __name__ == "__main__":
    main()
