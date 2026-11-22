"""Prepara una extensión mensual de silver con el contrato de features del equipo."""

import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from proyecto_mlops_equipo_9 import cleaning, dataset, features

OCTOBER_URL = "https://mibici.net/site/assets/files/133188/datos_abiertos_2026_10.csv"
OCTOBER_SHA256 = "0cdfb05299e6db658ac9cc7723494a9704b0749e3671c29c8566826ea889b900"


def download_october(destination):
    """Descarga octubre mediante el cliente del proyecto y devuelve la ruta local."""
    from proyecto_mlops_equipo_9.download import descargar

    destination = Path(destination)
    if not destination.is_file():
        descargar(f"{OCTOBER_URL}?v=202610", destination, "viajes")
    return destination


def _load_trips(path):
    raw = dataset._leer_csv(path, na_values=["NULL", ""], low_memory=False)
    raw = raw.rename(columns=dataset.nombre_estandar)
    if list(raw.columns) != dataset.COLUMNAS:
        raise ValueError(f"El CSV no tiene el esquema de viajes esperado: {path.name}")
    for column in ("inicio", "fin"):
        raw[column] = dataset._a_fecha(raw[column])
    raw["genero"] = raw["genero"].astype("category")
    raw["duracion_min"] = (raw["fin"] - raw["inicio"]).dt.total_seconds() / 60
    return raw


def _station_coords(catalog_path):
    stations = dataset.cargar_estaciones(catalog_path)
    if stations.id.isna().any() or not stations.id.is_unique:
        raise ValueError("El catálogo requiere IDs de estación únicos y no nulos.")
    coords = stations.set_index("id")[["latitude", "longitude"]].apply(
        pd.to_numeric, errors="coerce"
    )
    valid = (
        np.isfinite(coords).all(axis=1)
        & coords.latitude.between(-90, 90)
        & coords.longitude.between(-180, 180)
    )
    return coords.where(valid, np.nan)


def _existing_ids(silver, new_ids):
    seen = set()
    for directory in ("viajes", "revision"):
        for path in sorted((silver / directory).glob("viajes_????_??.parquet")):
            trip_ids = pd.read_parquet(path, columns=["viaje_id"])["viaje_id"]
            seen.update(trip_ids[trip_ids.isin(new_ids)].unique())
    return seen


def _summarize_source(raw, accepted, bitacora, period, path, fingerprint, coords):
    day_counts = raw.inicio.dropna().dt.normalize().value_counts()
    expected_days = pd.date_range(period.start_time, (period + 1).start_time, inclusive="left")
    absent_days = expected_days.difference(day_counts.index)
    return {
        "periodo": str(period),
        "archivo": path.name,
        "url": OCTOBER_URL if str(period) == "2026-10" else None,
        "sha256": fingerprint,
        "coincide_con_snapshot_octubre": (
            fingerprint == OCTOBER_SHA256 if str(period) == "2026-10" else None
        ),
        "bytes": path.stat().st_size,
        "filas_crudas": len(raw),
        "filas_aceptadas": len(accepted),
        "filas_excluidas": len(raw) - len(accepted),
        "exclusion_por_motivo": {
            row.paso: int(row.filas_excluidas)
            for row in bitacora.itertuples()
            if row.filas_excluidas
        },
        "estaciones_origen_crudas": int(raw.origen_id.nunique()),
        "estaciones_origen_aceptadas": int(accepted.origen_id.nunique()),
        "ids_estacion_fuera_catalogo": sorted(
            int(station_id)
            for station_id in set(raw.origen_id.dropna()).union(raw.destino_id.dropna())
            if station_id not in coords.index
        ),
        "dias_sin_registros_en_archivo": absent_days.strftime("%Y-%m-%d").tolist(),
    }


def prepare_month(project_root, month, month_csv, work_dir, *, prior_csvs=None):
    """Construye features hasta el mes solicitado con silver y los CSV posteriores.

    Para meses posteriores a octubre deben pasarse los CSV intermedios mediante
    'prior_csvs', un mapeo de mes AAAA-MM a ruta. El catálogo queda fijado al de
    silver 001 para comparar sobre la misma geografía. La ayuda de auditoría del
    pipeline registra sus descartes y evita extender el alcance sin evidencia.
    """
    root = Path(project_root).resolve()
    silver = root / "data/silver/version_001"
    manifest_path = silver / "manifest.json"
    policy_path = silver / "politica.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    historical = sorted((silver / "viajes").glob("viajes_????_??.parquet"))
    if not historical:
        raise FileNotFoundError(f"No hay particiones silver: {silver / 'viajes'}")
    last_period = pd.Period(historical[-1].stem[-7:].replace("_", "-"), freq="M")
    target = pd.Period(month, freq="M")
    new_files = {pd.Period(key, freq="M"): Path(path) for key, path in (prior_csvs or {}).items()}
    new_files[target] = Path(month_csv)
    expected = pd.period_range(last_period + 1, target, freq="M")
    if target <= last_period:
        raise ValueError(f"El mes solicitado debe ser posterior a {last_period}.")
    if set(new_files) != set(expected):
        required = ", ".join(map(str, expected))
        raise ValueError(f"Proporciona exactamente los archivos mensuales {required}.")
    file_hashes = {period: features.sha256_archivo(new_files[period]) for period in expected}

    catalog_name = manifest["campos_control_constantes"]["catalogo_version"]
    catalog = silver / "fuentes" / catalog_name
    manifest_hash = features.sha256_archivo(manifest_path)
    catalog_hash = features.sha256_archivo(catalog)
    if catalog_hash != manifest["campos_control_constantes"]["catalogo_sha256"]:
        raise ValueError("El catálogo no coincide con el manifiesto de silver.")
    identity = {
        "manifest_silver_sha256": manifest_hash,
        "catalogo": catalog.name,
        "catalogo_sha256": catalog_hash,
        "politica_sha256": features.sha256_archivo(policy_path),
        "codigo_sha256": {
            Path(path).name: features.sha256_archivo(path)
            for path in (cleaning.__file__, dataset.__file__, features.__file__, __file__)
        },
        "csv_sha256": {str(period): file_hashes[period] for period in expected},
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    output_dir = Path(work_dir).resolve() / str(target) / digest[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"features_hasta_{target}.parquet"
    evidence_path = output.with_suffix(".sources.json")
    if (
        output.is_file()
        and evidence_path.is_file()
        and output.with_suffix(".metadata.json").is_file()
    ):
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence.get("identidad") == identity and evidence.get(
            "features_sha256"
        ) == features.sha256_archivo(output):
            return output

    coords = _station_coords(catalog)
    raw_months = {period: _load_trips(path) for period, path in new_files.items()}
    for period, path in new_files.items():
        if features.sha256_archivo(path) != file_hashes[period]:
            raise ValueError(f"El archivo {path.name} cambió durante la lectura.")
    distinct_ids = pd.concat(
        [raw.viaje_id.dropna().drop_duplicates() for raw in raw_months.values()]
    )
    earlier_ids = _existing_ids(silver, distinct_ids.unique())
    later_repeated_ids = set(distinct_ids[distinct_ids.duplicated()].unique())
    duplicate_ids = earlier_ids | later_repeated_ids

    source_summaries = []
    with TemporaryDirectory(prefix="build-", dir=output_dir) as temporary:
        build = Path(temporary)
        stage = build / "silver"
        trips_path = stage / "viajes"
        trips_path.mkdir(parents=True)
        (stage / "fuentes").mkdir()
        for source in historical:
            shutil.copyfile(source, trips_path / source.name)
        shutil.copyfile(catalog, stage / "fuentes" / catalog.name)

        for period in expected:
            raw = raw_months[period]
            label = str(period).replace("-", "_")
            annotated, accepted, bitacora = cleaning.limpiar_mes(
                raw, label, coords, policy, duplicate_ids
            )
            del annotated
            if accepted.empty:
                raise ValueError(f"No hay viajes aceptados para {period}.")
            part = trips_path / f"viajes_{label}.parquet"
            columns = ["inicio", "fin", "origen_id", "destino_id", "es_valido"]
            accepted[columns].to_parquet(part, index=False)
            manifest["salidas"].append(
                {
                    "archivo": part.relative_to(stage).as_posix(),
                    "sha256": features.sha256_archivo(part),
                    "filas": len(accepted),
                }
            )
            summary = _summarize_source(
                raw, accepted, bitacora, period, new_files[period], file_hashes[period], coords
            )
            if summary["dias_sin_registros_en_archivo"]:
                raise ValueError(f"El CSV de {period} no cubre todos sus días.")
            source_summaries.append(summary)

        manifest["dataset_id"] = digest
        manifest["extension_monitoreo"] = identity
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        work_output = build / output.name
        result = features.ejecutar_features(
            input_path=trips_path, output_path=work_output, registrar_mlflow=False
        )
        evidence = {
            "identidad": identity,
            "id_extension_silver": digest,
            "fuentes_mensuales": source_summaries,
            "features": output.name,
            "features_sha256": result.metadata["parquet_sha256"],
            "filas_features": len(result.dataset),
            "ventana_desde": result.metadata["fuentes"]["desde"],
            "ventana_hasta_exclusivo": result.metadata["fuentes"]["hasta_exclusivo"],
            "politica_catalogo": "Catálogo de silver 001, con exclusiones cuantificadas por mes.",
            "politica_id_repetidos": "Archivos nuevos cotejados entre sí y contra silver 001; histórico fijo.",
        }
        work_evidence = build / evidence_path.name
        work_evidence.write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        work_output.replace(output)
        result.metadata_path.replace(output.with_suffix(".metadata.json"))
        work_evidence.replace(evidence_path)
    return output


def prepare_october(project_root, october_csv, work_dir):
    """Construye las features que incluyen octubre de 2026."""
    return prepare_month(project_root, "2026-10", october_csv, work_dir)
