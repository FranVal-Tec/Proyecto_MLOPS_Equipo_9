"""Abre el experimento compartido usando una copia de trabajo con rutas locales.

python -m proyecto_mlops_equipo_9.open_mlflow
python -m proyecto_mlops_equipo_9.open_mlflow --preparar
python -m proyecto_mlops_equipo_9.open_mlflow --port 5002
"""

import argparse
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATABASE = "mlflow_limpieza_integrada.db"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_only(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_path_text(uri):
    """Interpreta URIs de Windows o Unix sin depender del SO que recibe la copia."""
    parsed = urlsplit(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        return None
    path = unquote(parsed.path).replace("\\", "/")
    if len(path) >= 3 and path[0] == "/" and path[1].isalpha() and path[2] == ":":
        path = path[1:]
    return path.rstrip("/")


def source_roots(connection):
    roots = set()
    suffix = "/mlartifacts/limpieza_integrada"
    for (uri,) in connection.execute("SELECT artifact_location FROM experiments"):
        path = file_path_text(uri or "")
        if path and path.endswith(suffix):
            roots.add(path[: -len(suffix)])
    if not roots:
        raise ValueError("No se encontró la raíz de artefactos del experimento compartido.")
    return sorted(roots, key=len, reverse=True)


def relocate_uri(uri, roots, root):
    path = file_path_text(uri or "")
    if path is None:
        return uri
    for old_root in roots:
        if path.startswith(old_root + "/"):
            relative = path[len(old_root) + 1 :]
            if relative.split("/", 1)[0] not in {"data", "mlartifacts", "mlruns"}:
                raise ValueError(f"Referencia local fuera de las carpetas previstas: {relative}")
            destination = (root / relative).resolve()
            if not destination.is_relative_to(root):
                raise ValueError("La referencia intenta salir de la carpeta del proyecto.")
            return destination.as_uri()
    raise ValueError(f"Referencia file:// de otra raíz no reconocida: {uri}")


def server_running(work):
    record = work / "server.json"
    if not record.exists():
        return False
    info = json.loads(record.read_text(encoding="utf-8"))
    try:
        with socket.create_connection(("127.0.0.1", info["port"]), timeout=0.5):
            return True
    except OSError:
        return False


def inspect_local(database, root):
    """Comprueba referencias operativas y cuenta raw ausentes sin exigir su descarga."""
    with closing(read_only(database)) as connection:
        missing_raw, existing_data, missing_required = [], [], []
        run_ids = []
        for run_id, uri in connection.execute("SELECT run_uuid, artifact_uri FROM runs"):
            path = Path(file_path_text(uri) or "")
            if not path.is_relative_to(root) or not path.is_dir():
                missing_required.append(f"artefactos del run {run_id}")
            run_ids.append(run_id)
        for name, source in connection.execute("SELECT name, dataset_source FROM datasets"):
            config = json.loads(source)
            uri = config.get("uri")
            if not uri:
                raise ValueError(f"Fuente no soportada por esta utilidad: {name}")
            path = Path(file_path_text(uri) or "")
            if not path.is_relative_to(root):
                raise ValueError(f"Fuente fuera del proyecto actual: {name}")
            if path.is_file():
                existing_data.append(name)
            elif path.is_relative_to(root / "data/raw"):
                missing_raw.append(path.relative_to(root).as_posix())
            else:
                missing_required.append(name)
    if missing_required:
        raise FileNotFoundError(f"Faltan archivos compartidos: {missing_required}")
    return {
        "runs": run_ids,
        "datasets_disponibles": len(existing_data),
        "raw_no_descargados": missing_raw,
    }


def prepare(root=ROOT, *, refresh=False):
    root = Path(root).resolve()
    source = root / DATABASE
    if not source.is_file():
        raise FileNotFoundError(f"No existe la base compartida: {source}")
    work = root / ".local/mlflow"
    database = work / "tracking.db"
    report_path = work / "portabilidad.json"
    source_hash = sha256(source)
    if database.exists() and report_path.exists() and not refresh:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["source_sha256"] != source_hash or report["project_root"] != str(root):
            raise ValueError(
                "Cambió la base compartida o su ubicación. Usa --actualizar para adaptar una nueva copia."
            )
        inspect_local(database, root)
        return database, report
    if server_running(work):
        raise RuntimeError("Cierra el servidor de esta copia local antes de regenerar su base.")
    if database.exists() and not refresh:
        raise FileExistsError(
            "Hay una base local sin informe; usa --actualizar para conservarla como respaldo y regenerar."
        )
    work.mkdir(parents=True, exist_ok=True)
    if database.exists():
        backups = work / "backups"
        backups.mkdir(exist_ok=True)
        backup = backups / f"tracking_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S_%fZ')}.db"
        with closing(read_only(database)) as old, closing(sqlite3.connect(backup)) as destination:
            old.backup(destination)
    staged = work / f"tracking_{uuid.uuid4().hex}.tmp.db"
    # La API de backup de SQLite obtiene una copia consistente incluso si la base está abierta.
    with closing(read_only(source)) as shared, closing(sqlite3.connect(staged)) as local:
        shared.backup(local)
    try:
        with closing(sqlite3.connect(staged)) as local, local:
            roots = source_roots(local)
            before_runs = local.execute(
                "SELECT run_uuid,status FROM runs ORDER BY run_uuid"
            ).fetchall()
            before_datasets = Counter(
                local.execute("SELECT name,digest,dataset_profile FROM datasets")
            )
            counts = {"experiments": 0, "runs": 0, "datasets": 0, "tags": 0}
            for key, uri in local.execute(
                "SELECT experiment_id,artifact_location FROM experiments"
            ).fetchall():
                updated = relocate_uri(uri, roots, root)
                if updated != uri:
                    local.execute(
                        "UPDATE experiments SET artifact_location=? WHERE experiment_id=?",
                        (updated, key),
                    )
                    counts["experiments"] += 1
            for key, uri in local.execute("SELECT run_uuid,artifact_uri FROM runs").fetchall():
                updated = relocate_uri(uri, roots, root)
                if updated != uri:
                    local.execute(
                        "UPDATE runs SET artifact_uri=? WHERE run_uuid=?", (updated, key)
                    )
                    counts["runs"] += 1
            for key, kind, source_json in local.execute(
                "SELECT dataset_uuid,dataset_source_type,dataset_source FROM datasets"
            ).fetchall():
                if kind != "local":
                    raise ValueError(f"Tipo de fuente no soportado: {kind}")
                data = json.loads(source_json)
                updated = relocate_uri(data["uri"], roots, root)
                if updated != data["uri"]:
                    data["uri"] = updated
                    local.execute(
                        "UPDATE datasets SET dataset_source=? WHERE dataset_uuid=?",
                        (json.dumps(data), key),
                    )
                    counts["datasets"] += 1
            for run_id, key, value in local.execute(
                "SELECT run_uuid,key,value FROM tags WHERE key='almacenamiento_silver'"
            ).fetchall():
                updated = relocate_uri(value, roots, root)
                if updated != value:
                    local.execute(
                        "UPDATE tags SET value=? WHERE run_uuid=? AND key=?",
                        (updated, run_id, key),
                    )
                    counts["tags"] += 1
            if (
                before_runs
                != local.execute("SELECT run_uuid,status FROM runs ORDER BY run_uuid").fetchall()
            ):
                raise ValueError("Cambió la identidad o estado de los runs.")
            if before_datasets != Counter(
                local.execute("SELECT name,digest,dataset_profile FROM datasets")
            ):
                raise ValueError("Cambió la identidad o perfil de los datasets.")
            if local.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("La base local no superó la comprobación de integridad.")
        inspection = inspect_local(staged, root)
        if sha256(source) != source_hash:
            raise RuntimeError(
                "La base compartida cambió durante la preparación; vuelve a intentarlo."
            )
        os.replace(staged, database)
        report = {
            "project_root": str(root),
            "source_database": DATABASE,
            "source_sha256": source_hash,
            "working_database": str(database),
            "prepared_utc": datetime.now(UTC).isoformat(),
            "relocated": counts,
            "checks": inspection,
            "note": "Solo se adaptó la copia local; la base compartida, los Parquet y los artefactos no se modificaron.",
        }
        save_json(report_path, report)
        return database, report
    except Exception:
        staged.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preparar", action="store_true", help="Prepara y verifica rutas sin iniciar el servidor."
    )
    parser.add_argument(
        "--actualizar",
        action="store_true",
        help="Respalda la base local anterior y adapta otra copia de la compartida.",
    )
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Puerto fuera de rango.")
    if not args.preparar:
        try:
            with socket.create_connection(("127.0.0.1", args.port), timeout=0.5):
                parser.error(f"El puerto {args.port} está ocupado. Usa otro con --port.")
        except OSError:
            pass
    database, report = prepare(refresh=args.actualizar)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.preparar:
        return
    lock = database.parent / "server.json"
    if server_running(database.parent):
        raise RuntimeError("Ya hay un servidor usando esta copia local.")
    save_json(lock, {"pid": os.getpid(), "port": args.port})
    print(f"MLflow local: http://127.0.0.1:{args.port}", flush=True)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlflow",
                "server",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
                "--workers",
                "1",
                "--backend-store-uri",
                "sqlite:///" + database.as_posix(),
            ],
            cwd=ROOT,
            check=False,
        )
        raise SystemExit(result.returncode)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
