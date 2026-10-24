from contextlib import closing
import json
from pathlib import Path
import sqlite3

import pytest

from proyecto_mlops_equipo_9.open_mlflow import DATABASE, prepare, relocate_uri, sha256


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "proyecto copiado"
    root.mkdir()
    artifacts = root / "mlartifacts/limpieza_integrada/run1/artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "politica.json").write_text("{}", encoding="utf-8")
    silver = root / "data/silver/version_001/viajes.parquet"
    silver.parent.mkdir(parents=True)
    silver.write_bytes(b"fixture")
    old = "file:///C:/equipo/original"
    with closing(sqlite3.connect(root / DATABASE)) as db, db:
        db.executescript("""
            CREATE TABLE experiments (experiment_id INTEGER, artifact_location TEXT);
            CREATE TABLE runs (run_uuid TEXT, artifact_uri TEXT, status TEXT);
            CREATE TABLE datasets (dataset_uuid TEXT, name TEXT, digest TEXT,
                dataset_profile TEXT, dataset_source_type TEXT, dataset_source TEXT);
            CREATE TABLE tags (run_uuid TEXT, key TEXT, value TEXT);
        """)
        db.execute(
            "INSERT INTO experiments VALUES (?,?)", (1, old + "/mlartifacts/limpieza_integrada")
        )
        db.execute(
            "INSERT INTO runs VALUES (?,?,?)",
            ("run1", old + "/mlartifacts/limpieza_integrada/run1/artifacts", "FINISHED"),
        )
        for key, path in [
            ("silver", "data/silver/version_001/viajes.parquet"),
            ("raw", "data/raw/viajes/mes.csv"),
        ]:
            db.execute(
                "INSERT INTO datasets VALUES (?,?,?,?,?,?)",
                (
                    key,
                    key,
                    "digest-" + key,
                    '{"num_rows": 3}',
                    "local",
                    json.dumps({"uri": old + "/" + path}),
                ),
            )
        db.execute(
            "INSERT INTO tags VALUES (?,?,?)",
            ("run1", "almacenamiento_silver", old + "/data/silver/version_001"),
        )
    return root


def test_relocation_preserves_shared_database_and_identity(project):
    original = sha256(project / DATABASE)
    database, report = prepare(project)
    assert sha256(project / DATABASE) == original
    assert report["checks"]["datasets_disponibles"] == 1
    assert report["checks"]["raw_no_descargados"] == ["data/raw/viajes/mes.csv"]
    with closing(sqlite3.connect(database)) as db, db:
        assert db.execute("SELECT run_uuid,status FROM runs").fetchone() == ("run1", "FINISHED")
        assert (
            db.execute("SELECT digest FROM datasets WHERE name='silver'").fetchone()[0]
            == "digest-silver"
        )
        uri = json.loads(
            db.execute("SELECT dataset_source FROM datasets WHERE name='silver'").fetchone()[0]
        )["uri"]
        assert uri == (project / "data/silver/version_001/viajes.parquet").as_uri()
    assert prepare(project)[0] == database


def test_missing_silver_fails_without_publishing_local_database(project):
    (project / "data/silver/version_001/viajes.parquet").unlink()
    with pytest.raises(FileNotFoundError, match="silver"):
        prepare(project)
    assert not (project / ".local/mlflow/tracking.db").exists()


def test_refresh_requires_flag_and_preserves_previous_copy(project):
    database, _ = prepare(project)
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("INSERT INTO tags VALUES ('run1','personal','conservar')")
    with closing(sqlite3.connect(project / DATABASE)) as db, db:
        db.execute("INSERT INTO tags VALUES ('run1','nuevo','compartido')")
    with pytest.raises(ValueError, match="--actualizar"):
        prepare(project)
    prepare(project, refresh=True)
    backups = list((database.parent / "backups").glob("*.db"))
    assert len(backups) == 1
    with closing(sqlite3.connect(backups[0])) as db:
        assert db.execute("SELECT value FROM tags WHERE key='personal'").fetchone() == (
            "conservar",
        )


@pytest.mark.parametrize(
    "uri", ["file:///C:/otro/data/a.csv", "file:///C:/equipo/original/data/../../escape.csv"]
)
def test_rejects_paths_outside_project(tmp_path, uri):
    with pytest.raises(ValueError):
        relocate_uri(uri, ["C:/equipo/original"], Path(tmp_path).resolve())
