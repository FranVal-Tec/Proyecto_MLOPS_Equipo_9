"""Exporta el modelo registrado y los datos recientes para servirlos sin MLflow (p. ej., en Docker).

Uso:
    python -m proyecto_mlops_equipo_9.modeling.export              # con mlflow ui --port 5000
    python -m proyecto_mlops_equipo_9.modeling.export --verificar  # no requiere servidor

La exportación toma la versión con alias `campeon` del Model Registry (ver promote.py) y deja
en models/:
- el modelo en formato MLflow;
- `datos_recientes.parquet`: los últimos días del dataset de variables, que la API usa cuando
  no existe el Parquet completo de data/processed (como dentro del contenedor);
- `predicciones_referencia.parquet`: lo que el modelo predice para esos días;
- `modelo_servicio.json`: versión, run, huella SHA-256 del modelo, variables y versiones de las
  librerías con que se exportó.

La verificación vuelve a predecir los datos recientes con la copia local y compara contra la
referencia; sirve para comprobar que otro entorno (otra máquina o el contenedor) produce las
mismas salidas con el mismo binario.
"""

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from proyecto_mlops_equipo_9.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJ_ROOT
from proyecto_mlops_equipo_9.modeling.predict import predecir
from proyecto_mlops_equipo_9.modeling.promote import ALIAS_SERVICIO
from proyecto_mlops_equipo_9.modeling.train import ConfiguracionEntrenamiento
from proyecto_mlops_equipo_9.modeling.validation import OBJETIVOS, metricas

LLAVES = ["zona_id", "inicio_hora_objetivo"]
PREDICCIONES = ["salidas_pronosticadas", "llegadas_pronosticadas", "flujo_neto_pronosticado"]
ARCHIVO_METADATOS = "modelo_servicio.json"
ARCHIVO_DATOS = "datos_recientes.parquet"
ARCHIVO_REFERENCIA = "predicciones_referencia.parquet"
# Librerías que deciden cómo se carga y evalúa el modelo; deben coincidir con requirements.txt,
# porque la imagen de Docker instala esas versiones.
PAQUETES = ["mlflow", "scikit-learn", "skops", "numpy", "pandas"]
# El mismo binario debe dar las mismas salidas; solo se toleran diferencias de redondeo.
TOLERANCIA = 1e-6


def huella_directorio(ruta):
    """SHA-256 del directorio: ruta relativa y contenido de cada archivo, en orden fijo."""
    ruta = Path(ruta)
    digest = hashlib.sha256()
    for archivo in sorted(p for p in ruta.rglob("*") if p.is_file()):
        digest.update(archivo.relative_to(ruta).as_posix().encode("utf-8"))
        digest.update(hashlib.sha256(archivo.read_bytes()).digest())
    return digest.hexdigest()


def versiones_entorno(version_de=importlib.metadata.version):
    """Python y librerías que determinan cómo se carga y evalúa el modelo."""
    versiones = {"python": platform.python_version()}
    for paquete in PAQUETES:
        try:
            versiones[paquete] = version_de(paquete)
        except importlib.metadata.PackageNotFoundError:
            versiones[paquete] = None
    return versiones


def versiones_fijadas(texto):
    """`paquete==versión` de requirements.txt, con el nombre en minúsculas y sin extras."""
    fijadas = {}
    for linea in texto.splitlines():
        partes = re.match(r"^\s*([A-Za-z0-9._-]+)(\[[^\]]*\])?\s*==\s*([^\s;#]+)", linea)
        if partes:
            fijadas[partes[1].lower()] = partes[3]
    return fijadas


def diferencias_versiones(entorno, fijadas):
    """Librerías de PAQUETES cuya versión instalada no es la de requirements.txt."""
    return {
        paquete: {"instalada": entorno.get(paquete), "requirements": fijadas.get(paquete)}
        for paquete in PAQUETES
        if entorno.get(paquete) != fijadas.get(paquete)
    }


def seleccionar_recientes(df, variables, dias):
    """Últimos `dias` del dataset con las columnas que necesita el modelo, en orden estable."""
    fin = df.inicio_hora_objetivo.max()
    inicio = fin.floor("D") - pd.Timedelta(days=dias - 1)
    columnas = list(dict.fromkeys(LLAVES + list(variables) + OBJETIVOS))
    recientes = df.loc[df.inicio_hora_objetivo.ge(inicio), columnas]
    return recientes.sort_values(LLAVES).reset_index(drop=True)


def predecir_en_orden(modelo, filas):
    """Predicciones con el mismo código que el servicio, alineadas a las filas de entrada."""
    pronostico = predecir(modelo, filas)
    return filas[LLAVES].merge(pronostico, on=LLAVES, how="left", validate="one_to_one")


def metricas_disponibles(datos, referencia):
    """Métricas contra los objetivos reales en las horas que ya los tienen."""
    observados = datos[OBJETIVOS].notna().all(axis=1).to_numpy()
    if not observados.any():
        return None
    return metricas(datos.loc[observados, OBJETIVOS], referencia.loc[observados, PREDICCIONES[:2]])


def comparar_con_referencia(modelo, datos, referencia, tolerancia=TOLERANCIA):
    """Vuelve a predecir los datos y mide la mayor diferencia contra la referencia guardada."""
    nueva = predecir_en_orden(modelo, datos)
    # Se comparan como texto: el tipo de zona_id o la resolución de la fecha pueden cambiar
    # al pasar por Parquet sin que cambien los valores.
    mismas_filas = len(nueva) == len(referencia) and np.array_equal(
        nueva[LLAVES].astype(str).to_numpy(), referencia[LLAVES].astype(str).to_numpy()
    )
    if not mismas_filas:
        raise ValueError("Los datos recientes no corresponden a la referencia guardada.")
    diferencia = np.abs(nueva[PREDICCIONES].to_numpy() - referencia[PREDICCIONES].to_numpy())
    maxima = float(diferencia.max())
    return {
        "filas": len(referencia),
        "max_diferencia": maxima,
        "coincide": bool(maxima <= tolerancia),
        "metricas": metricas_disponibles(datos, nueva),
    }


def escribir_json(ruta, datos):
    Path(ruta).write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cargar_desde(ruta):
    """Carga un modelo en formato MLflow desde una carpeta local."""
    import mlflow.sklearn

    return mlflow.sklearn.load_model(str(ruta))


def cargar_modelo_local(modelos_dir=MODELS_DIR):
    """Modelo exportado y sus metadatos; no necesita servidor de MLflow."""
    modelos_dir = Path(modelos_dir)
    metadatos = json.loads((modelos_dir / ARCHIVO_METADATOS).read_text(encoding="utf-8"))
    return cargar_desde(modelos_dir / metadatos["directorio"]), metadatos


def descargar_version(nombre, alias, carpeta, tracking_uri):
    """Descarga en `carpeta` la versión a la que apunta el alias y devuelve su registro."""
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    cliente = MlflowClient()
    aliases = cliente.get_registered_model(nombre).aliases
    if alias not in aliases:
        raise LookupError(
            f"{nombre} no tiene el alias @{alias}; promueve primero una versión (make promote_model)."
        )
    version = cliente.get_model_version(nombre, aliases[alias])
    # Se descarga por número de versión: si el alias se mueve a mitad de la exportación,
    # los archivos y los metadatos siguen correspondiendo a la misma versión.
    uri = f"models:/{nombre}/{version.version}"
    ruta = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(carpeta)))
    if not (ruta / "MLmodel").is_file():
        raise FileNotFoundError(f"La descarga de {uri} no contiene un archivo MLmodel.")
    registro = {
        "nombre": nombre,
        "version": str(version.version),
        "alias": alias,
        "uri_registro": uri,
        "run_id": version.run_id,
        "etiquetas": dict(version.tags),
    }
    return ruta, registro


def exportar(
    dataset,
    modelos_dir,
    tracking_uri,
    dias=7,
    reemplazar=False,
    alias=ALIAS_SERVICIO,
    requirements=PROJ_ROOT / "requirements.txt",
    ignorar_versiones=False,
    config=None,
):
    """Copia el modelo registrado a `modelos_dir` con los datos recientes y su referencia."""
    config = config or ConfiguracionEntrenamiento()
    modelos_dir = Path(modelos_dir)
    destino = modelos_dir / config.nombre_registro
    if destino.exists() and not reemplazar:
        raise FileExistsError(f"{destino} ya existe; usa --reemplazar para sobrescribirlo.")
    entorno = versiones_entorno()
    diferencias = diferencias_versiones(
        entorno, versiones_fijadas(Path(requirements).read_text(encoding="utf-8"))
    )
    if diferencias and not ignorar_versiones:
        raise RuntimeError(
            "Las librerías instaladas no son las de requirements.txt, que es lo que instala "
            f"Docker: {json.dumps(diferencias)}. Instala las versiones fijadas "
            "(pip install -r requirements.txt) y vuelve a exportar."
        )

    with tempfile.TemporaryDirectory() as carpeta:
        ruta, registro = descargar_version(
            config.nombre_registro, alias, Path(carpeta), tracking_uri
        )
        # Se carga desde la copia local: así se comprueba que el servicio no necesita el servidor.
        modelo = cargar_desde(ruta)
        variables = list(modelo.feature_names_in_)
        columnas = list(dict.fromkeys(LLAVES + variables + OBJETIVOS))
        datos = seleccionar_recientes(pd.read_parquet(dataset, columns=columnas), variables, dias)
        referencia = predecir_en_orden(modelo, datos)
        # El modelo anterior solo se reemplaza cuando el nuevo ya cargó y predijo sin errores.
        if destino.exists():
            shutil.rmtree(destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ruta, destino)

    datos.to_parquet(modelos_dir / ARCHIVO_DATOS, index=False)
    referencia.to_parquet(modelos_dir / ARCHIVO_REFERENCIA, index=False)
    metadatos = {
        **registro,
        "directorio": destino.name,
        "sha256_modelo": huella_directorio(destino),
        "variables": variables,
        "objetivos": OBJETIVOS,
        "salidas": PREDICCIONES,
        "datos": {
            "archivo": ARCHIVO_DATOS,
            "referencia": ARCHIVO_REFERENCIA,
            "dataset_sha256": sha_dataset(dataset),
            "desde": str(datos.inicio_hora_objetivo.min()),
            "hasta": str(datos.inicio_hora_objetivo.max()),
            "filas": len(datos),
            "zonas": int(datos.zona_id.nunique()),
            "metricas": metricas_disponibles(datos, referencia),
        },
        "entorno": entorno,
        "versiones_distintas_a_requirements": diferencias or None,
        "exportado_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    escribir_json(modelos_dir / ARCHIVO_METADATOS, metadatos)
    return metadatos


def sha_dataset(dataset):
    """SHA-256 del Parquet de variables, tomado de sus metadatos si existen."""
    ruta = Path(dataset).with_suffix(".metadata.json")
    if not ruta.is_file():
        return None
    return json.loads(ruta.read_text(encoding="utf-8")).get("parquet_sha256")


def verificar(modelos_dir=MODELS_DIR, tolerancia=TOLERANCIA):
    """Compara el modelo exportado contra su referencia, sin servidor de MLflow."""
    modelos_dir = Path(modelos_dir)
    modelo, metadatos = cargar_modelo_local(modelos_dir)
    datos = pd.read_parquet(modelos_dir / metadatos["datos"]["archivo"])
    referencia = pd.read_parquet(modelos_dir / metadatos["datos"]["referencia"])
    huella = huella_directorio(modelos_dir / metadatos["directorio"])
    return {
        "modelo": metadatos["uri_registro"],
        "mismo_binario": huella == metadatos["sha256_modelo"],
        **comparar_con_referencia(modelo, datos, referencia, tolerancia),
        "metricas_referencia": metadatos["datos"]["metricas"],
        "entorno": versiones_entorno(),
        "entorno_referencia": metadatos["entorno"],
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROCESSED_DATA_DIR / "datos_feature_engineered_zona_hora.parquet",
    )
    parser.add_argument("--modelos-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--dias", type=int, default=7, help="Días finales del dataset a exportar")
    parser.add_argument("--alias", default=ALIAS_SERVICIO, help="Alias de la versión a exportar")
    parser.add_argument("--reemplazar", action="store_true", help="Sobrescribe el modelo local")
    parser.add_argument(
        "--ignorar-versiones",
        action="store_true",
        help="Exporta aunque las librerías instaladas no sean las de requirements.txt",
    )
    parser.add_argument("--verificar", action="store_true", help="Compara contra la referencia")
    args = parser.parse_args()

    if args.verificar:
        resultado = verificar(args.modelos_dir)
        print(json.dumps(resultado, ensure_ascii=False, indent=2))
        sys.exit(0 if resultado["coincide"] and resultado["mismo_binario"] else 1)

    try:
        metadatos = exportar(
            args.dataset,
            args.modelos_dir,
            args.tracking_uri,
            args.dias,
            args.reemplazar,
            args.alias,
            ignorar_versiones=args.ignorar_versiones,
        )
    except (FileExistsError, LookupError, RuntimeError) as error:
        sys.exit(f"No se exportó: {error}")
    resumen = {k: metadatos[k] for k in ("uri_registro", "run_id", "sha256_modelo")}
    print(json.dumps({**resumen, "datos": metadatos["datos"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
