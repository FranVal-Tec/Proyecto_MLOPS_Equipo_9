"""Promueve una versión del Model Registry al alias que usa el servicio.

Uso:
    python -m proyecto_mlops_equipo_9.modeling.promote              # la versión de @candidato
    python -m proyecto_mlops_equipo_9.modeling.promote --version 3  # una versión concreta

`train.py` asigna `candidato` en cada entrenamiento; `campeon` solo se mueve con este comando
y solo si el run de prueba de esa versión cumplió el criterio de despliegue (mejora mínima
contra la línea base). La promoción queda en etiquetas de la versión, incluida la versión
anterior, para poder revertir promoviendo esa versión con `--version`.
"""

import argparse
from datetime import UTC, datetime
import getpass
import json
import sys

from proyecto_mlops_equipo_9.modeling.train import ConfiguracionEntrenamiento

ALIAS_SERVICIO = "campeon"
ETIQUETA_CRITERIO = "cumple_criterio_despliegue"


def responsable_actual():
    try:
        return getpass.getuser()
    except (KeyError, OSError):  # algunos contenedores no definen un usuario
        return "desconocido"


def validar_criterio(version, etiquetas_run):
    """Solo se promueve una versión cuyo run de prueba cumplió el criterio de despliegue."""
    valor = etiquetas_run.get(ETIQUETA_CRITERIO)
    if valor != "True":
        raise ValueError(
            f"La versión {version} no cumple el criterio: {ETIQUETA_CRITERIO}={valor!r} "
            "en su run de prueba."
        )


def promover(
    nombre,
    version=None,
    alias=ALIAS_SERVICIO,
    alias_origen="candidato",
    responsable=None,
    cliente=None,
):
    """Mueve `alias` a `version` (por defecto, la de `alias_origen`) y registra el cambio."""
    if cliente is None:
        from mlflow.tracking import MlflowClient

        cliente = MlflowClient()
    aliases = {k: str(v) for k, v in cliente.get_registered_model(nombre).aliases.items()}
    if version is None:
        if alias_origen not in aliases:
            raise LookupError(f"{nombre} no tiene el alias @{alias_origen}.")
        version = aliases[alias_origen]
    version = str(version)
    anterior = aliases.get(alias)
    resultado = {"modelo": nombre, "alias": alias, "version": version, "anterior": anterior}
    if anterior == version:
        return {**resultado, "cambio": False}

    modelo_version = cliente.get_model_version(nombre, version)
    validar_criterio(version, cliente.get_run(modelo_version.run_id).data.tags)
    cliente.set_registered_model_alias(nombre, alias, version)
    etiquetas = {
        f"{alias}_desde_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        f"{alias}_promovido_por": responsable or responsable_actual(),
        f"{alias}_version_anterior": anterior or "ninguna",
    }
    for clave, valor in etiquetas.items():
        cliente.set_model_version_tag(nombre, version, clave, valor)
    return {**resultado, "cambio": True, "etiquetas": etiquetas}


def main():
    config = ConfiguracionEntrenamiento()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--version", default=None, help="Por defecto, la versión de @candidato")
    parser.add_argument("--alias", default=ALIAS_SERVICIO)
    parser.add_argument("--responsable", default=None, help="Por defecto, el usuario del sistema")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    args = parser.parse_args()

    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    try:
        resultado = promover(
            config.nombre_registro,
            args.version,
            args.alias,
            config.alias_registro,
            args.responsable,
        )
    except (LookupError, ValueError) as error:
        sys.exit(f"No se promovió: {error}")
    print(json.dumps(resultado, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
