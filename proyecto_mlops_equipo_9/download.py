"""Descarga los datos crudos de MiBici: viajes mensuales y catálogo de estaciones.

Uso (desde la raíz del repo, con el entorno activado):

    python -m proyecto_mlops_equipo_9.download --desde 2025-01 --hasta 2026-09

Las URLs no siguen un patrón, por eso viven en references/urls_mibici.csv. El script descarga
solo lo que falta en data/raw y al final escribe data/raw/manifest_descarga.csv con el tamaño
y la huella SHA-256 de cada archivo: así se puede comprobar que dos personas tienen exactamente
los mismos datos, y más adelante registrar esa versión en MLflow.
"""

import csv
import hashlib
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import typer
from loguru import logger

from proyecto_mlops_equipo_9.config import PROJ_ROOT, RAW_DATA_DIR

URLS_CSV = PROJ_ROOT / "references" / "urls_mibici.csv"
MANIFEST = RAW_DATA_DIR / "manifest_descarga.csv"

# El servidor de MiBici responde 406 (Not Acceptable) a clientes que no parecen un navegador.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,application/octet-stream,*/*;q=0.8",
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.8",
    "Referer": "https://mibici.net/es/datos-abiertos/",
}

app = typer.Typer()


def leer_urls(ruta: Path) -> list[dict]:
    """Lee el CSV de URLs (columnas: tipo, periodo, url)."""
    with open(ruta, newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def ruta_destino(fila: dict) -> Path:
    """Nombre estándar en data/raw, sin importar cómo se llame el archivo en el servidor."""
    periodo = fila["periodo"].replace("-", "_")
    if fila["tipo"] == "viajes":
        return RAW_DATA_DIR / "viajes" / f"datos_abiertos_{periodo}.csv"
    return RAW_DATA_DIR / "estaciones" / f"nomenclatura_{periodo}.csv"


def meses_entre(desde: str, hasta: str) -> list[str]:
    """Lista de meses 'AAAA-MM' entre dos meses, ambos incluidos."""
    anio, mes = map(int, desde.split("-"))
    meses = []
    while f"{anio:04d}-{mes:02d}" <= hasta:
        meses.append(f"{anio:04d}-{mes:02d}")
        anio, mes = (anio + 1, 1) if mes == 12 else (anio, mes + 1)
    return meses


def seleccionar(filas: list[dict], desde: str, hasta: str) -> list[dict]:
    """Viajes dentro del rango más el catálogo de estaciones más reciente."""
    pedidos = set(meses_entre(desde, hasta))
    viajes = [f for f in filas if f["tipo"] == "viajes" and f["periodo"] in pedidos]

    sin_url = sorted(pedidos - {f["periodo"] for f in viajes})
    if sin_url:
        logger.warning(f"Meses sin URL en {URLS_CSV.name}: {', '.join(sin_url)}")

    catalogos = [f for f in filas if f["tipo"] == "estaciones"]
    if not catalogos:
        return viajes
    return viajes + [max(catalogos, key=lambda f: f["periodo"])]


def validar(ruta: Path, tipo: str) -> None:
    """Revisa que lo descargado sea un CSV y no una página de error del servidor."""
    if ruta.stat().st_size == 0:
        raise ValueError("el archivo está vacío")
    with open(ruta, "rb") as archivo:
        inicio = archivo.read(1024).decode("latin-1").lower()
    if "<html" in inicio or "<!doctype" in inicio:
        raise ValueError("el servidor devolvió una página HTML, no un CSV")
    encabezado = inicio.splitlines()[0]
    esperado = "viaje" if tipo == "viajes" else "latitude"
    if esperado not in encabezado:
        raise ValueError(f"encabezado inesperado: {encabezado[:80]}")


def descargar(url: str, ruta: Path, tipo: str, intentos: int = 3) -> None:
    """Descarga a un archivo temporal y solo lo renombra si llegó completo y es válido."""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(".part")
    for intento in range(1, intentos + 1):
        try:
            peticion = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(peticion, timeout=120) as respuesta, open(temporal, "wb") as archivo:
                shutil.copyfileobj(respuesta, archivo)
            validar(temporal, tipo)
            temporal.replace(ruta)
            return
        except urllib.error.HTTPError as error:
            temporal.unlink(missing_ok=True)
            if error.code < 500 or intento == intentos:  # un 4xx no se arregla reintentando
                raise
        except (urllib.error.URLError, TimeoutError):
            temporal.unlink(missing_ok=True)
            if intento == intentos:
                raise
        except Exception:
            temporal.unlink(missing_ok=True)
            raise
        logger.warning(f"Reintentando {ruta.name} ({intento}/{intentos - 1})...")
        time.sleep(2 * intento)


def sha256(ruta: Path) -> str:
    huella = hashlib.sha256()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            huella.update(bloque)
    return huella.hexdigest()


def escribir_manifest(filas: list[dict]) -> int:
    """Registra todos los archivos de data/raw que vienen de urls_mibici.csv."""
    registros = []
    for fila in filas:
        ruta = ruta_destino(fila)
        if ruta.exists():
            modificado = datetime.fromtimestamp(ruta.stat().st_mtime, tz=timezone.utc)
            registros.append({
                "archivo": ruta.relative_to(RAW_DATA_DIR).as_posix(),
                "tipo": fila["tipo"],
                "periodo": fila["periodo"],
                "url": fila["url"],
                "bytes": ruta.stat().st_size,
                "sha256": sha256(ruta),
                "descargado_utc": modificado.isoformat(timespec="seconds"),
            })
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as archivo:
        columnas = ["archivo", "tipo", "periodo", "url", "bytes", "sha256", "descargado_utc"]
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(registros)
    return len(registros)


def validar_periodo(valor: str) -> str:
    try:
        datetime.strptime(valor, "%Y-%m")
    except ValueError:
        raise typer.BadParameter("usa el formato AAAA-MM, por ejemplo 2025-01") from None
    return valor


@app.command()
def main(
    desde: str = typer.Option("2025-01", callback=validar_periodo, help="Primer mes (AAAA-MM)."),
    hasta: str = typer.Option("2026-09", callback=validar_periodo, help="Último mes (AAAA-MM)."),
    forzar: bool = typer.Option(False, help="Vuelve a descargar aunque el archivo ya exista."),
    urls_csv: Path = typer.Option(URLS_CSV, help="CSV con las URLs (tipo, periodo, url)."),
):
    """Descarga a data/raw los viajes de MiBici del rango indicado y el catálogo de estaciones."""
    if desde > hasta:
        raise typer.BadParameter("--desde no puede ser posterior a --hasta")

    filas = leer_urls(urls_csv)
    seleccion = seleccionar(filas, desde, hasta)
    logger.info(f"{len(seleccion)} archivos en el rango {desde} a {hasta}")

    fallidos = []
    for fila in seleccion:
        ruta = ruta_destino(fila)
        if ruta.exists() and not forzar:
            logger.info(f"Ya existe: {ruta.name}")
            continue
        logger.info(f"Descargando {ruta.name} ...")
        try:
            descargar(fila["url"], ruta, fila["tipo"])
            logger.success(f"{ruta.name}: {ruta.stat().st_size / 1e6:.1f} MB")
        except Exception as error:
            logger.error(f"No se pudo descargar {ruta.name}: {error}")
            fallidos.append(fila)

    total = escribir_manifest(filas)
    logger.info(f"Manifiesto actualizado: {MANIFEST.relative_to(PROJ_ROOT)} ({total} archivos)")

    if fallidos:
        logger.error("Descarga estos archivos desde el navegador y guárdalos con este nombre:")
        for fila in fallidos:
            logger.error(f"  {ruta_destino(fila).relative_to(PROJ_ROOT)}  <-  {fila['url']}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
