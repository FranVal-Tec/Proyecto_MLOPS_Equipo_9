# Proyecto_MLOPS_Equipo_9

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto del curso TC5044 MLOps (Equipo 9). Con los [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/)
(bicicleta pública de Guadalajara) se construye un sistema para **pronosticar la demanda de bicicletas
por zona y hora**, siguiendo prácticas de MLOps: estructura Cookiecutter Data Science, código modular,
pruebas, versionado de datos y registro en MLflow.

Periodo de datos: enero 2025 a septiembre 2026 (21 meses). La llegada de bicicletas eléctricas en
junio 2026 se usa como caso real de cambio en los datos (*data drift*).

## Primer setup

Requiere Python 3.12. Desde la carpeta raíz del repositorio:

```bash
# Linux, macOS o WSL
python3.12 -m venv .venv
source .venv/bin/activate
make requirements
```

```powershell
# Windows (PowerShell)
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Pipeline de datos

| Paso | Comando | Resultado |
|---|---|---|
| 1. Descargar | `make data` | CSV mensuales en `data/raw/` y `manifest_descarga.csv` con SHA-256 |
| 2. Limpiar y versionar | `python -m proyecto_mlops_equipo_9.cleaning --version version_00N` | Versión inmutable en `data/silver/version_00N/` |
| 3. Verificar | `make verify_data` | Relee los Parquet de `version_001` y comprueba huellas y balance de filas |
| 4. Registrar en MLflow | `python -m proyecto_mlops_equipo_9.register_data --paso preparar` (luego `registrar` y `verificar`) | Run con los datasets de entrada y salida |
| 5. Consultar MLflow | `python -m proyecto_mlops_equipo_9.open_mlflow` | Interfaz en http://127.0.0.1:5001 |

- `data/raw/` no se sube a Git: se descarga con `make data`. Para agregar un mes, añadir su enlace a
  `references/urls_mibici.csv` (`viajes,AAAA-MM,<url>`) y correr el script con el nuevo `--hasta`.
- `data/silver/version_001/` sí está en el repositorio, así que para verificar los datos o abrir MLflow
  no hace falta descargar nada.
- Una versión existente no se sobrescribe: para generar otra, usar una etiqueta nueva (`version_002`).
- La salida de datos utilizando feature engineering se encuentra en `data/processed`, aquí se sobreescribe cada salida.

**Limpieza.** `cleaning.py` aplica las reglas justificadas en los notebooks 1.1 y 1.2: elimina
desanclajes fallidos (misma estación, menos de 60 s), velocidades imposibles (más de 30 km/h en línea
recta), préstamos de más de 180 min, estaciones fuera del catálogo, duplicados e IDs conflictivos, y
convierte en faltantes las edades fuera de 16 a 90 años. Resultado: 7,894,097 viajes originales,
7,644,409 válidos (`viajes/`) y 249,688 excluidos con su motivo (`revision/`).

**Ingeniería de características** `feature.py` aplica las reglas que se encuentra en el notebook enfocado en el preprocesamiento
de los datos previo a realizar entrenamiento con el modelo. Se puede ejecutar sencillamente solamente ejecutando `mlflow ui` desde
la ruta raíz del proyecto y posteriormente utilizando una entry previamente creada `mlflow run . -e **features** --experiment-name "Feature Engineering Experiment" --env-manager local`

Detalle del versionado y de MLflow: `docs/versionado_datos.md`, `docs/registro_mlflow.md` y
`docs/abrir_mlflow.md`.

## Pruebas

```bash
python -m pytest tests/
```

## Notebooks

| Notebook | Contenido |
|---|---|
| `1.0-LV-exploracion-inicial-ebikes` | Primera exploración y efecto de la llegada de e-bikes |
| `1.1-LV-reglas-limpieza` | Contrato de datos, pruebas de diagnóstico y justificación de cada regla |
| `1.1-AG-limpieza-datos` | Exploración temporal y por estación (usa una versión anterior de la limpieza) |
| `1.2-integracion-limpieza-final` | Reglas finales importadas de `cleaning.py` y sensibilidad de umbrales |
| `1.0-EDA-integral-mibici` | EDA completo de demanda por zona y hora, y preprocesamiento (requiere `make data`) |
| `1.0-FV-feature-engineering` | Proceso basado en preprocesamiento del EDA final, analizando nuevas posibilidades y utilizando datoss desde ruta |

## Organización del proyecto

```
├── Makefile                      <- Comandos: make requirements, make data, make clean_data, make verify_data
├── README.md
├── data
│   ├── raw                       <- CSV originales (no se versionan en Git)
│   └── silver/version_001        <- Versión limpia: Parquet, manifest con SHA-256, metadatos y código usado
├── docs                          <- Guías de versionado de datos, MLflow y EDA
├── mlartifacts                   <- Artefactos del run de MLflow
├── mlflow_limpieza_integrada.db  <- Base de MLflow con el registro de version_001
├── models                        <- Modelos entrenados (fase siguiente)
├── notebooks                     <- Convención: número, iniciales y descripción
├── references                    <- urls_mibici.csv y plantilla del ML Canvas
├── reports
│   └── versionado/version_001    <- Evidencia del registro en MLflow
├── requirements.txt
├── tests                         <- Pruebas de limpieza y de open_mlflow
└── proyecto_mlops_equipo_9
    ├── config.py                 <- Rutas del proyecto
    ├── download.py               <- Descarga de datos crudos
    ├── dataset.py                <- Carga estándar de los CSV (encoding, columnas, tipos)
    ├── cleaning.py               <- Reglas de limpieza y versiones silver
    ├── register_data.py          <- Registro de una versión en MLflow
    ├── open_mlflow.py            <- Abre la base de MLflow en otra computadora
    ├── eda.py                    <- Funciones del EDA (reutiliza las reglas de cleaning.py)
    ├── features.py               <- Variables para el modelo, concluyendo sobre variables objetivo
    └── modeling                  <- Entrenamiento y predicción (fase siguiente)
```

## Flujo de trabajo del equipo

Cada cambio se hace en una rama (`feature/...` o `fix/...`) y entra a `main` por pull request revisado
por otro integrante, con **Create Merge Commit**.
