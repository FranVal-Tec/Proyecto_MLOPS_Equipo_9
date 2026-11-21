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
| 6. Generar variables | `make features` o `make features_mlflow` | Genera el dataset por zona y hora en `data/processed/`, opcionalmente registrando el proceso en MLflow |


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

**Ingeniería de variables por zona y hora.** `features.py` genera observaciones
horarias sobre la cuadrícula de 1 km definida en el EDA. Las salidas y llegadas de
la siguiente hora se construyen por zona sobre un calendario continuo. Las
funciones se organizan en pipelines de Scikit-Learn y su importación no inicia
procesamientos ni registros de MLflow.

### Ejecución mediante Makefile

Para facilitar la reproducción de la ingeniería de variables, se incorporaron comandos al `Makefile` que permiten ejecutar el módulo `features.py` sin escribir manualmente sus argumentos.

Una vez disponibles los datos limpios y con el entorno Python 3.12 activado, ejecutar:

```bash
make features
```

Este comando genera el dataset de variables por zona y hora sin requerir una conexión activa con MLflow.

Para generar las variables y registrar el experimento en MLflow, iniciar previamente el servidor de tracking en el puerto 5000 y ejecutar:

```bash
make features_mlflow
```

Ambos comandos utilizan la configuración definida en `features.py` y generan por defecto el archivo `data/processed/datos_feature_engineered_zona_hora.parquet`.

**Validación:** se ejecutó la suite de pruebas automatizadas con Python 3.12.6, obteniendo 39 pruebas aprobadas. También se ejecutó la ingeniería de variables sobre los datos versionados de MiBici: se generaron 891,854 registros y 48 columnas. La ejecución se repitió en las mismas condiciones y produjo un archivo Parquet idéntico, verificado mediante SHA-256.


Desde la raíz del repositorio, con el entorno Python activado, abrir una terminal:

```bash
mlflow ui --port 5000
```

En una segunda terminal, ejecutar:

```bash
python -m proyecto_mlops_equipo_9.features --experiment-name mibici-feature-engineering
```

También se admite la entrada `features` de `MLproject`. Para que el comando y el
script utilicen el mismo servidor y experimento, configurar primero la conexión:

```bash
# Git Bash, Linux o macOS
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow run . -e features --experiment-name mibici-feature-engineering --env-manager local
```

```powershell
# PowerShell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow run . -e features --experiment-name mibici-feature-engineering --env-manager local
```

La salida es `data/processed/datos_feature_engineered_zona_hora.parquet`, acompañada
de un contrato `.metadata.json` con predictores, objetivos, columnas de auditoría,
parámetros y huellas de las fuentes. `--output-path` permite elegir otra ruta;
`--sin-mlflow` genera los archivos sin registrar una ejecución. No se entrenan modelos.

La política de cobertura asigna ceros después de la primera actividad zonal en
horas con señal de actividad global, excluyendo fallas conocidas. Los rezagos
requieren su hora exacta y las medias de 24 y 168 horas exigen al menos 75% de
observaciones. Esta hipótesis de cobertura y la latencia de aceptación de silver
requieren validación operativa antes del despliegue.

El notebook `1.0-FC-feature-engineering.ipynb` documenta la metodología y compara
la salida del script con una referencia independiente, usando un archivo separado.
La versión validada produce 806,774 filas y 43 columnas (29 predictores, 2 objetivos
y 12 campos de auditoría), con igualdad exacta de valores y tipos.

MLflow registra parámetros, métricas de preparación, perfil del dataset, contrato
y código fuente en `http://127.0.0.1:5000`. El Parquet y las nuevas bases y artefactos
locales de MLflow no forman parte del cambio en Git; cada integrante puede
reproducirlos con las fuentes versionadas. La consulta de la entrega de limpieza
del puerto 5001 corresponde a su registro histórico y conserva su flujo existente.

**Entrenamiento y selección de modelos.** El entrenamiento utiliza MLflow para registrar ejecuciones, métricas y modelos. Para mantener aislado este proceso del registro histórico de limpieza, se puede iniciar un servidor de tracking independiente en el puerto `5003`.

En una terminal, desde la raíz del repositorio:

```bash
mkdir -p .local/mlflow-entrenamiento

mlflow server \
  --backend-store-uri sqlite:///$PWD/.local/mlflow-entrenamiento/tracking.db \
  --artifacts-destination $PWD/.local/mlflow-entrenamiento/artifacts \
  --host 127.0.0.1 \
  --port 5003
```

En una segunda terminal, ejecutar el entrenamiento:

```bash
# Prueba rápida con un modelo y una configuración
python -m proyecto_mlops_equipo_9.modeling.train \
  --experiment-name mibici-entrenamiento \
  --tracking-uri http://127.0.0.1:5003 \
  --modelos hist_gradient_boosting \
  --n-iter 1 \
  --salida .local/mlflow-entrenamiento/reports/modelado

# Corrida completa: línea base, regresión Ridge, Random Forest y gradient boosting
python -m proyecto_mlops_equipo_9.modeling.train \
  --tracking-uri http://127.0.0.1:5003

# Pronóstico con el modelo registrado
python -m proyecto_mlops_equipo_9.modeling.predict \
  --tracking-uri http://127.0.0.1:5003
```

La selección utiliza seis ventanas con fecha de corte móvil (octubre de 2025 a marzo de 2026) y el MAE promedio. El ganador se evalúa en abril y mayo de 2026, se compara mes a mes en el escenario de drift (junio a septiembre de 2026) y se publica en el Model Registry con el alias `candidato`. Los resúmenes se guardan en la ruta indicada mediante `--salida`.

### Reproducibilidad del entrenamiento

Las dependencias del proyecto se encuentran fijadas mediante versiones específicas
en `requirements.txt`. La configuración de entrenamiento utiliza la semilla aleatoria
`9`, aplicada a los algoritmos y al proceso de búsqueda de hiperparámetros.

Para validar el registro del modelo se utilizó un servidor MLflow independiente
en el puerto `5003`, evitando modificar el registro histórico de limpieza y
manteniendo aislados los artefactos de entrenamiento.

La validación se realizó con `HistGradientBoostingRegressor` y una iteración de
búsqueda de hiperparámetros (`n_iter=1`). El entrenamiento utilizó seis ventanas
temporales, con evaluación en abril y mayo de 2026. El modelo obtenido alcanzó:

| Métrica | Resultado |
|---|---:|
| MAE | 2.0237 |
| WAPE | 26.70% |
| Mejora frente a la línea base | 28.67% |
| Criterio mínimo de mejora | 10% |
| Criterio de despliegue | Cumplido |

El modelo se registró en MLflow como `mibici-demanda-zona-hora`, versión `1`,
con el alias `candidato`.

**Validación de reproducibilidad entre entornos:** se exportó el artefacto
`model.skops` junto con un conjunto fijo de entradas y sus predicciones de
referencia. Posteriormente, el modelo se cargó y ejecutó desde un segundo entorno
Conda independiente, utilizando Python 3.12.6 y las mismas versiones de las
dependencias de ejecución.

La comparación confirmó que el archivo del modelo conserva el mismo SHA-256:

`d697a222823feea0f0127abb84960fd76aa396b974b3dc7fa521ceba078129e5`

Las predicciones obtenidas en ambos entornos fueron idénticas, con una diferencia
máxima de `0.0`.

Esta validación demuestra reproducibilidad entre dos entornos Conda con la misma
versión de Python, dependencias y arquitectura. No representa todavía una
validación entre sistemas operativos o arquitecturas diferentes.

Los archivos auxiliares de esta prueba y los artefactos locales de MLflow se
almacenan en `.local/`, directorio excluido del control de versiones.

Las figuras de comparación de modelos, drift y error por volumen se generan con
`python -m proyecto_mlops_equipo_9.plots` y se guardan en `reports/figures/`.

Detalle del versionado y de MLflow: `docs/versionado_datos.md`, `docs/registro_mlflow.md` y
`docs/abrir_mlflow.md`.

## Pruebas

```bash
python -m pytest tests/
```

## Modelo para el servicio

La API y el contenedor no consultan el servidor de MLflow: usan una copia del modelo registrado en
`models/`. La versión que se sirve se elige con alias del Model Registry:

| Alias | Lo asigna | Uso |
|---|---|---|
| `candidato` | `train.py`, en cada entrenamiento | Versión propuesta tras la prueba; no se sirve |
| `campeon` | `promover`, solo si el run tiene `cumple_criterio_despliegue = True` | Versión que se exporta y se sirve |

Con `mlflow ui --port 5000` activo y el Parquet de variables generado (`make features`):

```bash
make promote_model   # @candidato -> @campeon, si cumplió el criterio de despliegue
make export_model    # copia @campeon y los últimos 7 días de variables a models/
make verify_model    # vuelve a predecir esos días con la copia local (no requiere servidor)
```

| Artefacto | Ruta | Versión |
|---|---|---|
| Modelo en el Model Registry | `models:/mibici-demanda-zona-hora/2` (alias `campeon`) | 2 |
| Copia que sirve la API y va en la imagen | `models/mibici-demanda-zona-hora/` | 2 (ver `models/modelo_servicio.json`) |

- `models/modelo_servicio.json`: versión, run de origen, huella SHA-256 de los archivos del modelo,
  variables de entrada y versiones de Python y librerías con que se exportó.
- `models/datos_recientes.parquet`: últimos días del dataset de variables; la API los usa cuando no
  existe el Parquet completo de `data/processed/` (dentro del contenedor).
- `models/predicciones_referencia.parquet`: lo que predice el modelo para esos días.

La exportación se niega a correr si las versiones instaladas de mlflow, scikit-learn, skops, numpy o
pandas no son las de `requirements.txt`, porque esas son las que instala Docker. La verificación
compara el binario contra su huella y las predicciones contra la referencia, y termina con error si
no coinciden. Al promover una versión nueva, se exporta con `--reemplazar` y se actualiza la tabla.
Para revertir: `python -m proyecto_mlops_equipo_9.modeling.promover --version <n>`.

## FastAPI

```bash
make api          # o: uvicorn proyecto_mlops_equipo_9.fastAPI:app --port 8000
```

| Ruta | Descripción |
|---|---|
| `GET /` | Confirma que el servicio está funcionando |
| `POST /predict` | Pronóstico de salidas, llegadas y flujo neto por zona para un periodo |
| `GET /docs` | Swagger con el schema de entrada y salida y ejemplos (OpenAPI en `/openapi.json`) |

**Entrada** (cuerpo JSON, opcional): `desde` y `hasta` en ISO 8601. Sin cuerpo, se pronostica la
última hora disponible; con `desde` y sin `hasta`, solo esa hora; `hasta` no se incluye. Un periodo
inválido o sin datos responde 422.

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"desde": "2026-09-30T18:00:00", "hasta": "2026-09-30T21:00:00"}'
```

**Salida:** `status`, `metadata` (`fecha_prediccion`, `zonas_totales`, `modelo_utilizado`) y
`predicciones`, una por zona y hora con `salidas_pronosticadas`, `llegadas_pronosticadas`,
`flujo_neto_pronosticado` y `tendencia` (`positive`, `negative` o `balanced`).

El modelo y los datos los decide el servidor, no el cliente (`modeling/servicio.py`). Si existe
`models/modelo_servicio.json` se usa la exportación; si no, el Model Registry, como en desarrollo
con `mlflow ui`. Variables de entorno opcionales:

| Variable | Uso |
|---|---|
| `MODELO_DIR` | Carpeta de la exportación (por defecto `models/`) |
| `DATASET_SERVICIO` | Parquet de variables a usar en lugar del predeterminado |
| `MODELO_URI` | Modelo del Model Registry cuando no hay exportación (por defecto `@candidato`) |
| `MLFLOW_TRACKING_URI` | Servidor de MLflow cuando no hay exportación (por defecto `http://127.0.0.1:5000`) |

## Docker

La imagen (`python:3.12-slim`) instala las versiones fijas de `requirements.txt` sin las herramientas
de desarrollo, copia el paquete y `models/`, y levanta la API con uvicorn en el puerto 8000. Durante el
build corre `exportar --verificar`: si el modelo no es el mismo binario o no reproduce sus
predicciones de referencia en ese entorno, la imagen no se construye.

```bash
docker build -t ml-service:latest .
docker run -p 8000:8000 ml-service:latest    # http://localhost:8000/docs
```

Publicación en DockerHub con tags versionados (`make docker_build VERSION=1.0.0` y
`make docker_push DOCKERHUB_USUARIO=<usuario> VERSION=1.0.0` hacen lo mismo):

```bash
docker build --build-arg VERSION=1.0.0 -t ml-service:latest -t ml-service:1.0.0 .
docker tag ml-service:1.0.0 <usuario>/ml-service:1.0.0
docker tag ml-service:1.0.0 <usuario>/ml-service:latest
docker push <usuario>/ml-service:1.0.0
docker push <usuario>/ml-service:latest
```

| Tag | Contenido |
|---|---|
| `1.0.0` | API 1.0.0 con el modelo `mibici-demanda-zona-hora` versión 2 |
| `latest` | Último tag publicado |

Cada cambio en la API o en el modelo servido se publica con un tag nuevo (`1.0.1`, `1.1.0`, …) y se
agrega a esta tabla; `latest` se mueve al más reciente.

## Notebooks

| Notebook | Contenido |
|---|---|
| `1.0-LV-exploracion-inicial-ebikes` | Primera exploración y efecto de la llegada de e-bikes |
| `1.1-LV-reglas-limpieza` | Contrato de datos, pruebas de diagnóstico y justificación de cada regla |
| `1.1-AG-limpieza-datos` | Exploración temporal y por estación (usa una versión anterior de la limpieza) |
| `1.2-integracion-limpieza-final` | Reglas finales importadas de `cleaning.py` y sensibilidad de umbrales |
| `1.0-EDA-integral-mibici` | EDA completo de demanda por zona y hora, y preprocesamiento (requiere `make data`) |
| `1.0-FV-feature-engineering` | Exploración inicial de ingeniería de variables |
| `1.0-FC-feature-engineering` | Variables por zona y hora, trazabilidad en MLflow y validación de `features.py` |

## Organización del proyecto

```
├── Dockerfile                    <- Imagen del servicio (FastAPI + modelo exportado)
├── Makefile  <- Instalación, descarga, limpieza, verificación, features, modelo, API y Docker
├── README.md
├── data
│   ├── raw                       <- CSV originales (no se versionan en Git)
│   └── silver/version_001        <- Versión limpia: Parquet, manifest con SHA-256, metadatos y código usado
├── docs                          <- Guías de versionado de datos, MLflow y EDA
├── mlartifacts                   <- Artefactos del run de MLflow
├── mlflow_limpieza_integrada.db  <- Base de MLflow con el registro de version_001
├── models                        <- Modelo exportado para el servicio, datos recientes y referencia
├── notebooks                     <- Convención: número, iniciales y descripción
├── references                    <- urls_mibici.csv y plantilla del ML Canvas
├── reports
│   └── versionado/version_001    <- Evidencia del registro en MLflow
├── requirements.txt
├── tests                         <- Pruebas de limpieza, MLflow, variables, modelado, exportación y API
└── proyecto_mlops_equipo_9
    ├── config.py                 <- Rutas del proyecto
    ├── download.py               <- Descarga de datos crudos
    ├── dataset.py                <- Carga estándar de los CSV (encoding, columnas, tipos)
    ├── cleaning.py               <- Reglas de limpieza y versiones silver
    ├── register_data.py          <- Registro de una versión en MLflow
    ├── open_mlflow.py            <- Abre la base de MLflow en otra computadora
    ├── eda.py                    <- Funciones del EDA (reutiliza las reglas de cleaning.py)
    ├── features.py               <- Pipelines de variables por zona/hora y registro explícito en MLflow
    ├── fastAPI.py                <- API: rutas, schema de entrada/salida y manejo de errores
    └── modeling                  <- Entrenamiento, validación, predicción, promoción, exportación y servicio
```

## Flujo de trabajo del equipo

Cada cambio se hace en una rama (`feature/...` o `fix/...`) y entra a `main` por pull request revisado
por otro integrante, con **Create Merge Commit**.
