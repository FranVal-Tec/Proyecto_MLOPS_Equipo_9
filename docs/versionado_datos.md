# Preparación de datos silver y versionado

Los CSV originales se conservan en `data/raw/`. La limpieza produce una entrega inmutable en
`data/silver/version_001/`. El notebook final explica las reglas y las importa del mismo módulo
`proyecto_mlops_equipo_9/cleaning.py` que genera la entrega.

## Ejecutar desde la raíz del proyecto

Con el entorno del proyecto y `requirements.txt` instalados:

```powershell
python -m pytest tests
python -m proyecto_mlops_equipo_9.cleaning --version version_001
python -m proyecto_mlops_equipo_9.cleaning --verificar data/silver/version_001
```

La versión existente no se sobrescribe. Para una entrega futura se debe elegir otra etiqueta
(por ejemplo `version_002`) y documentar los cambios. No es necesario crear una versión artificial
solo para repetir pruebas. El módulo no descarga datos ni registra runs de MLflow.

## Estructura

```text
data/silver/version_001/
  viajes/                      # 21 Parquet de viajes válidos
  revision/                    # 21 Parquet de registros excluidos y sus motivos
  fuentes/
    codigo/                    # copia exacta de módulos utilizados
    notebook_limpieza.ipynb     # referencia de metodología y diagnósticos
    nomenclatura_2026_09.csv    # catálogo fijado
  metadatos/
    resumen_mensual.csv
    bitacora.csv
    correcciones.csv
    indicadores_calidad.csv
    manifest_entradas.csv
    layout.json
    entorno.json
  politica.json
  manifest.json                # entradas, salidas, SHA-256, código, política y control
  layout.md                    # definición de todas las columnas y tipos
  verificacion.json
  _SUCCESS.json                # publicación completa y huellas de manifiesto y verificación
  README.md
```

Las dos carpetas de Parquet forman una partición del universo original: válidos + excluidos =
originales. No se duplica el dataset completo como una tercera vista anotada. Ambas conservan banderas;
`motivo_exclusion` es nulo para válidos y contiene el primer rechazo para excluidos. Los valores
demográficos declarados permanecen en columnas con sufijo `_original`.

## Campos de control

| Campo | Significado |
|---|---|
| `usuario` | Cuenta del operador del proceso. Se obtiene del sistema o se fija con `--usuario`. |
| `usuario_id` | Identificador original del usuario de MiBici; no es el operador. |
| `insrt_dt` | Momento de materialización en UTC, con zona horaria. |
| `updt_dt` | Última escritura en UTC; igual a `insrt_dt` porque la versión es inmutable. |
| `version_datos` | Etiqueta de entrega, como `version_001`. |
| `version_limpieza` | Versión de la política de reglas. |
| `dataset_id` | Huella de insumos, código de limpieza/carga, política y entorno. |
| `lote_id` | UUID de esta ejecución; no es un run_id de MLflow. |
| `archivo_origen`, `fila_csv` | Archivo y número lógico de registro en la entrada. |
| `sha256_origen` | Huella completa del archivo de entrada. |
| `catalogo_version`, `catalogo_sha256` | Catálogo fijado y su huella. |
| `codigo_limpieza_sha256` | Huella del código ejecutado. |

`inicio` y `fin` conservan las fechas del negocio tal como las interpreta el cargador. El proceso no
les asigna una zona horaria que la fuente no declara. Solo los campos temporales de auditoría son UTC.

`dataset_id` excluye usuario, fecha y lote para distinguir el contenido lógico de una ejecución concreta.
Los SHA-256 de los Parquet identifican sus bytes exactos e incluyen los campos de auditoría.
El commit registrado es el commit base: los snapshots y sus huellas identifican el código ejecutado
cuando todavía existen cambios locales sin commit.

## Controles de publicación

1. Identificar todos los insumos con SHA-256 y comprobar su contrato.
2. Detectar IDs compartidos entre archivos y conflictos dentro de cada mes.
3. Aplicar las reglas del notebook final y validar el balance por mes.
4. Escribir cada Parquet y releerlo para comparar valores y tipos.
5. Volver a comprobar insumos y código; verificar hashes y controles de todas las salidas.
6. Publicar la carpeta definitiva con `_SUCCESS.json` al finalizar correctamente.

Un fallo deja una carpeta oculta `.version_001.en_proceso_<lote>` con `_FAILED.json`, sin publicar
la versión final. No debe considerarse un dataset completo. Un archivo de bloqueo evita publicaciones
concurrentes de la misma etiqueta.

## Registro de MLflow en sesión asistida

Crear la versión silver y registrarla en MLflow son pasos distintos. Esta etapa no modifica la base de
MLflow, no inicia runs y no ejecuta el registro opcional del notebook. El manifiesto deja constancia
de que el registro estaba pendiente al crear la versión.

El registro asistido revisa: ubicación de la base y de artefactos, identificación de datasets
de entrada y salida, metadatos con `mlflow.log_input`, política, métricas y archivos de trazabilidad.
La relación entre `dataset_id`, `version_001` y el futuro `run_id` se guardará fuera de la versión
inmutable. No se considerará registrado hasta comprobarlo en MLflow.

La carpeta `data/` ya está excluida de Git. El código, notebook, esta documentación y la evidencia
pequeña en `reports/versionado/version_001/` pueden revisarse y versionarse en Git. Las copias de
datos deben conservarse en su almacenamiento; una huella no sustituye los archivos.

## Registro actual de version_001

- Experimento: `mibici-limpieza-integrada`
- Run: `silver_version_001`
- Run ID: `9c5d722c76e24ef9a3a603955c04d2ae`
- Estado: `VERIFICADO`; run `FINISHED`
- Datasets asociados: 64
- Contextos: 21 raw, 21 silver, 21 revisión y 1 catálogo.
- Base local: `mlflow_limpieza_integrada.db`.
- Archivos de artefactos: `mlartifacts/limpieza_integrada/`.
- Recibo: `reports/versionado/version_001/registro_mlflow.json`.

La versión silver y su manifiesto permanecen intactos. El campo `mlflow_al_crear` documenta el estado histórico al generar los Parquet; el recibo externo contiene el estado actual del registro.

Comandos separados, desde el entorno del proyecto con requirements.txt instalado:

```powershell
python -m proyecto_mlops_equipo_9.register_data --paso preparar
python -m proyecto_mlops_equipo_9.register_data --paso registrar
python -m proyecto_mlops_equipo_9.register_data --paso verificar
python -m mlflow server --host 127.0.0.1 --port 5001 --backend-store-uri sqlite:///mlflow_limpieza_integrada.db
```

Si el recibo ya corresponde a un run terminado de esta versión, registrar no crea otro run.
Las entradas se asocian mediante mlflow.log_input; el contexto y el tag rol distinguen insumos de salidas.
El digest compacto usa los primeros 32 caracteres del SHA-256, y la huella completa se conserva en los tags de cada dataset y en el manifiesto.
Para cambiar reglas productivas, actualizar POLITICA_DEFAULT en cleaning.py y producir otra etiqueta de versión.
