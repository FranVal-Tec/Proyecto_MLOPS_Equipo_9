# Layout de viajes silver

Ambas carpetas usan el mismo esquema: viajes/ contiene filas válidas y revision/ las excluidas.
Las columnas de negocio se mantienen; las de control identifican el proceso y la versión.

| Campo | Tipo Arrow / Parquet | Definición |
|---|---|---|
| `viaje_id` | `int64` | Identificador del viaje. |
| `usuario_id` | `int64` | Identificador del usuario de MiBici. |
| `genero` | `dictionary<values=string, indices=int8, ordered=0>` | Categoría M/F o faltante tras corrección. |
| `anio_nacimiento` | `double` | Año de nacimiento numérico o faltante tras corrección. |
| `inicio` | `timestamp[us]` | Fecha y hora original de inicio; zona del servicio no declarada en el CSV. |
| `fin` | `timestamp[us]` | Fecha y hora original de fin; no se convierte automáticamente a UTC. |
| `origen_id` | `int64` | Estación de origen. |
| `destino_id` | `int64` | Estación de destino. |
| `duracion_min` | `double` | Diferencia entre fin e inicio en minutos. |
| `fila_csv` | `int64` | Número lógico de registro en el CSV contando el encabezado como 1. |
| `mes_archivo` | `large_string` | Periodo del archivo original, AAAA_MM. |
| `flag_duplicado_exacto` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `flag_id_conflictivo` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `flag_id_entre_archivos` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `flag_id_esencial_nulo` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `flag_fecha_invalida` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `flag_fuera_mes` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `flag_duracion_no_positiva` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `viaje_corto` | `bool` | Duración no negativa y menor al umbral configurado. |
| `corto_distinta_estacion` | `bool` | Viaje corto con distinto origen y destino. |
| `flag_misma_estacion_corta` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `duracion_larga` | `bool` | Duración superior al límite configurado. |
| `flag_estacion_sin_catalogo` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `flag_sin_coordenadas` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `velocidad_kmh` | `double` | Distancia Haversine / duración; faltante si no es calculable. |
| `flag_velocidad_imposible` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `anio_nacimiento_original` | `double` | Valor declarado antes de corregir la demografía. |
| `flag_edad_imposible` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `flag_nacimiento_no_numerico` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `edad` | `double` | Año del viaje menos nacimiento; faltante si está fuera de política. |
| `genero_original` | `dictionary<values=string, indices=int8, ordered=0>` | Categoría declarada antes de corregir la demografía. |
| `flag_genero_invalido` | `bool` | Indicador de calidad calculado antes de filtrar. |
| `motivo_exclusion` | `large_string` | Primer motivo de rechazo secuencial; faltante para viajes válidos. |
| `es_valido` | `bool` | Resultado de la política; True en viajes/ y False en revision/. |
| `usuario` | `large_string` | Cuenta que ejecutó la limpieza; no es el usuario de MiBici (usuario_id). |
| `insrt_dt` | `timestamp[us, tz=UTC]` | Fecha de materialización de la versión en UTC, con zona horaria. |
| `updt_dt` | `timestamp[us, tz=UTC]` | Fecha de última escritura UTC; igual a insrt_dt en una versión inmutable. |
| `version_datos` | `large_string` | Etiqueta de la entrega, por ejemplo version_001. |
| `version_limpieza` | `large_string` | Versión semántica de la política de limpieza. |
| `dataset_id` | `large_string` | SHA-256 de insumos, código, política y entorno; no incluye fecha ni usuario. |
| `lote_id` | `large_string` | UUID de la ejecución que materializó los archivos. |
| `catalogo_version` | `large_string` | Nombre del catálogo de estaciones fijado para esta versión. |
| `catalogo_sha256` | `large_string` | SHA-256 del catálogo utilizado. |
| `codigo_limpieza_sha256` | `large_string` | SHA-256 del archivo cleaning.py ejecutado. |
| `archivo_origen` | `large_string` | Ruta relativa al proyecto del CSV original. |
| `sha256_origen` | `large_string` | SHA-256 de todos los bytes del CSV original. |

Los tiempos insrt_dt y updt_dt son UTC del proceso; inicio y fin mantienen el tiempo de negocio.
No se actualizan archivos dentro de una versión completa: los cambios producen otra versión.
Las banderas originales pueden solaparse; motivo_exclusion corresponde al primer paso que rechazó la fila.
