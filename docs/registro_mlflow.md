# Registro asistido de la versión de datos

## 1. Preparar el experimento

Se creó el experimento **mibici-limpieza-integrada** en una base SQLite local dedicada.
Un experimento agrupa ejecuciones; este paso no creó un run.

## 2. Registrar la versión

Se creó el run **silver_version_001** y se asociaron 64 datasets con nombre, digest, fuente,
esquema, perfil y contexto. Los contextos distinguen 21 archivos raw, 21 silver, 21 de revisión y un
catálogo. La etiqueta `rol` distingue entrada y salida de la preparación.

Se guardaron reglas, parámetros, conteos, bitácora, correcciones, manifiestos, esquema, código y
notebook de referencia como artefactos. Los archivos de viajes continúan en data/raw y data/silver:
registrar metadatos no hace una copia automática de todos los datos en la base.

## 3. Verificar

Se recuperó el run desde MLflow y se comprobaron estado FINISHED, los 64 datasets, las huellas
completas, los perfiles con sus conteos, los contextos, las métricas y la presencia de artefactos.

- Estado del registro: **VERIFICADO**.
- Run ID: `9c5d722c76e24ef9a3a603955c04d2ae`.
- Versión de datos: `version_001`.
- Dataset ID: `7813aa70c6be348833409a731db19b72dae6001eaca9b1a20a5f0a82dc76c7a3`.
- MLflow utilizado: `3.17.2`.
- Originales: 7,894,097; válidos: 7,644,409; excluidos: 249,688.

## Qué consultar en la interfaz

Abrir `http://127.0.0.1:5001`, entrar al experimento y seleccionar silver_version_001.

1. Parameters: límites y opciones que definieron la limpieza.
2. Metrics: conteos antes/después y porcentaje excluido.
3. Datasets: fuentes, huellas, esquemas, perfiles y contextos.
4. Artifacts: documentación y evidencia de las transformaciones.
5. Tags: versión, dataset_id y huella del código ejecutado.

El recibo enlaza esta ejecución con la versión silver sin modificar sus archivos inmutables.
El commit posterior permitirá versionar el código y esta evidencia en Git; los Parquet y la base
local están excluidos de Git. No se hizo commit ni push durante esta preparación.
