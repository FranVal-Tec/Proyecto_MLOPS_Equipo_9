# Consultar el experimento en otra computadora

Desde la raíz del proyecto, con el entorno de Python y las dependencias del proyecto instaladas:

```powershell
python -m proyecto_mlops_equipo_9.open_mlflow
```

Abre http://127.0.0.1:5001. Si el puerto está ocupado:

```powershell
python -m proyecto_mlops_equipo_9.open_mlflow --port 5002
```

El comando detecta la carpeta actual, obtiene una copia consistente de
`mlflow_limpieza_integrada.db` mediante el respaldo de SQLite y adapta en esa copia las
rutas de experimentos, runs, fuentes de datasets y la etiqueta de ubicación silver.
Conserva los identificadores, estados, digests, perfiles y métricas del experimento.
La utilidad está preparada para el esquema de MLflow usado por este proyecto.

La copia operativa queda en `.local/mlflow/tracking.db`, excluida de Git. La base
compartida, los Parquet y los artefactos originales no se modifican.
`portabilidad.json` registra la ubicación, el hash de la base compartida y las verificaciones.
Los documentos históricos dentro de artefactos conservan sus rutas originales como evidencia.

Se necesitan la base compartida, `mlartifacts/` y los archivos de `data/silver/`.
Los raw pueden estar ausentes: se conservan sus metadatos, pero para volver a cargar
sus archivos o repetir la limpieza hay que descargarlos. Esto no impide consultar
el run, sus métricas, los artefactos y los datos silver.

Para comprobar las referencias sin iniciar el servidor:

```powershell
python -m proyecto_mlops_equipo_9.open_mlflow --preparar
```

Si recibes otra versión de la base compartida, cierra el servidor y ejecuta:

```powershell
python -m proyecto_mlops_equipo_9.open_mlflow --actualizar
```

La copia local anterior se respalda en `.local/mlflow/backups/` antes de regenerarla.
Los cambios que realices en el servidor local quedan en esa copia; el comando no
los publica ni los incorpora automáticamente a la base compartida.
