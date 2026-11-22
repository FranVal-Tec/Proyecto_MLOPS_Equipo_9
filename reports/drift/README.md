# Monitoreo de drift

Evidencia mensual del modelo que sirve la API. Cada mes evaluado tiene su carpeta en
`v<versión del modelo>/monthly/<AAAA-MM>/`, generada por el notebook
`notebooks/2.0-FC-monitoreo-drift-mibici.ipynb` con `proyecto_mlops_equipo_9/monitoring.py`.
El modelo no se reentrena: se evalúa la misma exportación que usa la API (`models/`), después de
comprobar su huella SHA-256 y que reproduce las predicciones de referencia del servicio.

## Contenido de cada carpeta

| Archivo | Contenido |
|---|---|
| `metricas.csv` | MAE (total, salidas, llegadas y flujo neto) y WAPE del modelo y de la línea base, por escenario y segmento |
| `drift.csv` | Cambio relativo de la media de salidas, llegadas y sus rezagos de 23 h contra la referencia |
| `alertas.csv` | Cada regla con su valor, umbral, si se activó y la acción propuesta |
| `evidencia.json` | Modelo evaluado, huellas del CSV y del Parquet de variables, descartes de limpieza, umbrales y run de MLflow |
| `monitoreo.png` | MAE por escenario y cambio en las entradas |

**Escenarios**

- `referencia_abril_mayo`: periodo de prueba del modelo (abril y mayo de 2026).
- `mes_real`: el mes evaluado, con datos abiertos publicados después del entrenamiento.
- `sintetico_20` y `sintetico_40`: la referencia con un sobreconteo de 20 % y 40 % en las salidas
  y sus variables derivadas (rezagos y medias), con las etiquetas originales. Simulan una falla
  en el conteo de la fuente.

**Segmentos**

- `total`: todas las zonas.
- `zonas_referencia`: zonas presentes en la referencia; es la base de las alertas.
- `otras_zonas`: zonas nuevas del mes.

El drift de entrada se mide en estratos comunes de zona, hora y día de la semana, con el mismo peso
para cada estrato.

## Alertas

Los umbrales son piloto (`Umbrales` en `monitoring.py`).

| Regla | Se activa si | Acción propuesta |
|---|---|---|
| `entrada_<señal>` | el cambio relativo de la señal es de 25 % o más | Revisar fuente, cobertura y conteos antes de ajustar el modelo |
| `aumento_mae` | el MAE sube 20 % o más respecto a la referencia | Revisar error por zona y hora; si persiste con datos válidos, evaluar reentrenamiento |
| `ventaja_naive` | el modelo tiene menos de 10 % menos error que la línea base | Comparar con la línea base y evaluar un candidato con datos recientes |

## Evaluar un mes nuevo

Requisitos: el entorno de `requirements.txt`, la versión silver `data/silver/version_001` y la
exportación del modelo en `models/` (las dos están en el repositorio).

```bash
make monitor                       # octubre de 2026; descarga el CSV de MiBici
make monitor MES=2026-11 CSV_MES=data/raw/datos_abiertos_2026_11.csv
```

`make monitor` ejecuta el notebook sin abrirlo y guarda la copia ejecutada y los resultados en
`data/processed/monitoreo/`, que no se versiona.

- **Meses después de octubre:** la versión silver termina en septiembre de 2026, así que deben
  pasarse también los CSV intermedios. Agrégalos en `CSV_PREVIOS`, en la primera celda de código
  del notebook; por ejemplo, para noviembre: `{"2026-10": Path("data/raw/datos_abiertos_2026_10.csv")}`.
  Los CSV se descargan del [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/).
- **Variables ya preparadas:** si tienes el Parquet de variables del mes (su `.sources.json` debe
  estar junto a él), pásalo con `MIBICI_FEATURES=<ruta completa>` y el notebook no vuelve a
  construirlo.

## Publicar la evaluación

Para guardar la carpeta del mes aquí y registrar la corrida en el experimento `mibici-monitoreo`
de MLflow, ejecuta el notebook con:

```bash
MIBICI_REGISTRAR_MLFLOW=1 MIBICI_TRACKING_URI=sqlite:///$PWD/mlflow.db make monitor MES=<AAAA-MM>
```

La ruta de `mlflow.db` va completa porque el notebook se ejecuta desde `notebooks/`.

El experimento `mibici-monitoreo` guarda sus artefactos en una carpeta local de la computadora donde
se creó, así que el registro solo funciona ahí mientras el equipo no use un servidor de MLflow
compartido. La evidencia completa también queda en la carpeta del mes de este directorio, que se
sube al repositorio con su pull request.

| Mes | Modelo | Resultado |
|---|---|---|
| 2026-10 | v3 | Alertas de entrada (+79 %) y de MAE (+56 %) activadas; el modelo sigue 21 % por debajo de la línea base. Decisión: investigar calidad, cobertura y error por zona antes de reentrenar. |
