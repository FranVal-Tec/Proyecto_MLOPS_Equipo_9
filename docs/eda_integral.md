# EDA integral MiBici — Equipo 9

Objetivo confirmado: pronosticar demanda de bicicletas por zona y hora. El análisis
mide demanda atendida observada y documenta las restricciones de cobertura y calidad.

## Archivos

- `notebooks/1.0-EDA-integral-mibici.ipynb`: notebook con narrativa y resultados.
- `proyecto_mlops_equipo_9/eda.py`: funciones de inventario, agregación, sensibilidad,
  zonificación exploratoria y exportación, utilizando el cargador del proyecto y criterios de evaluación propios del EDA.
- `EDA-integral-mibici.html`: versión de lectura con tablas y gráficas incrustadas.
- `reports/eda_integral/`: figuras, tablas agregadas, manifiesto de fuentes y registro local en MLflow.

No reemplaza el notebook exploratorio original ni modifica la limpieza productiva.
No genera otra versión silver. No se copiaron los CSV originales a esta entrega.

## Repetir

Colocar el notebook en `notebooks/` y el módulo en `proyecto_mlops_equipo_9/` del
repositorio que contiene los datos descargados. Ejecutar todas las celdas en el
entorno del proyecto, con las dependencias de `requirements.txt`.

El notebook encuentra la raíz desde su ubicación. Si se abre fuera del proyecto,
definir `MIBICI_PROJ_ROOT` con la ruta de ese repositorio. `MIBICI_EDA_OUT` permite
cambiar el destino de informes. El registro MLflow crea una corrida local nueva;
puede desactivarse cambiando `REGISTRAR_MLFLOW` en el notebook.

Las tablas de conteos y la sensibilidad cubren todos los registros de los archivos
disponibles. Las distribuciones gráficas detalladas usan hasta 15,000 registros por
archivo, semilla 9 y pesos para compensar tamaños mensuales diferentes.

## Decisiones pendientes

Las zonas son una cuadrícula exploratoria de 1 km, no polígonos operativos aprobados.
Se comparan tamaños alternativos. Debe acordarse con negocio la zonificación,
el horizonte y los costos de error, así como revisar los umbrales de duración y
velocidad. Los días bajos no se eliminan automáticamente. No se atribuyen causas
a e-bikes ni se infiere inventario o demanda insatisfecha a partir de viajes.

El periodo completo ya fue inspeccionado en el EDA: no debe declararse una prueba
ciega. El panel horario y el preprocesador son demostrativos; faltan disponibilidad
operativa histórica y un protocolo de evaluación temporal antes de modelar.

## Trazabilidad y colaboración

Las huellas SHA-256 identifican insumos y código; no sustituyen conservar los archivos.
El tracking MLflow local conserva rutas absolutas: al mover el proyecto hay que
ejecutar nuevamente o migrar el almacén. Versionar notebook, módulo y dependencias
con Git según el flujo del equipo; los datos y las bases locales de MLflow requieren
la política de almacenamiento del proyecto. No se crearon commits ni se publicaron cambios.

Elaborado con asistencia de OpenAI Codex (GPT-6) para estructuración, programación y
revisión. La revisión académica y la aprobación de las decisiones corresponden al equipo.
