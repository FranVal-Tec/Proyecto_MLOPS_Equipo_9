 .gitignore                                         |   11 [32m+[m[31m-[m
 Makefile                                           |   11 [32m+[m[31m-[m
 data/silver/version_001/README.md                  |   20 [32m+[m
 data/silver/version_001/_SUCCESS.json              |    6 [32m+[m
 data/silver/version_001/fuentes/codigo/__init__.py |    1 [32m+[m
 data/silver/version_001/fuentes/codigo/cleaning.py |  727 [32m++++[m
 data/silver/version_001/fuentes/codigo/config.py   |   32 [32m+[m
 data/silver/version_001/fuentes/codigo/dataset.py  |  115 [32m+[m
 .../version_001/fuentes/nomenclatura_2026_09.csv   |  485 [32m+++[m
 .../version_001/fuentes/notebook_limpieza.ipynb    | 3881 [32m++++++++++++++++++++[m
 data/silver/version_001/layout.md                  |   57 [32m+[m
 data/silver/version_001/manifest.json              |  667 [32m++++[m
 data/silver/version_001/metadatos/bitacora.csv     |  253 [32m++[m
 data/silver/version_001/metadatos/correcciones.csv |   64 [32m+[m
 data/silver/version_001/metadatos/entorno.json     |    8 [32m+[m
 .../version_001/metadatos/indicadores_calidad.csv  |  358 [32m++[m
 data/silver/version_001/metadatos/layout.json      |  232 [32m++[m
 .../version_001/metadatos/manifest_entradas.csv    |   23 [32m+[m
 .../version_001/metadatos/resumen_mensual.csv      |   22 [32m+[m
 data/silver/version_001/politica.json              |   14 [32m+[m
 .../version_001/revision/viajes_2025_01.parquet    |  Bin [31m0[m -> [32m325942[m bytes
 .../version_001/revision/viajes_2025_02.parquet    |  Bin [31m0[m -> [32m316561[m bytes
 .../version_001/revision/viajes_2025_03.parquet    |  Bin [31m0[m -> [32m364599[m bytes
 .../version_001/revision/viajes_2025_04.parquet    |  Bin [31m0[m -> [32m317610[m bytes
 .../version_001/revision/viajes_2025_05.parquet    |  Bin [31m0[m -> [32m347613[m bytes
 .../version_001/revision/viajes_2025_06.parquet    |  Bin [31m0[m -> [32m340741[m bytes
 .../version_001/revision/viajes_2025_07.parquet    |  Bin [31m0[m -> [32m337471[m bytes
 .../version_001/revision/viajes_2025_08.parquet    |  Bin [31m0[m -> [32m349043[m bytes
 .../version_001/revision/viajes_2025_09.parquet    |  Bin [31m0[m -> [32m352699[m bytes
 .../version_001/revision/viajes_2025_10.parquet    |  Bin [31m0[m -> [32m363485[m bytes
 .../version_001/revision/viajes_2025_11.parquet    |  Bin [31m0[m -> [32m359237[m bytes
 .../version_001/revision/viajes_2025_12.parquet    |  Bin [31m0[m -> [32m323904[m bytes
 .../version_001/revision/viajes_2026_01.parquet    |  Bin [31m0[m -> [32m374278[m bytes
 .../version_001/revision/viajes_2026_02.parquet    |  Bin [31m0[m -> [32m491414[m bytes
 .../version_001/revision/viajes_2026_03.parquet    |  Bin [31m0[m -> [32m386805[m bytes
 .../version_001/revision/viajes_2026_04.parquet    |  Bin [31m0[m -> [32m346286[m bytes
 .../version_001/revision/viajes_2026_05.parquet    |  Bin [31m0[m -> [32m376466[m bytes
 .../version_001/revision/viajes_2026_06.parquet    |  Bin [31m0[m -> [32m667785[m bytes
 .../version_001/revision/viajes_2026_07.parquet    |  Bin [31m0[m -> [32m616734[m bytes
 .../version_001/revision/viajes_2026_08.parquet    |  Bin [31m0[m -> [32m575307[m bytes
 .../version_001/revision/viajes_2026_09.parquet    |  Bin [31m0[m -> [32m658354[m bytes
 data/silver/version_001/verificacion.json          |    7 [32m+[m
 .../version_001/viajes/viajes_2025_01.parquet      |  Bin [31m0[m -> [32m11300349[m bytes
 .../version_001/viajes/viajes_2025_02.parquet      |  Bin [31m0[m -> [32m11086376[m bytes
 .../version_001/viajes/viajes_2025_03.parquet      |  Bin [31m0[m -> [32m11880771[m bytes
 .../version_001/viajes/viajes_2025_04.parquet      |  Bin [31m0[m -> [32m10939962[m bytes
 .../version_001/viajes/viajes_2025_05.parquet      |  Bin [31m0[m -> [32m11708507[m bytes
 .../version_001/viajes/viajes_2025_06.parquet      |  Bin [31m0[m -> [32m10715957[m bytes
 .../version_001/viajes/viajes_2025_07.parquet      |  Bin [31m0[m -> [32m10976323[m bytes
 .../version_001/viajes/viajes_2025_08.parquet      |  Bin [31m0[m -> [32m11130870[m bytes
 .../version_001/viajes/viajes_2025_09.parquet      |  Bin [31m0[m -> [32m11197135[m bytes
 .../version_001/viajes/viajes_2025_10.parquet      |  Bin [31m0[m -> [32m12265872[m bytes
 .../version_001/viajes/viajes_2025_11.parquet      |  Bin [31m0[m -> [32m11282163[m bytes
 .../version_001/viajes/viajes_2025_12.parquet      |  Bin [31m0[m -> [32m10216063[m bytes
 .../version_001/viajes/viajes_2026_01.parquet      |  Bin [31m0[m -> [32m10691857[m bytes
 .../version_001/viajes/viajes_2026_02.parquet      |  Bin [31m0[m -> [32m9980805[m bytes
 .../version_001/viajes/viajes_2026_03.parquet      |  Bin [31m0[m -> [32m11179364[m bytes
 .../version_001/viajes/viajes_2026_04.parquet      |  Bin [31m0[m -> [32m10554589[m bytes
 .../version_001/viajes/viajes_2026_05.parquet      |  Bin [31m0[m -> [32m9623606[m bytes
 .../version_001/viajes/viajes_2026_06.parquet      |  Bin [31m0[m -> [32m10410556[m bytes
 .../version_001/viajes/viajes_2026_07.parquet      |  Bin [31m0[m -> [32m11269262[m bytes
 .../version_001/viajes/viajes_2026_08.parquet      |  Bin [31m0[m -> [32m11411348[m bytes
 .../version_001/viajes/viajes_2026_09.parquet      |  Bin [31m0[m -> [32m13649679[m bytes
 docs/abrir_mlflow.md                               |   45 [32m+[m
 docs/registro_mlflow.md                            |   42 [32m+[m
 docs/versionado_datos.md                           |  131 [32m+[m
 .../artifacts/codigo_limpieza/__init__.py          |    1 [32m+[m
 .../artifacts/codigo_limpieza/cleaning.py          |  727 [32m++++[m
 .../artifacts/codigo_limpieza/config.py            |   32 [32m+[m
 .../artifacts/codigo_limpieza/dataset.py           |  115 [32m+[m
 .../artifacts/codigo_registro/register_data.py     |  310 [32m++[m
 .../artifacts/metadatos/bitacora.csv               |  253 [32m++[m
 .../artifacts/metadatos/correcciones.csv           |   64 [32m+[m
 .../artifacts/metadatos/entorno.json               |    8 [32m+[m
 .../artifacts/metadatos/indicadores_calidad.csv    |  358 [32m++[m
 .../artifacts/metadatos/layout.json                |  232 [32m++[m
 .../artifacts/metadatos/manifest_entradas.csv      |   23 [32m+[m
 .../artifacts/metadatos/resumen_mensual.csv        |   22 [32m+[m
 .../artifacts/metodologia/notebook_limpieza.ipynb  | 3881 [32m++++++++++++++++++++[m
 .../artifacts/registro/registro_mlflow.json        |  542 [32m+++[m
 .../artifacts/version/README.md                    |   20 [32m+[m
 .../artifacts/version/_SUCCESS.json                |    6 [32m+[m
 .../artifacts/version/layout.md                    |   57 [32m+[m
 .../artifacts/version/manifest.json                |  667 [32m++++[m
 .../artifacts/version/politica.json                |   14 [32m+[m
 .../artifacts/version/verificacion.json            |    7 [32m+[m
 mlflow_limpieza_integrada.db                       |  Bin [31m0[m -> [32m1212416[m bytes
 notebooks/1.2-integracion-limpieza-final.ipynb     | 3881 [32m++++++++++++++++++++[m
 proyecto_mlops_equipo_9/cleaning.py                |  914 [32m++++[m[31m-[m
 proyecto_mlops_equipo_9/open_mlflow.py             |  300 [32m++[m
 proyecto_mlops_equipo_9/register_data.py           |  310 [32m++[m
 reports/versionado/version_001/README.md           |   17 [32m+[m
 reports/versionado/version_001/_SUCCESS.json       |    6 [32m+[m
 reports/versionado/version_001/bitacora.csv        |  253 [32m++[m
 reports/versionado/version_001/correcciones.csv    |   64 [32m+[m
 reports/versionado/version_001/entorno.json        |    8 [32m+[m
 reports/versionado/version_001/layout.md           |   57 [32m+[m
 reports/versionado/version_001/manifest.json       |  667 [32m++++[m
 .../versionado/version_001/manifest_entradas.csv   |   23 [32m+[m
 reports/versionado/version_001/politica.json       |   14 [32m+[m
 .../versionado/version_001/registro_mlflow.json    |  542 [32m+++[m
 reports/versionado/version_001/resumen_mensual.csv |   22 [32m+[m
 reports/versionado/version_001/verificacion.json   |    7 [32m+[m
 tests/test_cleaning.py                             |  237 [32m+[m[31m-[m
 tests/test_open_mlflow.py                          |  107 [32m+[m
 105 files changed, 21706 insertions(+), 274 deletions(-)
