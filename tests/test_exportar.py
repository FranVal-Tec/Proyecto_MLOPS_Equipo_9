"""Pruebas de la exportación del modelo y de su referencia de reproducibilidad."""

from conftest import VARIABLES_EXPORTACION
import pandas as pd
import pytest

from proyecto_mlops_equipo_9.modeling import exportar as ex
from proyecto_mlops_equipo_9.modeling.validation import OBJETIVOS


def test_recientes_toma_los_ultimos_dias_completos(panel_exportacion):
    recientes = ex.seleccionar_recientes(panel_exportacion, VARIABLES_EXPORTACION, dias=7)
    assert recientes.inicio_hora_objetivo.min() == pd.Timestamp("2026-05-25 00:00")
    assert recientes.inicio_hora_objetivo.max() == pd.Timestamp("2026-05-31 23:00")
    assert len(recientes) == 7 * 24 * 3
    assert list(recientes.columns[:2]) == ex.LLAVES
    assert set(VARIABLES_EXPORTACION + OBJETIVOS) <= set(recientes.columns)
    assert recientes[ex.LLAVES].equals(recientes[ex.LLAVES].sort_values(ex.LLAVES))

    un_dia = ex.seleccionar_recientes(panel_exportacion, VARIABLES_EXPORTACION, dias=1)
    assert len(un_dia) == 24 * 3


def test_versiones_fijadas_y_diferencias():
    fijadas = ex.versiones_fijadas(
        "# dependencias\nscikit-learn==1.9.1\nfastapi[standard]==0.120.0\n"
        'numpy==2.5.4 ; python_version >= "3.12"\npandas\n-e .\n'
    )
    assert fijadas == {"scikit-learn": "1.9.1", "fastapi": "0.120.0", "numpy": "2.5.4"}

    entorno = {"python": "3.12.6", "scikit-learn": "1.9.1", "numpy": "2.5.4", "pandas": "3.1.1"}
    entorno |= {"mlflow": None, "skops": None}
    assert ex.diferencias_versiones(entorno, fijadas) == {
        "pandas": {"instalada": "3.1.1", "requirements": None}
    }


def test_referencia_se_reproduce_y_detecta_cambios(panel_exportacion, modelo_exportacion):
    datos = ex.seleccionar_recientes(panel_exportacion, VARIABLES_EXPORTACION, dias=2)
    referencia = ex.predecir_en_orden(modelo_exportacion, datos)
    pd.testing.assert_frame_equal(referencia[ex.LLAVES], datos[ex.LLAVES])
    assert (referencia[ex.PREDICCIONES[:2]] >= 0).all().all()

    resultado = ex.comparar_con_referencia(modelo_exportacion, datos, referencia)
    assert resultado["coincide"] and resultado["max_diferencia"] == 0
    assert set(resultado["metricas"]) >= {"mae", "wape"}

    alterada = referencia.assign(salidas_pronosticadas=referencia.salidas_pronosticadas + 0.5)
    assert not ex.comparar_con_referencia(modelo_exportacion, datos, alterada)["coincide"]

    with pytest.raises(ValueError, match="no corresponden"):
        ex.comparar_con_referencia(modelo_exportacion, datos.iloc[1:], referencia)


def test_metricas_ignoran_horas_sin_objetivo(panel_exportacion, modelo_exportacion):
    datos = ex.seleccionar_recientes(panel_exportacion, VARIABLES_EXPORTACION, dias=1)
    referencia = ex.predecir_en_orden(modelo_exportacion, datos)
    ultima = datos.inicio_hora_objetivo.eq(datos.inicio_hora_objetivo.max())
    sin_objetivo = datos.assign(**{OBJETIVOS[0]: datos[OBJETIVOS[0]].mask(ultima)})

    completas = ex.metricas_disponibles(datos[~ultima], referencia[~ultima])
    assert ex.metricas_disponibles(sin_objetivo, referencia) == completas
    assert ex.metricas_disponibles(datos.assign(**{OBJETIVOS[1]: None}), referencia) is None


def test_huella_depende_del_contenido_no_del_orden_de_creacion(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "sub").mkdir(parents=True)
    (a / "MLmodel").write_text("flavors")
    (a / "sub" / "modelo.skops").write_bytes(b"123")
    (b / "sub").mkdir(parents=True)
    (b / "sub" / "modelo.skops").write_bytes(b"123")
    (b / "MLmodel").write_text("flavors")
    assert ex.huella_directorio(a) == ex.huella_directorio(b)

    (b / "MLmodel").write_text("otro")
    assert ex.huella_directorio(a) != ex.huella_directorio(b)


def test_exportar_y_verificar_sin_servidor(exportacion, requirements_actuales):
    metadatos, modelos = exportacion.metadatos, exportacion.modelos
    assert metadatos["uri_registro"].endswith("/7")
    assert metadatos["alias"] == "campeon"
    assert metadatos["datos"]["filas"] == 7 * 24 * 3
    assert metadatos["versiones_distintas_a_requirements"] is None
    for archivo in (ex.ARCHIVO_METADATOS, ex.ARCHIVO_DATOS, ex.ARCHIVO_REFERENCIA):
        assert (modelos / archivo).is_file()

    resultado = ex.verificar(modelos)
    assert resultado["coincide"] and resultado["mismo_binario"]

    with pytest.raises(FileExistsError):
        ex.exportar(exportacion.dataset, modelos, "x", requirements=requirements_actuales)

    (modelos / metadatos["directorio"] / "agregado.txt").write_text("cambio")
    assert not ex.verificar(modelos)["mismo_binario"]


def test_exportar_exige_las_versiones_de_requirements(panel_exportacion, descarga_falsa, tmp_path):
    pytest.importorskip("pyarrow")
    dataset = tmp_path / "features.parquet"
    panel_exportacion.to_parquet(dataset, index=False)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("scikit-learn==0.0.1\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="scikit-learn"):
        ex.exportar(dataset, tmp_path / "models", "x", requirements=requirements)
    assert not (tmp_path / "models").exists()

    metadatos = ex.exportar(
        dataset, tmp_path / "models", "x", requirements=requirements, ignorar_versiones=True
    )
    diferencias = metadatos["versiones_distintas_a_requirements"]
    assert diferencias["scikit-learn"]["requirements"] == "0.0.1"
