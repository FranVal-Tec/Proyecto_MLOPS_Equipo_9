"""Pruebas de las figuras del modelado."""

import json

import pandas as pd

from proyecto_mlops_equipo_9.plots import generar_figuras


def test_genera_las_tres_figuras(tmp_path):
    entrada = tmp_path / "modelado"
    entrada.mkdir()
    pd.DataFrame(
        {
            "modelo": ["linea_base", "ridge", "hist_gradient_boosting"],
            "mae_promedio": [2.9, 2.5, 2.1],
            "mae_desviacion": [0.1, 0.06, 0.03],
        }
    ).to_csv(entrada / "comparacion_modelos.csv", index=False)
    pd.DataFrame(
        {
            "mes": ["2026-06", "2026-07"],
            "mae_fijo": [2.1, 2.0],
            "mae_reentrenado": [2.09, 1.97],
            "mae_linea_base": [2.9, 2.7],
        }
    ).to_csv(entrada / "drift_mensual.csv", index=False)
    prueba = {"prueba": {"mae": 2.0, "mae_volumen_bajo": 0.9, "mae_volumen_alto": 3.6}}
    (entrada / "resultado_prueba.json").write_text(json.dumps(prueba), encoding="utf-8")

    rutas = generar_figuras(entrada, tmp_path / "figuras")

    assert [r.name for r in rutas] == [
        "2_comparacion_modelos.png",
        "2_drift_mensual.png",
        "2_error_por_volumen.png",
    ]
    assert all(r.stat().st_size > 10_000 for r in rutas)
