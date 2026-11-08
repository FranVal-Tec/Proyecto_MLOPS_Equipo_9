"""Figuras del modelado a partir de los resultados guardados en reports/modelado.

Uso: python -m proyecto_mlops_equipo_9.plots
Genera en reports/figures/ la comparación de modelos, el drift mensual y el error por
volumen de zona. Importar este módulo no lee archivos ni dibuja.
"""

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure
import pandas as pd

from proyecto_mlops_equipo_9.config import FIGURES_DIR, REPORTS_DIR

NOMBRES = {
    "linea_base": "Línea base",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
VOLUMENES = ["bajo", "medio", "alto", "sin_historial"]

# Paleta: un color por papel, no por posición.
COLORES = {
    "modelo": "#2a78d6",
    "reentrenado": "#eb6834",
    "base": "#898781",
    "texto": "#0b0b0b",
    "texto_secundario": "#52514e",
    "rejilla": "#e1e0d9",
    "eje": "#c3c2b7",
    "fondo": "#ffffff",
}


def _figura(ancho=8, alto=4.5, columnas=1):
    figura = Figure(figsize=(ancho, alto), dpi=150, facecolor=COLORES["fondo"])
    ejes = figura.subplots(1, columnas)
    for eje in ejes if columnas > 1 else [ejes]:
        eje.set_facecolor(COLORES["fondo"])
        for lado in ("top", "right"):
            eje.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            eje.spines[lado].set_color(COLORES["eje"])
        eje.tick_params(colors=COLORES["texto_secundario"], labelsize=9)
        eje.set_axisbelow(True)
    return figura, ejes


def _titulo(eje, titulo, subtitulo=None):
    eje.set_title(titulo, loc="left", fontsize=12, color=COLORES["texto"], pad=18)
    if subtitulo:
        eje.text(
            0, 1.02, subtitulo, transform=eje.transAxes, fontsize=9,
            color=COLORES["texto_secundario"],
        )


def cargar_resultados(entrada_dir):
    """Lee la comparación de modelos, el drift mensual y el resultado de prueba."""
    entrada_dir = Path(entrada_dir)
    comparacion = pd.read_csv(entrada_dir / "comparacion_modelos.csv")
    drift = pd.read_csv(entrada_dir / "drift_mensual.csv")
    prueba = json.loads((entrada_dir / "resultado_prueba.json").read_text(encoding="utf-8"))
    return comparacion, drift, prueba


def grafica_comparacion(comparacion):
    """MAE promedio de las ventanas por modelo, con su dispersión; menor es mejor."""
    tabla = comparacion.sort_values("mae_promedio", ascending=True).reset_index(drop=True)
    base = tabla.loc[tabla.modelo.eq("linea_base"), "mae_promedio"].iloc[0]
    figura, eje = _figura(alto=3.6)
    posiciones = range(len(tabla))
    colores = [COLORES["base"] if m == "linea_base" else COLORES["modelo"] for m in tabla.modelo]
    eje.barh(
        posiciones, tabla.mae_promedio, xerr=tabla.mae_desviacion, height=0.55,
        color=colores, error_kw={"ecolor": COLORES["texto_secundario"], "lw": 1, "capsize": 3},
    )
    for i, fila in tabla.iterrows():
        etiqueta = f"{fila.mae_promedio:.2f}"
        if fila.modelo != "linea_base":
            etiqueta += f"   −{1 - fila.mae_promedio / base:.0%} vs. línea base"
        eje.text(
            fila.mae_promedio + fila.mae_desviacion + 0.05, i, etiqueta, va="center",
            fontsize=9, color=COLORES["texto"],
        )
    eje.set_yticks(list(posiciones), [NOMBRES.get(m, m) for m in tabla.modelo])
    eje.set_xlim(0, tabla.mae_promedio.max() * 1.55)
    eje.set_xlabel("MAE (viajes por zona y hora)", color=COLORES["texto_secundario"])
    eje.xaxis.grid(True, color=COLORES["rejilla"], lw=0.8)
    _titulo(
        eje, "Error de validación por modelo",
        "MAE promedio en 6 ventanas mensuales (oct 2025–mar 2026); la barra fina es ±1 desv.",
    )
    figura.tight_layout()
    return figura


def grafica_drift(drift):
    """Error mensual del modelo fijo, el reentrenado y la línea base, y ventaja de reentrenar."""
    meses = [f"{MESES[int(m[5:7]) - 1]} {m[:4]}" for m in drift.mes]
    figura, (eje_mae, eje_ventaja) = _figura(ancho=10, alto=4.2, columnas=2)
    series = [
        ("mae_linea_base", "Línea base", COLORES["base"], "--"),
        ("mae_fijo", "Modelo fijo (hasta mar 2026)", COLORES["modelo"], "-"),
        ("mae_reentrenado", "Reentrenado cada mes", COLORES["reentrenado"], "-"),
    ]
    for columna, nombre, color, estilo in series:
        eje_mae.plot(meses, drift[columna], color=color, ls=estilo, lw=2, marker="o", ms=5,
                     label=nombre)
    eje_mae.set_ylim(0, drift[[s[0] for s in series]].max().max() * 1.2)
    eje_mae.set_ylabel("MAE (viajes por zona y hora)", color=COLORES["texto_secundario"])
    eje_mae.yaxis.grid(True, color=COLORES["rejilla"], lw=0.8)
    eje_mae.legend(frameon=False, fontsize=8, loc="lower left")
    _titulo(eje_mae, "Error mensual en el periodo de drift")

    ventaja = 1 - drift.mae_reentrenado / drift.mae_fijo
    eje_ventaja.bar(meses, ventaja * 100, width=0.55, color=COLORES["reentrenado"])
    for i, valor in enumerate(ventaja):
        eje_ventaja.text(i, valor * 100 + 0.05, f"{valor:.1%}", ha="center", fontsize=9,
                         color=COLORES["texto"])
    eje_ventaja.set_ylim(0, max(ventaja.max() * 100 * 1.3, 1))
    eje_ventaja.set_ylabel("Reducción del MAE (%)", color=COLORES["texto_secundario"])
    eje_ventaja.yaxis.grid(True, color=COLORES["rejilla"], lw=0.8)
    _titulo(eje_ventaja, "Ventaja de reentrenar frente al modelo fijo")
    figura.tight_layout()
    return figura


def grafica_volumen(prueba):
    """MAE de prueba según el volumen de la zona en entrenamiento."""
    metricas = prueba["prueba"]
    grupos = [g for g in VOLUMENES if f"mae_volumen_{g}" in metricas]
    valores = [metricas[f"mae_volumen_{g}"] for g in grupos]
    figura, eje = _figura(ancho=6.5, alto=3.8)
    etiquetas = [g.replace("_", " ").capitalize() for g in grupos]
    eje.bar(etiquetas, valores, width=0.55, color=COLORES["modelo"])
    for i, valor in enumerate(valores):
        eje.text(i, valor + 0.05, f"{valor:.2f}", ha="center", fontsize=9, color=COLORES["texto"])
    eje.axhline(metricas["mae"], color=COLORES["texto_secundario"], lw=1, ls=":")
    eje.text(
        -0.45, metricas["mae"] + 0.06, f"MAE global {metricas['mae']:.2f}",
        ha="left", fontsize=8, color=COLORES["texto_secundario"],
    )
    eje.set_ylim(0, max(valores) * 1.25)
    eje.set_xlabel("Volumen de la zona", color=COLORES["texto_secundario"])
    eje.set_ylabel("MAE (viajes por zona y hora)", color=COLORES["texto_secundario"])
    eje.yaxis.grid(True, color=COLORES["rejilla"], lw=0.8)
    _titulo(eje, "Error de prueba por volumen de zona", "Abril–mayo 2026, modelo registrado")
    figura.tight_layout()
    return figura


def generar_figuras(entrada_dir=REPORTS_DIR / "modelado", salida_dir=FIGURES_DIR):
    """Guarda las tres figuras del modelado y devuelve sus rutas."""
    comparacion, drift, prueba = cargar_resultados(entrada_dir)
    salida_dir = Path(salida_dir)
    salida_dir.mkdir(parents=True, exist_ok=True)
    figuras = {
        "2_comparacion_modelos.png": grafica_comparacion(comparacion),
        "2_drift_mensual.png": grafica_drift(drift),
        "2_error_por_volumen.png": grafica_volumen(prueba),
    }
    rutas = []
    for nombre, figura in figuras.items():
        ruta = salida_dir / nombre
        figura.savefig(ruta, facecolor=figura.get_facecolor())
        rutas.append(ruta)
    return rutas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=REPORTS_DIR / "modelado")
    parser.add_argument("--salida", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    for ruta in generar_figuras(args.entrada, args.salida):
        print(f"Figura guardada en {ruta}")


if __name__ == "__main__":
    main()
