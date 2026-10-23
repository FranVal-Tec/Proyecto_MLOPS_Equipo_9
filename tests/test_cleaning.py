import pandas as pd

from proyecto_mlops_equipo_9.cleaning import limpiar_viajes


def test_limpiar_viajes():
    datos = pd.DataFrame({
        "Viaje_Id": [1, 1, 2, 3],
        "Usuario_Id": [10, 10, 20, 30],
        "Genero": ["F", "F", None, "M"],
        "Año_de_nacimiento": [2000, 2000, None, 1995],
        "Inicio_del_viaje": [
            "2026-09-01 10:00:00",
            "2026-09-01 10:00:00",
            "2026-09-01 11:00:00",
            "2026-09-01 12:00:00",
        ],
        "Fin_del_viaje": [
            "2026-09-01 10:10:00",
            "2026-09-01 10:10:00",
            "2026-09-01 11:00:30",
            "2026-09-01 16:00:00",
        ],
        "Origen_Id": [1, 1, 2, 3],
        "Destino_Id": [2, 2, 5, 3],
    })

    resultado = limpiar_viajes(datos)

    # Se elimina una copia exacta
    assert len(resultado) == 3

    # Se conservan los nulos demográficos
    assert resultado["genero"].isna().sum() == 1

    # Se identifica el viaje corto entre estaciones distintas
    assert resultado["corto_distinta_estacion"].sum() == 1

    # Se identifica el viaje de cuatro horas
    assert resultado["duracion_larga"].sum() == 1

    # El DataFrame original permanece intacto
    assert "duracion_min" not in datos.columns