from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from proyecto_mlops_equipo_9.modeling.servicio import generar_pronostico


class PredictInput(BaseModel):
    """Periodo a pronosticar. Sin periodo, se usa la última hora disponible.

    El modelo y los datos los decide el servidor (ver modeling/servicio.py): el cliente no
    puede elegir archivos ni servidores.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"desde": "2026-09-30T18:00:00", "hasta": "2026-09-30T21:00:00"}]
        },
    )

    desde: datetime | None = Field(
        default=None, description="Primera hora objetivo, incluida (ISO 8601)."
    )
    hasta: datetime | None = Field(
        default=None,
        description="Fin del periodo, excluido. Sin `hasta`, se pronostica solo la hora `desde`.",
    )

    @model_validator(mode="after")
    def validar_periodo(self):
        if self.hasta is not None and self.desde is None:
            raise ValueError("`hasta` requiere `desde`.")
        if self.desde is not None and self.hasta is not None and self.hasta <= self.desde:
            raise ValueError("`hasta` debe ser posterior a `desde`.")
        return self


class PrediccionMetadata(BaseModel):
    fecha_prediccion: str
    zonas_totales: int
    modelo_utilizado: str


class ZonaPrediccion(BaseModel):
    zona_id: str
    salidas_pronosticadas: float
    llegadas_pronosticadas: float
    flujo_neto_pronosticado: float
    tendencia: Literal["positive", "negative", "balanced"]


class PredictOutput(BaseModel):
    status: Literal["success", "error"]
    metadata: PrediccionMetadata
    predicciones: list[ZonaPrediccion]


app = FastAPI(
    title="Pronóstico MiBici - Equipo 9",
    description="API para pronosticar salidas y llegadas por zona.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {"message": "API de pronóstico MiBici funcionando", "version": "1.0.0"}


@app.post("/predict", response_model=PredictOutput)
def predict_endpoint(params: PredictInput | None = None):
    params = params or PredictInput()
    try:
        resultado, modelo_usado = generar_pronostico(desde=params.desde, hasta=params.hasta)

        # Columnas a redondear a dos decimales para leerse más sencillo en la API
        numeric_columns = [
            "salidas_pronosticadas",
            "llegadas_pronosticadas",
            "flujo_neto_pronosticado",
        ]
        resultado[numeric_columns] = resultado[numeric_columns].round(2)
        resultado["tendencia"] = resultado["flujo_neto_pronosticado"].apply(
            lambda x: "positive" if x > 0 else "negative" if x < 0 else "balanced"
        )

        # Extract prediction date
        fecha_prediccion = resultado["inicio_hora_objetivo"].iloc[0]

        return {
            "status": "success",
            "metadata": {
                "fecha_prediccion": str(fecha_prediccion),
                # Con un periodo de varias horas hay una fila por zona y hora.
                "zonas_totales": int(resultado["zona_id"].nunique()),
                "modelo_utilizado": modelo_usado,
            },
            "predicciones": resultado.drop(columns=["inicio_hora_objetivo"]).to_dict(
                orient="records"
            ),
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:  # noqa: BLE001 - cualquier otro error se reporta como 500
        raise HTTPException(
            status_code=500, detail="Error al generar predicción con error:" + str(e)
        )
