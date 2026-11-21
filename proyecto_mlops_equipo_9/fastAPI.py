from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal, Optional

from proyecto_mlops_equipo_9.modeling.predict import generar_pronostico_api

class PredictInput(BaseModel):
    desde: Optional[str] = None
    hasta: Optional[str] = None
    dataset: Optional[str] = None
    modelo_uri: Optional[str] = None
    tracking_uri: Optional[str] = None

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
    version="1.0.0"
)

@app.get("/")
def home():
    return {
        "message": "API de pronóstico MiBici funcionando",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictOutput)
def predict_endpoint(
    params: PredictInput
):
    try:
        resultado, modelo_usado = generar_pronostico_api(
            desde=params.desde,
            hasta=params.hasta,
            dataset=params.dataset,
            modelo_uri=params.modelo_uri, 
            tracking_uri=params.tracking_uri
        )

        #Columnas a redondear a dos decimales para leerse más sencillo en la API
        numeric_columns = [
            "salidas_pronosticadas",
            "llegadas_pronosticadas",
            "flujo_neto_pronosticado"
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
                "zonas_totales": len(resultado),
                "modelo_utilizado": modelo_usado
            },
            "predicciones": resultado.drop(
                columns=["inicio_hora_objetivo"]
            ).to_dict(orient="records"),
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="Error al generar predicción con error:"+str(e)
        )