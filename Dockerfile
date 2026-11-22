# Servicio de pronóstico MiBici (Equipo 9): API de FastAPI con el modelo exportado en models/.
#   docker build -t ml-service:latest .
#   docker run -p 8000:8000 ml-service:latest      ->  http://localhost:8000/docs
FROM python:3.12.15-slim

ARG VERSION=dev
LABEL org.opencontainers.image.title="ml-service" \
      org.opencontainers.image.description="Pronóstico de salidas y llegadas de MiBici por zona y hora" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.source="https://github.com/FranVal-Tec/proyecto_mlops_equipo_9"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Mismas versiones fijas de requirements.txt, sin las herramientas de notebooks, documentación,
# pruebas y lint que el servicio no usa. El código del proyecto se importa desde /app.
COPY requirements.txt .
RUN grep -vE '^(-e |ipython==|jupyterlab==|mkdocs==|notebook==|pytest==|ruff==|uv(==|$))' \
        requirements.txt > requirements-servicio.txt \
    && pip install -r requirements-servicio.txt

COPY proyecto_mlops_equipo_9 ./proyecto_mlops_equipo_9
COPY models ./models

# Reproducibilidad: el build falla si el modelo no es el mismo binario que se exportó o si en
# este entorno no reproduce sus predicciones de referencia.
RUN python -m proyecto_mlops_equipo_9.modeling.export --verificar

RUN useradd --create-home --uid 1000 servicio
USER servicio

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3)"

CMD ["uvicorn", "proyecto_mlops_equipo_9.fastAPI:app", "--host", "0.0.0.0", "--port", "8000"]
