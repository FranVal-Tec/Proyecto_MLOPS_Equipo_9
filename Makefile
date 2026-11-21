#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = proyecto_mlops_equipo_9
PYTHON_VERSION = 3.12.6
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset

## Descargar datos crudos de MiBici
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m proyecto_mlops_equipo_9.download --desde 2025-01 --hasta 2026-09


## Generar version silver verificada (data/raw -> data/silver/version_001)
.PHONY: clean_data
clean_data:
	$(PYTHON_INTERPRETER) -m proyecto_mlops_equipo_9.cleaning --version version_001

## Verificar Parquet y manifiestos de la version silver
.PHONY: verify_data
verify_data:
	$(PYTHON_INTERPRETER) -m proyecto_mlops_equipo_9.cleaning --verificar data/silver/version_001


## Generar variables por zona y hora sin MLflow

.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m proyecto_mlops_equipo_9.features --sin-mlflow


## Generar variables por zona y hora registrando en MLflow

.PHONY: features_mlflow
features_mlflow:
	$(PYTHON_INTERPRETER) -m proyecto_mlops_equipo_9.features


## Promover a @campeon la versión de @candidato si cumplió el criterio (requiere mlflow ui --port 5000)

.PHONY: promote_model
promote_model:
	$(PYTHON_INTERPRETER) -m proyecto_mlops_equipo_9.modeling.promover


## Exportar a models/ el modelo @campeon y los datos recientes (requiere mlflow ui --port 5000)

.PHONY: export_model
export_model:
	$(PYTHON_INTERPRETER) -m proyecto_mlops_equipo_9.modeling.exportar


## Verificar que el modelo exportado reproduce su referencia, sin servidor de MLflow

.PHONY: verify_model
verify_model:
	$(PYTHON_INTERPRETER) -m proyecto_mlops_equipo_9.modeling.exportar --verificar


## Levantar la API en http://127.0.0.1:8000/docs

.PHONY: api
api:
	$(PYTHON_INTERPRETER) -m uvicorn proyecto_mlops_equipo_9.fastAPI:app --host 127.0.0.1 --port 8000


#################################################################################
# DOCKER                                                                        #
#################################################################################

IMAGEN ?= ml-service
VERSION ?= 1.0.0
DOCKERHUB_USUARIO ?=


## Construir la imagen ml-service con tags latest y VERSION (verifica el modelo en el build)

.PHONY: docker_build
docker_build:
	docker build --build-arg VERSION=$(VERSION) -t $(IMAGEN):latest -t $(IMAGEN):$(VERSION) .


## Correr la imagen en http://localhost:8000/docs

.PHONY: docker_run
docker_run:
	docker run --rm -p 8000:8000 $(IMAGEN):latest


## Publicar en DockerHub: make docker_push DOCKERHUB_USUARIO=<usuario> VERSION=1.0.0

.PHONY: docker_push
docker_push:
	@test -n "$(DOCKERHUB_USUARIO)" || (echo "Indica DOCKERHUB_USUARIO=<usuario>"; exit 1)
	docker tag $(IMAGEN):$(VERSION) $(DOCKERHUB_USUARIO)/$(IMAGEN):$(VERSION)
	docker tag $(IMAGEN):$(VERSION) $(DOCKERHUB_USUARIO)/$(IMAGEN):latest
	docker push $(DOCKERHUB_USUARIO)/$(IMAGEN):$(VERSION)
	docker push $(DOCKERHUB_USUARIO)/$(IMAGEN):latest



#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
