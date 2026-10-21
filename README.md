# Proyecto_MLOPS_Equipo_9

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Este proyecto tiene como propósito que experimentes de manera práctica cómo se construye, organiza y despliega un sistema de Machine Learning en producción, siguiendo los principios de MLOps.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         proyecto_mlops_equipo_9 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── proyecto_mlops_equipo_9   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes proyecto_mlops_equipo_9 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

##Primer setup local

La primera vez que se descargue el repositorio desde github es necesario hacer setup al ambiente virtual de python, además tambien es necesario realizar la instalación del archivo de requerimientos.
Es necesario recalcar que el repositorio fue creado con la versión Python 3.12.6, para verificar la versión de python instalada se utiliza el siguiente comando:

py --version

Para crear el ambiente virtual es necesario estar dentro de la carpeta raíz y ejecutar:

py -m venv .venv

Una vez creado es necesario activarlo utilizando el comando:

.\.venv\Scripts\Activate.ps1

Una vez se encuentran con el ambiente virtual creado es necesario actualizar pip y luego instalar los requerimientos del proyecto, esto con los siguientes comandos:

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

#Setup de datos

Los datos no se suben al github debido a que estos pueden llegar a ser muy pesados para el repositorio en github, por lo que se agregó la ruta de ./data/ a gitignore, por lo tanto es necesario agregarlos manualmente a dicha carpeta.