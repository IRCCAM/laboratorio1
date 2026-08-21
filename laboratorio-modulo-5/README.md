# laboratorio1

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

laboratorio 1. modulo 5

## Pipeline reproducible con DVC

El proyecto usa Git para versionar código y configuración, y DVC para reproducir
el entrenamiento y almacenar datasets y artefactos pesados en Google Drive. El
remoto predeterminado está declarado como `gdrive_remote` en `.dvc/config`.

El grafo de `dvc.yaml` contiene cuatro etapas:

1. `prepare`: valida el dataset, crea la partición estratificada y ajusta el
   preprocesador únicamente con entrenamiento.
2. `train_mlp`: entrena la red supervisada sensible al costo.
3. `train_autoencoder`: entrena el detector de anomalías solo con operaciones
   legítimas.
4. `evaluate`: selecciona umbrales con validación y calcula las métricas finales
   una sola vez sobre test.

Después de instalar las dependencias, el flujo habitual es:

```bash
# Recuperar el dataset y artefactos disponibles desde Google Drive
dvc pull

# Ejecutar solamente las etapas nuevas o afectadas por cambios
dvc repro

# Consultar y comparar las métricas declaradas en dvc.yaml
dvc metrics show
dvc metrics diff

# Sincronizar los nuevos artefactos pesados con Google Drive
dvc push
```

Los hiperparámetros se modifican en `params.yaml`. Por ejemplo, cambiar
`mlp.epochs` invalida `train_mlp` y `evaluate`, pero no vuelve a preparar los
datos ni a entrenar el autoencoder. Se deben registrar en Git `params.yaml`,
`dvc.yaml`, `dvc.lock`, el código y los archivos `.gitignore`; los contenidos
pesados se transfieren con DVC.

La primera autenticación contra Google Drive puede abrir el navegador. Las
credenciales generadas por DVC son locales y no deben agregarse al repositorio.

## Seguimiento de experimentos con MLflow

MLflow complementa el pipeline DVC: DVC reproduce las etapas y versiona los
artefactos pesados, mientras MLflow registra una corrida para la MLP y otra para
el autoencoder. Cada corrida contiene hiperparámetros, métricas finales, firma
de entrada, modelo Keras y trazabilidad mediante hashes de Git, DVC y dataset.

La etapa `track_mlflow` se ejecuta después de `evaluate` como parte de:

```bash
dvc repro
```

El backend local se configura en `params.yaml` y utiliza `mlflow.db`; los
artefactos de los runs se guardan en `mlartifacts/`. Ambos son locales y están
excluidos de Git y DVC. Para abrir la interfaz:

```bash
python -m mlflow server --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts --port 5000
```

Luego visita `http://127.0.0.1:5000`. Si se recuperaron modelos con `dvc pull`
pero no existen runs locales, se puede reconstruir el tracking sin reentrenar:

```bash
python -m laboratorio1.tracking
```

Esta configuración local es apropiada para desarrollo individual. Un equipo
debe sustituir `mlflow.tracking_uri` por la URL de un Tracking Server compartido.

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
│                         laboratorio1 and configuration for tools like black
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
└── laboratorio1   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes laboratorio1 a Python module
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

