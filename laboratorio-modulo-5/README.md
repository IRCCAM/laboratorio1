# laboratorio1

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

laboratorio 1. modulo 5

## FastAPI local con el MLP en Docker

FastAPI se ejecuta localmente; la API no se dockeriza. Al iniciar, carga desde
`models/` únicamente el preprocesador y el umbral de decisión. La inferencia se
delega al contenedor del modelo MLP mediante `GET /health` y `POST /invocations`,
el contrato HTTP estándar de MLflow.

Primero se recupera el modelo versionado con DVC y se construye el contenedor
que publica únicamente el MLP en el puerto `8080`:

```powershell
cd C:\Modulo5\laboratorio1\laboratorio-modulo-5
.\.venv\Scripts\python.exe -m dvc pull
docker compose up --build mlp-model
```

Cuando la terminal muestre que MLflow escucha en `0.0.0.0:8080`, se abre una
segunda terminal de PowerShell para iniciar FastAPI localmente:

```powershell
cd C:\Modulo5\laboratorio1\laboratorio-modulo-5
$env:MLP_SERVICE_URL = "http://127.0.0.1:8080"
$env:MLP_SERVICE_TIMEOUT_SECONDS = "10"
.\.venv\Scripts\python.exe -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

La documentación interactiva queda disponible en `http://127.0.0.1:8000/docs`.
`GET /health` confirma que FastAPI puede comunicarse con el contenedor y
`POST /predict` acepta un objeto `data` con una o más transacciones. Cada
transacción debe incluir `Time`, `Amount` y las variables `V1` a `V28`:

```json
{
  "data": [
    {
      "Time": 0,
      "Amount": 149.62,
      "V1": -1.3598,
      "V2": -0.0728,
      "V3": 2.5363,
      "V4": 1.3782,
      "V5": -0.3383,
      "V6": 0.4624,
      "V7": 0.2396,
      "V8": 0.0987,
      "V9": 0.3638,
      "V10": 0.0908,
      "V11": -0.5516,
      "V12": -0.6178,
      "V13": -0.9914,
      "V14": -0.3112,
      "V15": 1.4682,
      "V16": -0.4704,
      "V17": 0.208,
      "V18": 0.0258,
      "V19": 0.404,
      "V20": 0.2514,
      "V21": -0.0183,
      "V22": 0.2778,
      "V23": -0.1105,
      "V24": 0.0669,
      "V25": 0.1285,
      "V26": -0.1891,
      "V27": 0.1336,
      "V28": -0.0211
    }
  ]
}
```

Para detener solamente el contenedor MLP al finalizar:

```powershell
docker compose down
```

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

El puerto `5000` corresponde al seguimiento y la interfaz de MLflow. La
inferencia consumida por FastAPI es un servicio separado que el contenedor MLP
publica en el puerto `8080`.

El modelo del contenedor, el preprocesador y el umbral deben proceder de la
misma ejecución de DVC/MLflow; mezclar versiones puede producir predicciones
incorrectas aunque ambos servicios respondan correctamente.

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

---
