FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /opt/mlp

COPY requirements-mlp-docker.txt ./requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY models/modelo_mlp_fraude.keras /opt/source/modelo_mlp_fraude.keras
COPY docker/export_mlp_model.py ./export_mlp_model.py

RUN python export_mlp_model.py \
    --source /opt/source/modelo_mlp_fraude.keras \
    --destination /opt/mlflow/model

RUN useradd --create-home --uid 10001 modeluser \
    && chown -R modeluser:modeluser /opt/mlp /opt/mlflow

USER modeluser

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=45s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=5)"]

CMD ["python", "-m", "mlflow", "models", "serve", "-m", "/opt/mlflow/model", "--host", "0.0.0.0", "--port", "8080", "--env-manager", "local"]
