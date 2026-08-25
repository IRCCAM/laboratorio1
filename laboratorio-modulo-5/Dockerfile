FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY laboratorio1 ./laboratorio1

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]