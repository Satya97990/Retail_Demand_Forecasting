FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip && pip install "mlflow>=2.12,<4"
WORKDIR /mlflow
EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:////mlflow/mlflow.db", "--default-artifact-root", "/mlflow/artifacts"]
