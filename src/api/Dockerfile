# Slim Python base. Install build essentialdue pyarrow / sklearn
# compiler on minor for less mismatches
FROM python:3.11-slim

WORKDIR /app

# System dependencies for scientific Python.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first so layer caching kicks in across rebuilds.
COPY src/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Application code
COPY src/api /app/src/api

# Where the trained model + reference parquet live; mounted as a named volume.
RUN mkdir -p /app/artifacts
ENV ARTIFACTS_DIR=/app/artifacts \
    MODEL_PATH=/app/artifacts/model.joblib \
    REFERENCE_DATA_PATH=/app/artifacts/reference.parquet \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

EXPOSE 8080

# Single uvicorn worker is enough for monitoring -- avoids duplicated
# Prometheus counters across worker processes.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
