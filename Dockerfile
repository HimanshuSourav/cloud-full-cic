# Stage 1: Builder for compiling native extensions (serve deps only — ISS-05)
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .

RUN pip install --user --no-cache-dir -r requirements-serve.txt

# Stage 2: Runtime image
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY models/ ./models/
COPY deploy_api.py ./
COPY model_bundle.py ./

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

CMD ["uvicorn", "deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]
