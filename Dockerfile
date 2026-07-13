# Stage 1: Builder for compiling dependencies
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install packages to a temporary directory
RUN pip install --user --no-cache-dir -r requirements.txt
RUN pip install --user --no-cache-dir lightgbm xgboost

# Stage 2: Runtime image
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local
COPY models/ ./models/
COPY deploy_api.py ./
COPY model_bundle.py ./

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "deploy_api:app", "--host", "0.0.0.0", "--port", "8000"]
