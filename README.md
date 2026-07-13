# IoT Network Anomaly Detection — Docker Deploy

Baseline documentation for the current state of this repository.  
This document describes **what exists today** (components, flows, contracts). Known issues and probable fixes live in [`docs/ISSUES.md`](docs/ISSUES.md).

> **Note:** `iot_anomaly_detection.tar` is gitignored (exceeds GitHub’s file size limit) and is not published with this repository.

---

## 1. Purpose

Train and serve classifiers that detect network anomalies / attack types in IoT traffic using features typical of CICFlowMeter-style flow records (e.g. from the **ACI-IoT-2023** dataset).

Primary stack:

| Stage | Tech |
|--------|------|
| Training | pandas, scikit-learn, XGBoost, LightGBM, Random Forest, MLflow |
| Artifacts | joblib (zlib-compressed) under `models/model_<timestamp>/` |
| Serving | FastAPI + Uvicorn |
| Packaging | Docker (multi-stage image) |

---

## 2. Repository layout

```
docker-deploy/
├── train_model.py                          # Main training pipeline (custom DataPreprocessor)
├── train_model_quantized.py                # Near-duplicate of train_model.py
├── train_model_quantized_no_customclass.py # Sklearn Pipeline / ColumnTransformer variant
├── quantize.py                             # TFLite float16 conversion script (experimental)
├── deploy_api.py                           # Production-oriented FastAPI app (used by Dockerfile)
├── model_bundle.py                         # Shared load/save contract for model artifacts
├── Dockerfile                              # Multi-stage image for deploy_api
├── Original.Dockerfile                     # Earlier single-stage Dockerfile (legacy)
├── requirements.txt                        # Shared Python dependencies
├── models/
│   └── model_20250728_222231/              # Currently checked-in model bundle
├── mlruns/                                 # Local MLflow run store
├── mcp/                                    # Alternate FastAPI + MCP-style wrapper (separate nested git repo)
└── iot_anomaly_detection.tar               # Large Docker image archive (gitignored)
```

Canonical paths for the baseline:

| Concern | Canonical file |
|---------|----------------|
| Train (custom preprocessor) | `train_model.py` |
| Train (sklearn pipeline) | `train_model_quantized_no_customclass.py` |
| Serve | `deploy_api.py` |
| Container | `Dockerfile` |

---

## 3. End-to-end flow

```text
ACI-IoT-2023.csv  -->  enhance features  -->  preprocess / scale
                                              -->  train RF / XGB / LGBM
                                              -->  evaluate + MLflow log
                                              -->  save models/model_<YYYYMMDD_HHMMSS>/
                                                        |
                                                        v
                                              FastAPI /predict  (Docker :8000)
```

1. Load CSV in chunks from `../ACI-IoT-2023.csv` (path relative to this repo’s parent directory).
2. Derive simple ratio features (`packet_ratio`, `byte_ratio` when source columns exist).
3. Drop non-feature / leaky columns when present:  
   `Flow Bytes/s`, `Flow Packets/s`, `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`.
4. Encode labels, scale numeric features, one-hot categoricals (implementation differs by training script).
5. Stratified 80/20 train/test split (`random_state=42`).
6. Train three models; log metrics and models to MLflow.
7. Persist timestamped artifact directory under `models/`.
8. Serving loads the **lexicographically latest** `models/model_*` folder and runs inference.

---

## 4. Data expectations

### Required for training

| Column | Role |
|--------|------|
| `Label` | Target class (attack type / benign) |

### Core numeric features used by feature enhancement and API schemas

| Column (dataset naming) | Meaning |
|-------------------------|---------|
| `Total Fwd Packets` | Forward packet count |
| `Total Backward Packets` | Backward packet count |
| `Total Length of Fwd Packets` | Forward byte volume |
| `Total Length of Bwd Packets` | Backward byte volume |
| `Flow Duration` | Flow duration (microseconds in schema descriptions) |

Training scripts also consume the rest of the CSV after dropping the list above. Categorical columns are one-hot encoded during fit.

### Derived features (when inputs exist)

- `packet_ratio` ≈ forward / backward packets  
- `byte_ratio` ≈ forward / backward bytes  

---

## 5. Training

### Prerequisites

- Python 3.9+ recommended (Dockerfile uses `python:3.9-slim`).
- Dataset file at: `../ACI-IoT-2023.csv` relative to this repo.
- Dependencies from `requirements.txt` (and XGBoost / LightGBM as also installed in Docker).

```bash
pip install -r requirements.txt
```

### Run training

**Custom preprocessor path:**

```bash
python train_model.py
```

**Sklearn `ColumnTransformer` path** (also saves a separate `label_encoder.joblib`):

```bash
python train_model_quantized_no_customclass.py
```

### Models trained

| Key | Algorithm | Default notes |
|-----|-----------|---------------|
| `random_forest` | `RandomForestClassifier` | `n_estimators=100`, `n_jobs=-1` |
| `xgboost` | `XGBClassifier` | `n_estimators=100`, `n_jobs=-1` |
| `lightgbm` | `LGBMClassifier` | Tuned leaves/depth/reg; `n_estimators=100` |

### Metrics logged

Per model: accuracy, precision, recall, F1 (weighted averages where applicable).  
Stored in `metadata.json` and in MLflow under `mlruns/`.

### Checked-in baseline metrics (`model_20250728_222231`)

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| random_forest | 0.9996 | 0.9996 | 0.9996 | 0.9996 |
| xgboost | 0.9988 | 0.9988 | 0.9988 | 0.9988 |
| lightgbm | 0.9995 | 0.9995 | 0.9995 | 0.9995 |

Artifact compression noted in metadata: **zlib level 3**.  
Serve contract: `sklearn_column_transformer_v1` (see `model_bundle.py`).

> This bundle includes a standalone `label_encoder.joblib`, matching `train_model_quantized_no_customclass.py` / `save_bundle`.

---

## 6. Model artifact bundle

Canonical layout (ISS-03), written by `model_bundle.save_bundle`:

```text
models/model_<YYYYMMDD_HHMMSS>/
├── random_forest.joblib
├── xgboost.joblib
├── lightgbm.joblib
├── preprocessor.joblib      # ColumnTransformer
├── label_encoder.joblib     # required for label decode at serve
└── metadata.json
```

`metadata.json` fields:

| Field | Description |
|-------|-------------|
| `timestamp` | Same stamp as folder name |
| `contract` | `sklearn_column_transformer_v1` |
| `results` | Per-model metrics |
| `input_feature_names` | Raw columns expected by the preprocessor |
| `feature_names` | Post-transform feature names |
| `label_encoder_classes` | Class label strings |
| `compression` | Joblib compression setting |

Serving loads the latest `models/model_*` folder via `model_bundle.load_latest_bundle` (startup cache).

---

## 7. Serving — root API (`deploy_api.py`)

### Run locally

```bash
uvicorn deploy_api:app --host 0.0.0.0 --port 8000
# or
python deploy_api.py
```

### Docker

```bash
docker build -t iot-anomaly-api .
docker run --rm -p 8000:8000 iot-anomaly-api
```

Image behavior:

- Multi-stage build (`builder` + slim runtime).
- Copies `models/`, `deploy_api.py`, and `model_bundle.py`.
- Installs `libgomp1` for LightGBM.
- Entrypoint: `uvicorn deploy_api:app --host 0.0.0.0 --port 8000`.

### Endpoint: `POST /predict`

Query parameter:

| Param | Default | Description |
|-------|---------|-------------|
| `model_name` | `random_forest` | One of `random_forest`, `xgboost`, `lightgbm` |

JSON body (Pydantic `InputData` — **ACI-IoT raw column names**; plural/underscore aliases still accepted):

```json
{
  "Total Fwd Packet": 10.0,
  "Total Bwd packets": 8.0,
  "Total Length of Fwd Packet": 1500.0,
  "Total Length of Bwd Packet": 1200.0,
  "Flow Duration": 1000000.0
}
```

Other preprocessor input columns are filled as missing (NaN → imputer) when omitted. For production-quality scores, send the full raw feature set listed in `metadata.input_feature_names`.

Example response shape:

```json
{
  "prediction": "<attack_or_benign_label>",
  "confidence": 0.0,
  "class_probabilities": { "<label>": 0.0 },
  "model_used": "random_forest"
}
```

OpenAPI docs: `http://localhost:8000/docs` when the server is running.

### Runtime load behavior

At **process startup** (FastAPI lifespan):

1. Discovers the latest `models/model_*` folder under `MODEL_BASE_PATH` (default `models`)
2. Loads all classifier joblibs + preprocessor into memory once

Each `/predict` reuses that cache (no per-request disk reload).  
Also available: `GET /health` (liveness), `GET /ready` (artifacts loaded).

Optional env:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_BASE_PATH` | `models` | Root directory containing `model_*` folders |
| `DEFAULT_MODEL_NAME` | `random_forest` | Classifier used when `model_name` query param omitted |

---

## 8. Alternate path — `mcp/`

Nested package with a smaller dependency set and a slightly different contract. Useful reference, not the Dockerfile entrypoint.

| File | Role |
|------|------|
| `mcp/deploy_fastapi.py` | FastAPI app; loads `models/preprocessor.joblib` + `models/random_forest.joblib` (paths relative to `mcp/`) |
| `mcp/preprocessor.py` | Custom `DataPreprocessor` (similar concepts to root trainer) |
| `mcp/mcp_wrapper.py` | Thin `MCPModel` class: load + `predict(input_dict)` |
| `mcp/requirements.txt` | fastapi, uvicorn, joblib, pydantic, scikit-learn, pandas, tqdm |

Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/predict` | Inference; body uses spaced aliases (`Total Fwd Packets`, …) |
| `GET` | `/metadata` | Static model/MCP metadata |
| `GET` | `/schema` | Input/output field documentation |

Response is `{ "prediction": <numeric or list> }` (raw model output, not decoded labels).

---

## 9. Experimental / secondary scripts

| Script | Intent |
|--------|--------|
| `quantize.py` | Convert a TensorFlow SavedModel directory to float16 TFLite |
| `train_model_quantized.py` | Same structure as `train_model.py` (quantization/experiment naming) |
| `Original.Dockerfile` | Older image; CMD references `main:app` (superseded by current `Dockerfile`) |

These are not required for the baseline train → Docker serve path.

---

## 10. Dependencies

From `requirements.txt` (unpinned as of this baseline):

- Training / experiment: tensorflow, numpy, pandas, scikit-learn, xgboost, lightgbm, matplotlib, seaborn, mlflow, tqdm, joblib  
- Serving: fastapi, uvicorn (+ joblib, sklearn family models)

Docker additionally runs `pip install lightgbm xgboost` in the builder stage and installs `libgomp1` in the runtime stage.

---

## 11. Configuration assumptions (baseline)

| Item | Current value |
|------|----------------|
| Dataset path | `../ACI-IoT-2023.csv` |
| Train/test split | 80/20, stratified, `random_state=42` |
| Chunk / batch size | 10_000 |
| Default serve model | `random_forest` |
| Serve port | `8000` |
| Model selection | Latest `models/model_*` directory name |

---

## 12. Documentation roadmap

| Doc | Status |
|-----|--------|
| This README — baseline description | **Current** |
| [Known issues & probable fixes](docs/ISSUES.md) | **Current** |
| Hardened ops runbook (health checks, version pins, CI) | Future |

Fix work follows the priority order in `docs/ISSUES.md`. Update that file’s status table as each issue is resolved; change baseline facts in this README only when behavior actually changes.
