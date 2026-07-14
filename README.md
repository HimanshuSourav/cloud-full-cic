# IoT Network Anomaly Detection — Docker Deploy

Baseline documentation for the current state of this repository.  
This document describes **what exists today** (components, flows, contracts). Known issues and probable fixes live in [`docs/ISSUES.md`](docs/ISSUES.md).

> **Note:** `iot_anomaly_detection.tar` is gitignored (exceeds GitHub’s file size limit) and is not published with this repository.

**Production serve path (this repo):** Docker builds root [`Dockerfile`](Dockerfile) → `uvicorn deploy_api:app` on `:8000`, loading `models/model_*` via [`model_bundle.py`](model_bundle.py).  
Anything under an optional local `mcp/` clone is a **separate project** and is not the Docker entrypoint (see [§8](#8-alternate-path--external-mcp-project)).

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
├── train_model.py                          # Canonical trainer (sklearn ColumnTransformer → save_bundle)
├── deploy_api.py                           # Production FastAPI app (Dockerfile CMD)
├── model_bundle.py                         # Shared load/save contract for model artifacts
├── preproc_scale.py                        # Streaming StandardScaler helper (legacy trainer)
├── Dockerfile                              # Multi-stage image for deploy_api
├── Original.Dockerfile                     # Earlier single-stage Dockerfile (legacy)
├── requirements-serve.txt                  # Pinned deps for Docker / API
├── requirements-train.txt                  # Pinned deps for training (+ serve)
├── requirements.txt                        # Alias → requirements-train.txt
├── models/
│   ├── model_20260713_162252/              # Honest-protocol release bundle (latest)
│   └── model_20250728_222231/              # Historical leaky-protocol bundle
├── mlruns/                                 # Local MLflow run store
├── experiments/                            # Non-canonical / historical scripts only
│   ├── train_model_custom_preprocessor.py  # Legacy custom DataPreprocessor path
│   └── quantize_tflite.py                  # TFLite float16 conversion (experimental)
└── iot_anomaly_detection.tar               # Large Docker image archive (gitignored)
```

Canonical paths for the baseline:

| Concern | Canonical file |
|---------|----------------|
| Train | `train_model.py` |
| Serve | `deploy_api.py` |
| Container | `Dockerfile` → `uvicorn deploy_api:app` |

`mcp/` is **not** part of this tree (gitignored optional clone of a separate GitHub repo).

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

Training scripts also consume the rest of the CSV after dropping non-feature / ID-like columns.

### Columns dropped for the canonical (ISS-06) trainer

| Column | Reason |
|--------|--------|
| `Flow ID`, `Src IP`, `Dst IP`, `Timestamp` | Identifiers / time |
| `Flow Bytes/s`, `Flow Packets/s` | Rate fields (legacy drop) |
| `Src Port`, `Dst Port` | High-importance but lab-leaky ports (Phase 2 ablation) |

Canonical raw ACI names used at serve time include singular forms such as `Total Fwd Packet`, `Total Bwd packets` (see OpenAPI / `model_bundle.REQUEST_COLUMN_ALIASES`).

### Derived features (when inputs exist)

- `packet_ratio` ≈ forward / backward packets  
- `byte_ratio` ≈ forward / backward bytes  

---

## 5. Training

### Prerequisites

- Python 3.12 recommended (Dockerfile uses `python:3.12-slim`).
- Dataset file at: `../ACI-IoT-2023.csv` relative to this repo.
- Dependencies: `pip install -r requirements-train.txt` (or `requirements.txt` alias).

### Run training

**Canonical path** (ISS-03/06/07 — train-only preprocess, drop ports, `save_bundle`):

```bash
python train_model.py
```

Legacy custom-preprocessor and TFLite scripts live under `experiments/` and are **not** on the Docker serve path.
### Models trained

| Key | Algorithm | Default notes |
|-----|-----------|---------------|
| `random_forest` | `RandomForestClassifier` | `n_estimators=100`, `n_jobs=-1` |
| `xgboost` | `XGBClassifier` | `n_estimators=100`, `n_jobs=-1` |
| `lightgbm` | `LGBMClassifier` | Tuned leaves/depth/reg; `n_estimators=100` |

### Metrics logged

Per model: accuracy, precision, recall, F1 weighted + F1 macro.  
Also `classification_reports.json` / `confusion_matrices.json` in the model folder.  
Scalar metrics in `metadata.json` and MLflow under `mlruns/`.

### Release metrics (`model_20260713_162252`, honest protocol)

`evaluation_protocol: train_only_preprocess_v1` — split raw first, fit preprocess on train only, ports dropped.

| Model | Accuracy | F1 weighted | F1 macro |
|-------|----------|-------------|----------|
| random_forest | 0.9990 | 0.9990 | 0.9079 |
| xgboost | 0.9969 | 0.9969 | 0.9035 |
| lightgbm | 0.9968 | 0.9968 | 0.8988 |

Macro F1 is lower mainly due to ultra-rare **ARP Spoofing**. See [`docs/ISS06_VERIFICATION.md`](docs/ISS06_VERIFICATION.md).

### Historical metrics (`model_20250728_222231`, leaky protocol)

Fit-all-then-split; ports kept. Kept for comparison only.

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| random_forest | 0.9996 | 0.9996 | 0.9996 | 0.9996 |
| xgboost | 0.9988 | 0.9988 | 0.9988 | 0.9988 |
| lightgbm | 0.9995 | 0.9995 | 0.9995 | 0.9995 |

Serve contract: `sklearn_column_transformer_v1` (see `model_bundle.py`).

---

## 6. Model artifact bundle

Canonical layout (ISS-03), written by `model_bundle.save_bundle`:

```text
models/model_<YYYYMMDD_HHMMSS>/
├── random_forest.joblib
├── xgboost.joblib
├── lightgbm.joblib
├── preprocessor.joblib      # ColumnTransformer (fit on train only)
├── label_encoder.joblib     # required for label decode at serve
├── classification_reports.json
├── confusion_matrices.json
└── metadata.json
```

`metadata.json` fields:

| Field | Description |
|-------|-------------|
| `timestamp` | Same stamp as folder name |
| `contract` | `sklearn_column_transformer_v1` |
| `evaluation_protocol` | e.g. `train_only_preprocess_v1` |
| `dropped_columns` | Columns removed before fit |
| `results` | Per-model scalar metrics |
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

Error status codes (ISS-08):

| Status | When |
|--------|------|
| `404` | Unknown `model_name` (detail lists available models) |
| `422` | Missing / invalid JSON body fields |
| `400` | Feature transform / encode failed (actionable `detail`) |
| `503` | Artifacts not loaded (`/ready` and `/predict`) |
| `500` | Unexpected server error |
Optional env:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_BASE_PATH` | `models` | Root directory containing `model_*` folders |
| `DEFAULT_MODEL_NAME` | `random_forest` | Classifier used when `model_name` query param omitted |

---

## 8. Alternate path — external MCP project

**Not maintained in this repository.** A historically related FastAPI + MCP-style wrapper lives in a **separate** GitHub project:

https://github.com/HimanshuSourav/MCP-Compliant-IoT-Network-Anomaly-Detection

| This repo (`docker-deploy`) | External MCP repo |
|-----------------------------|-------------------|
| `deploy_api.py` + `model_bundle.py` | Its own `deploy_fastapi.py` / wrapper |
| `sklearn_column_transformer_v1` bundles under `models/model_*` | Own `models/*.joblib` layout |
| Root `Dockerfile` serve path | Different deps / schema / response shape |

Optional local clone (gitignored here so it is not dual-maintained by accident):

```bash
git clone https://github.com/HimanshuSourav/MCP-Compliant-IoT-Network-Anomaly-Detection.git mcp
```

Do **not** point production Docker or CI at `mcp/`.

---

## 9. Experimental / secondary scripts

| Script | Intent |
|--------|--------|
| `experiments/train_model_custom_preprocessor.py` | Legacy custom `DataPreprocessor` trainer (does **not** write `sklearn_column_transformer_v1`) |
| `experiments/quantize_tflite.py` | Convert a TensorFlow SavedModel directory to float16 TFLite |
| `Original.Dockerfile` | Older image; CMD references `main:app` (superseded by current `Dockerfile`) |

These are not required for the baseline train → Docker serve path. Only `train_model.py` produces the serve bundle via `model_bundle.save_bundle`.

---

## 10. Dependencies

Split and pinned (ISS-05):

| File | Use |
|------|-----|
| `requirements-serve.txt` | Docker / FastAPI serving (numpy, pandas, scikit-learn, xgboost, lightgbm, joblib, fastapi, uvicorn, pydantic) |
| `requirements-train.txt` | Local training & experiments (`-r` serve + tqdm, matplotlib, seaborn, mlflow, tensorflow) |
| `requirements.txt` | Alias → `requirements-train.txt` (backward compatible) |

```bash
# Serve / API
pip install -r requirements-serve.txt

# Train locally
pip install -r requirements-train.txt
```

Docker installs **only** `requirements-serve.txt` (Python 3.12 slim) and runtime `libgomp1` for LightGBM.

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
