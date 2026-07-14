# Known Issues & Probable Fixes

Companion to the baseline [README](../README.md).  
This catalog reflects defects found in the Jul 2026 review. Fix work proceeds **one issue at a time** in priority order; mark status as items land.

| Status | Meaning |
|--------|---------|
| Open | Not started |
| In progress | Actively being fixed |
| Done | Merged / verified |

---

## Priority summary

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| [ISS-01](#iss-01-feature-name-mismatch-train-vs-api) | Critical | Feature-name mismatch (train vs API) | Done |
| [ISS-02](#iss-02-models-reloaded-on-every-request) | High | Models reloaded on every `/predict` | Done |
| [ISS-03](#iss-03-trainserve-artifact-contract-drift) | Critical | Train/serve artifact contract drift | Done |
| [ISS-04](#iss-04-scaler-fit-on-first-batch-only) | High | `StandardScaler` fit on first 10k rows only | Done |
| [ISS-05](#iss-05-bloated-serving-image--unpinned-deps) | Medium | Bloated serving image / unpinned deps | Done |
| [ISS-06](#iss-06-suspiciously-high-metrics--verify--fix) | High | ~99.9% metrics — verify leakage / eval inflation, then fix | Done |
| [ISS-07](#iss-07-duplicateexperimental-training-scripts) | Low | Duplicate / experimental training scripts | Done |
| [ISS-08](#iss-08-api-error-handling--observability) | Low | Opaque 500s; unclear client errors | Done |
| [ISS-09](#iss-09-mcp-path-diverges-from-canonical-serve) | Low | `mcp/` alternate path diverges | Done |

Suggested fix order:

1. Serving correctness: **ISS-01 → ISS-03 → ISS-02**
2. Training soundness: **ISS-04 → ISS-06** (verify high accuracy, then fix root causes)
3. Hygiene: **ISS-05 → ISS-07 → ISS-08 → ISS-09**

---

## ISS-01: Feature-name mismatch (train vs API)

**Severity:** Critical  
**Status:** Done  
**Where:** `deploy_api.py` (`InputData`), training scripts / dataset columns

### Problem

Training and feature engineering use **spaced** CICFlowMeter-style names, e.g.:

- `Total Fwd Packets`
- `Total Backward Packets`
- `Total Length of Fwd Packets`
- `Flow Duration`

The FastAPI request schema previously used **underscored** names (`Total_Fwd_Packets`, …).  
`pd.DataFrame([input_data.dict()])` therefore produced columns that did **not** match what the preprocessor/`reindex` expects.

Note: MLflow input examples in this repo also show singular variants (`Total Fwd Packet`, `Total Bwd packets`). Broader alias coverage can be added later if needed.

### Fix applied

1. `InputData` JSON aliases use train-time spaced names; `to_feature_frame()` builds the DataFrame with those columns.
2. Legacy underscored keys are remapped via `normalize_feature_keys` / a pre-validator.
3. README `/predict` contract updated; mapping covered by `tests/test_iss01_feature_names.py`.

**Follow-ups (not ISS-01):** persisting full `feature_names` in metadata and validating at serve time remains under ISS-03. End-to-end `/predict` against the checked-in bundle still depends on ISS-03 (preprocessor / `label_encoder` contract).

### Acceptance

- [x] Spaced and legacy underscore payloads produce a DataFrame with spaced column names (`tests/test_iss01_feature_names.py`).
- [ ] Full transform non-zero check against a compatible preprocessor (blocked on ISS-03 bundle contract).

---

## ISS-02: Models reloaded on every request

**Severity:** High  
**Status:** Done  
**Where:** `deploy_api.py` — previously `predict()` constructed `ModelDeployment()` and called `load_latest_model()` per request

### Problem

Every `/predict` reloaded all `.joblib` models plus the preprocessor from disk. That added large latency, disk I/O, and risk of inconsistent state under concurrent load. Model files are multi‑MB (`random_forest.joblib` alone is ~8.5 MB).

### Fix applied

1. FastAPI `lifespan` loads artifacts once into a process-wide cache (`load_artifacts_into_cache`).
2. `/predict` reads from that cache only; returns **503** if startup load failed.
3. Env vars: `MODEL_BASE_PATH` (default `models`), `DEFAULT_MODEL_NAME` (default `random_forest`).
4. Added `GET /health` (liveness) and `GET /ready` (artifacts loaded).
5. Covered by `tests/test_iss02_startup_cache.py` (mock load; assert single call across two predicts).

### Acceptance

- [x] Second `/predict` does not call `load_latest_model` again (unit test).
- [x] `/health` and `/ready` endpoints present.

---

## ISS-03: Train/serve artifact contract drift

**Severity:** Critical  
**Status:** Done  
**Where:** `model_bundle.py`, `deploy_api.py`, `train_model.py` (then named `train_model_quantized_no_customclass.py`), `models/model_20250728_222231/`

### Problem

Serving assumed a **custom** preprocessor API (`preprocessor.label_encoder`, …) while the checked-in bundle is a sklearn **`ColumnTransformer`** plus standalone **`label_encoder.joblib`**, with null metadata fields. Train and serve had diverged.

### Fix applied (Option B)

1. Added shared `model_bundle.py` with `load_bundle` / `save_bundle` / `transform_raw` (`contract`: `sklearn_column_transformer_v1`).
2. `deploy_api` loads a `ModelBundle` (preprocessor + label encoder + classifiers) and decodes via the standalone encoder.
3. Request keys remap to ACI raw names (`Total Fwd Packet`, `Total Bwd packets`, …); missing raw columns are aligned (NaN-filled) before transform.
4. Canonical serve trainer (`train_model.py`, then named `train_model_quantized_no_customclass.py`) saves through `save_bundle` (populated metadata).
5. Checked-in `metadata.json` backfilled with `input_feature_names`, `feature_names`, `label_encoder_classes`.
6. Dockerfile copies `model_bundle.py`; tests in `tests/test_iss03_model_bundle.py`.

### Acceptance

- [x] Cold-start load of the current model dir succeeds.
- [x] One `/predict` returns a class string and probabilities without 500.
- [x] `metadata.json` non-null for feature names and label classes (backfilled now; new trains write them).

---

## ISS-04: Scaler fit on first batch only

**Severity:** High  
**Status:** Done  
**Where:** `preproc_scale.py`, legacy custom trainers (now `experiments/train_model_custom_preprocessor.py`; formerly also `train_model_quantized.py`), `mcp/preprocessor.py`

### Problem

Custom `DataPreprocessor` paths called `scaler.fit_transform` on the **first 10 000 rows only**, then `transform` on later batches. Mean/variance ignored most of the data.

### Fix applied

1. Added `preproc_scale.fit_transform_scaled`: **`scaler.fit(X)` on all rows**, then batch `transform`.
2. Wired into the legacy custom-preprocessor trainers (now under `experiments/`).
3. Same fix inlined in `mcp/preprocessor.py`.
4. Canonical serve trainer (`train_model.py`) already fits via `ColumnTransformer` on the full matrix — unchanged.
5. Covered by `tests/test_iss04_scaler_fit.py`.

**Note:** Retrain is still required before custom-preprocessor metrics reflect this fix (see ISS-06). Bundles already produced with the sklearn path are unaffected by this scaler bug.

### Acceptance

- [x] Scaler `n_samples_seen_` matches full frame length (unit test).
- [ ] Full dataset retrain under custom preprocessor (optional; prefer ISS-06 honest eval on sklearn path).

---

## ISS-05: Bloated serving image & unpinned deps

**Severity:** Medium  
**Status:** Done  
**Where:** `requirements-serve.txt`, `requirements-train.txt`, `requirements.txt`, `Dockerfile`

### Problem

- Single unpinned `requirements.txt` pulled TensorFlow / MLflow / plotting into the serve image.
- Duplicate package lines; redundant `pip install lightgbm xgboost` in Docker.

### Fix applied

1. Split pinned deps:
   - `requirements-serve.txt` — FastAPI stack + sklearn/xgb/lgbm only
   - `requirements-train.txt` — `-r requirements-serve.txt` plus tqdm, matplotlib, seaborn, mlflow, tensorflow
   - `requirements.txt` — thin alias to train file for backward compatibility
2. Dockerfile installs **only** `requirements-serve.txt`; removed second lightgbm/xgboost install.
3. Base image bumped to `python:3.12-slim` to match pinned sklearn 1.7 / local env.

### Acceptance

- [x] Serve requirements exclude TensorFlow / MLflow / plotting.
- [x] Dockerfile copies serve-only requirements; no duplicate pip of boost libs.
- [x] Image builds (`iot-anomaly-serve:iss05`); `/health`, `/ready`, `/predict` smoke OK (~1.18 GB — still includes xgboost’s optional CUDA NCCL wheel; CPU-only slim further is a follow-up).

---

## ISS-06: Suspiciously high metrics — verify & fix

**Severity:** High (investigation + training correctness)  
**Status:** Done — see [`ISS06_VERIFICATION.md`](ISS06_VERIFICATION.md)  
**Where:** `scripts/verify_iss06.py`, `train_model.py`, `model_bundle.py`, `models/model_20260713_162252/`

### Outcome

Weighted ~99.9% accuracy on ACI is **largely real (suspect C)** under random stratified eval, not mainly fit-all leakage. Honest full retrain without ports still ≈99.7–99.9% weighted, but **macro F1 ≈0.90** because rare **ARP Spoofing** fails (test support=1).

| Suspect | Result |
|---------|--------|
| A fit-all before split | Closed in code; Δ≈0 on subsample |
| B Src/Dst Port | Dropped in release trainer; small Δ |
| C easy separation | **Primary** |
| D global-only metrics | Fixed via per-class artifacts |
| E temporal/group | Not pursued |

### Fix applied

1. Phases 1–2 verification harness + notes  
2. Canonical trainer: train-only preprocess, drop ports, classification reports  
3. New release bundle `model_20260713_162252` (`evaluation_protocol: train_only_preprocess_v1`)  
4. Historical `model_20250728_222231` kept (leaky protocol / ports)

### Acceptance

- [x] Verification notes recorded  
- [x] Preprocessor fit on train only  
- [x] Drop/keep policy documented (README + this file + verification log)  
- [x] Bundle includes classification reports + confusion matrices  
- [x] Honest-protocol release metrics published; old bundle marked historical  

---

## ISS-07: Duplicate / experimental training scripts

**Severity:** Low  
**Status:** Done  
**Where:** `train_model.py`, `experiments/`

### Problem

Near-duplicate trainers (`train_model.py`, `train_model_quantized.py`, `train_model_quantized_no_customclass.py`) increased drift risk (ISS-03). Quantization / TFLite (`quantize.py`) was experimental and not on the Docker serve path.

### Fix applied

1. Designated **one** canonical trainer: root `train_model.py` (formerly `train_model_quantized_no_customclass.py`), the only script that calls `model_bundle.save_bundle`.
2. Removed byte-identical duplicate `train_model_quantized.py`.
3. Moved legacy custom-preprocessor trainer and TFLite script to `experiments/` with explicit non-canonical headers.
4. README layout / canonical-paths table / training section updated to match.

### Acceptance

- [x] README “canonical paths” table matches remaining scripts.
- [x] No two trainers claim to produce the serve bundle format without matching `load_bundle`.

---

## ISS-08: API error handling & observability

**Severity:** Low  
**Status:** Done  
**Where:** `deploy_api.py`

### Problem

- Exceptions were logged then rethrown as a generic `"Prediction failed."` — clients could not distinguish bad input vs missing model vs contract errors.
- Originally no `/health` or `/ready` (added under ISS-02; kept here as acceptance).
- Dual Pydantic v1/v2 `InputData` shims after pins made the v1 path dead code.

### Fix applied

1. Unknown `model_name` → **404** with available model list in `detail`.
2. Feature transform / encode failures (`ValueError` / `KeyError` / `TypeError`) → **400** with actionable message (`Invalid feature payload: …`).
3. Missing / invalid body fields → FastAPI/Pydantic **422** (unchanged).
4. Confirmed `GET /health` (liveness) and `GET /ready` (artifacts loaded).
5. Dropped unused Pydantic v1 branch; serve pins `pydantic==2.10.6` (`requirements-serve.txt`).
6. Covered by `tests/test_iss08_api_errors.py`.

### Acceptance

- [x] Unknown model name → 404 with detail.
- [x] Schema mismatch → 422 (body) / 400 (transform) with actionable message.
- [x] `/health` returns 200 when process is up.

---

## ISS-09: `mcp/` path diverges from canonical serve

**Severity:** Low  
**Status:** Done  
**Where:** README, `.gitignore` (external MCP repo)

### Problem

An alternate FastAPI + wrapper used different contracts and dependencies. Sitting as an untracked nested clone under `mcp/` made it easy to confuse with production `deploy_api.py` + root `Dockerfile`, and invited silent dual maintenance.

### Fix applied

1. Documented production serve path at the top of the README: root `Dockerfile` → `deploy_api:app` only.
2. Replaced in-tree `mcp/` docs with an explicit **external project** section linking to  
   https://github.com/HimanshuSourav/MCP-Compliant-IoT-Network-Anomaly-Detection  
   and a side-by-side contract contrast table.
3. Gitignored `mcp/` so an optional local clone is not absorbed into this repo.
4. Removed `mcp/` from the repository layout tree.

### Acceptance

- [x] Contributors can tell which entrypoint Docker uses without reading two codepaths.
- [x] No silent dual maintenance requirement (MCP kept as a separate repo; not tracked here).

---

## Fix workflow

For each issue:

1. Set status → **In progress** in the summary table.
2. Implement the smallest fix that meets Acceptance.
3. Smoke-test train and/or serve as applicable.
4. Update this file (status → **Done**) and adjust baseline README only where behavior actually changed.
5. Pause for review before starting the next ID.

---

## Out of scope (for later runbook)

- CI/CD, model versioning/registry beyond timestamp folders  
- Auth / rate limiting on the API  
- GPU / production autoscaling  
- Pinning and publishing a hardened ops runbook (README roadmap item)
