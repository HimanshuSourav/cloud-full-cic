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
| [ISS-05](#iss-05-bloated-serving-image--unpinned-deps) | Medium | Bloated serving image / unpinned deps | Open |
| [ISS-06](#iss-06-suspiciously-high-metrics--verify--fix) | High | ~99.9% metrics — verify leakage / eval inflation, then fix | Open |
| [ISS-07](#iss-07-duplicateexperimental-training-scripts) | Low | Duplicate / experimental training scripts | Open |
| [ISS-08](#iss-08-api-error-handling--observability) | Low | Opaque 500s; no health/ready endpoints | Open |
| [ISS-09](#iss-09-mcp-path-diverges-from-canonical-serve) | Low | `mcp/` alternate path diverges | Open |

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
**Where:** `model_bundle.py`, `deploy_api.py`, `train_model_quantized_no_customclass.py`, `models/model_20250728_222231/`

### Problem

Serving assumed a **custom** preprocessor API (`preprocessor.label_encoder`, …) while the checked-in bundle is a sklearn **`ColumnTransformer`** plus standalone **`label_encoder.joblib`**, with null metadata fields. Train and serve had diverged.

### Fix applied (Option B)

1. Added shared `model_bundle.py` with `load_bundle` / `save_bundle` / `transform_raw` (`contract`: `sklearn_column_transformer_v1`).
2. `deploy_api` loads a `ModelBundle` (preprocessor + label encoder + classifiers) and decodes via the standalone encoder.
3. Request keys remap to ACI raw names (`Total Fwd Packet`, `Total Bwd packets`, …); missing raw columns are aligned (NaN-filled) before transform.
4. `train_model_quantized_no_customclass.py` saves through `save_bundle` (populated metadata).
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
**Where:** `preproc_scale.py`, `train_model.py`, `train_model_quantized.py`, `mcp/preprocessor.py`

### Problem

Custom `DataPreprocessor` paths called `scaler.fit_transform` on the **first 10 000 rows only**, then `transform` on later batches. Mean/variance ignored most of the data.

### Fix applied

1. Added `preproc_scale.fit_transform_scaled`: **`scaler.fit(X)` on all rows**, then batch `transform`.
2. Wired into `train_model.py` and `train_model_quantized.py`.
3. Same fix inlined in `mcp/preprocessor.py`.
4. Canonical serve trainer (`train_model_quantized_no_customclass.py`) already fits via `ColumnTransformer` on the full matrix — unchanged.
5. Covered by `tests/test_iss04_scaler_fit.py`.

**Note:** Retrain is still required before custom-preprocessor metrics reflect this fix (see ISS-06). Bundles already produced with the sklearn path are unaffected by this scaler bug.

### Acceptance

- [x] Scaler `n_samples_seen_` matches full frame length (unit test).
- [ ] Full dataset retrain under custom preprocessor (optional; prefer ISS-06 honest eval on sklearn path).

---

## ISS-05: Bloated serving image & unpinned deps

**Severity:** Medium  
**Status:** Open  
**Where:** `requirements.txt`, `Dockerfile`

### Problem

- `requirements.txt` includes **training / experiment** stacks unused at serve time: `tensorflow`, `mlflow`, `matplotlib`, `seaborn`, etc.
- Packages are **unpinned**; builds are non-reproducible.
- Duplicates in the file (`pandas`, `scikit-learn`, `joblib` listed twice).
- Dockerfile also runs `pip install lightgbm xgboost` again after installing from requirements.

Serving only needs: FastAPI, uvicorn, pandas/numpy, joblib, scikit-learn, xgboost, lightgbm (and runtime `libgomp1`).

### Probable fix

1. Split deps: `requirements-train.txt` vs `requirements-serve.txt` (pinned versions).
2. Dockerfile `COPY` only serve requirements; drop TensorFlow/MLflow/plotting from the image.
3. Remove redundant second `pip install lightgbm xgboost` if already in serve requirements.
4. Optionally multi-stage copy of site-packages is already present — keep it once deps are lean.

### Acceptance

- Image builds with serve-only deps.
- Image size materially smaller than current training-heavy install.
- Container still serves `/predict` for RF / XGB / LGBM.

---

## ISS-06: Suspiciously high metrics — verify & fix

**Severity:** High (investigation + training correctness)  
**Status:** Open  
**Where:** `models/model_20250728_222231/metadata.json`, `train_model.py`, `train_model_quantized_no_customclass.py`

### Symptom

Checked-in bundle reports near-perfect scores on all three classifiers:

| Model | Accuracy (metadata) |
|-------|---------------------|
| random_forest | ~0.9996 |
| xgboost | ~0.9988 |
| lightgbm | ~0.9995 |

That is a **suspect case**: either the task is truly easy with these features, or (more likely) evaluation is inflated by leakage / protocol mistakes. We should **verify first**, then fix whatever the evidence supports—not blindly lower accuracy.

### Concrete suspects (code-backed)

| Suspect | Evidence in repo | Why it inflates metrics |
|---------|------------------|-------------------------|
| **A. Fit preprocessor on full dataset before split** | `train_model*.py`: `preprocess` / `ColumnTransformer.fit_transform(X)` runs on **all rows**, then `train_test_split` | Scaler mean/variance and one-hot categories see test rows → optimistic test scores |
| **B. Identifier / high-cardinality leftovers** | Drop list removes `Flow ID`, `Src IP`, `Dst IP`, `Timestamp`, rates — but **`Src Port` / `Dst Port`** (and similar) remain in MLflow input examples | Ports / hosts often proxy attack type in lab datasets → near-memorization |
| **C. Easy lab separation** | ACI-IoT-2023 is a controlled capture; many CIC-style features are strongly class-conditional | High accuracy can be real; must show with honest holdout |
| **D. Global accuracy only** | Trainer logs accuracy / macro-ish aggregates; no per-class report in model metadata | Rare attack classes may be wrong while majority “Benign” inflates accuracy |
| **E. Chunk / order artifacts** | CSV loaded in file order; random split mixes times | Related flows land in both sets (group leakage) |

Suspect **A** is the strongest code smell and should be fixed regardless of whether accuracy stays high after an honest re-eval.

### Verification plan (do this before / while fixing)

Work through these checks; record results under `docs/` or in the model `metadata.json` when done.

1. **Baseline reproduce**  
   Retrain with current code; confirm metrics still ≈99.9%. If not reproducible, note env/data drift.

2. **Leakage check A — fit on train only**  
   Split **raw** rows first (`train_test_split` on dataframe indices).  
   `fit` preprocessor on train only; `transform` train and test.  
   Retrain; compare test accuracy / F1 to baseline.  
   - Large drop → A confirmed.  
   - Still ~99.9% → look at B/C/E.

3. **Feature ablation B**  
   Retrain with an extended drop list, e.g. also remove:  
   `Src Port`, `Dst Port`, and any remaining ID-like or absolute address columns.  
   Optionally drop protocol if it is nearly label-deterministic.  
   Compare metrics and top feature importances (RF / LGBM `feature_importances_`).

4. **Per-class report D**  
   Save `classification_report` + confusion matrix for the test set.  
   Check whether minority attack classes are actually learned or only majority classes drive accuracy.

5. **Harder holdout C/E (optional but recommended)**  
   - Time-ordered split (train earlier flows, test later), **or**  
   - Group split by a flow/session key if one exists, **or**  
   - External held-out CSV shard never seen during fit.  
   Report this as the “honest” score next to the random stratified split.

6. **Sanity: shuffle-label control**  
   Train once with randomly shuffled labels; test accuracy should collapse toward chance.  
   If it stays high, the metric pipeline itself is wrong (bug), not leakage.

### Probable fixes (apply based on verification)

| If verified… | Fix |
|--------------|-----|
| **A** | Always `train_test_split` (or CV folds) on raw data **before** fitting scaler / encoder / imputer. Persist only train-fitted preprocessor. |
| **B** | Expand documented drop list; retrain; store `feature_names` + importances in metadata. |
| **D** | Log and save per-class metrics + confusion matrix into the model bundle / MLflow. Stop treating single accuracy as the headline. |
| **E** | Prefer time- or group-based split for the reported “release” metrics; keep random split only as a secondary number. |
| **C only** | Document that ACI features are highly separable; still ship honest holdout metrics and the shuffle-label sanity result. |

Also re-run after **ISS-04** (full scaler fit) so scaling bugs are not confounded with leakage results.

### Acceptance

- [ ] Verification notes recorded (which suspects confirmed / rejected).  
- [ ] Preprocessor is fit **only** on training folds/split (suspect A closed in code).  
- [ ] Drop / keep feature policy documented (README or this file).  
- [ ] Model bundle includes classification report (or equivalent) + confusion matrix artifact.  
- [ ] Reported “release” metrics come from the honest protocol (train-only fit + chosen holdout); old ~99.9% figures either reproduced under that protocol or explicitly marked obsolete.

---

## ISS-07: Duplicate / experimental training scripts

**Severity:** Low  
**Status:** Open  
**Where:** `train_model.py`, `train_model_quantized.py`, `train_model_quantized_no_customclass.py`, `quantize.py`

### Problem

Near-duplicate trainers increase drift risk (ISS-03). Quantization / TFLite (`quantize.py`) appears experimental and is not on the Docker serve path.

### Probable fix

1. After ISS-03, designate **one** canonical trainer; mark others deprecated in README or remove.
2. Keep experiments under a clearly named folder or branch only if still needed.

### Acceptance

- README “canonical paths” table matches remaining scripts.
- No two trainers claim to produce the serve bundle format without matching `load_bundle`.

---

## ISS-08: API error handling & observability

**Severity:** Low  
**Status:** Open  
**Where:** `deploy_api.py`

### Problem

- Exceptions are logged then rethrown as a generic `"Prediction failed."` — clients cannot distinguish bad input vs missing model vs contract errors.
- No `/health` or `/ready`.
- Pydantic v1-style `schema_extra` / `.dict()` may need updating depending on installed Pydantic major version.

### Probable fix

1. Map known failures to 4xx with clear messages (unknown `model_name`, schema/column mismatch).
2. Add health/ready (pairs with ISS-02).
3. Align Pydantic v1/v2 APIs intentionally and pin the version (pairs with ISS-05).

### Acceptance

- Invalid model name → 404 with detail.
- Schema mismatch → 422/400 with actionable message.
- `/health` returns 200 when process is up.

---

## ISS-09: `mcp/` path diverges from canonical serve

**Severity:** Low  
**Status:** Open  
**Where:** `mcp/` (nested layout; spaced column names in its FastAPI schema)

### Problem

Alternate FastAPI + wrapper uses different contracts and dependencies. Easy to confuse with production `deploy_api.py` + root `Dockerfile`.

### Probable fix

1. Document clearly as non-canonical (already started in README).
2. Later: either sync to the same `load_bundle` helper or isolate as a separate package with its own docs.

### Acceptance

- Contributors can tell which entrypoint Docker uses without reading two codepaths.
- No silent dual maintenance requirement unless intentional.

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
