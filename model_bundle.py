"""Shared train/serve model bundle contract (ISS-03).

Canonical artifact layout under ``models/model_<timestamp>/``:

- ``preprocessor.joblib`` — sklearn ``ColumnTransformer`` (or custom with ``transform``)
- ``label_encoder.joblib`` — standalone ``LabelEncoder`` (required for serve decode)
- ``<model_name>.joblib`` — classifiers (random_forest, xgboost, lightgbm, …)
- ``metadata.json`` — metrics + ``input_feature_names`` + ``label_encoder_classes``
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CONTRACT_ID = "sklearn_column_transformer_v1"

# Map common API / plural / underscore names → ACI-IoT-2023 column names used
# by the checked-in ColumnTransformer (singular Packet, "Total Bwd packets", …).
REQUEST_COLUMN_ALIASES = {
    "Total_Fwd_Packets": "Total Fwd Packet",
    "Total Fwd Packets": "Total Fwd Packet",
    "Total Fwd Packet": "Total Fwd Packet",
    "Total_Backward_Packets": "Total Bwd packets",
    "Total Backward Packets": "Total Bwd packets",
    "Total Bwd packets": "Total Bwd packets",
    "Total_Length_of_Fwd_Packets": "Total Length of Fwd Packet",
    "Total Length of Fwd Packets": "Total Length of Fwd Packet",
    "Total Length of Fwd Packet": "Total Length of Fwd Packet",
    "Total_Length_of_Bwd_Packets": "Total Length of Bwd Packet",
    "Total Length of Bwd Packets": "Total Length of Bwd Packet",
    "Total Length of Bwd Packet": "Total Length of Bwd Packet",
    "Flow_Duration": "Flow Duration",
    "Flow Duration": "Flow Duration",
}

CLASSIFIER_SKIP = {"preprocessor", "label_encoder"}


@dataclass
class ModelBundle:
    """In-memory serving bundle."""

    model_dir: str
    models: Dict[str, Any]
    preprocessor: Any
    label_encoder: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_feature_names: List[str] = field(default_factory=list)

    @property
    def label_classes(self) -> List[str]:
        classes = getattr(self.label_encoder, "classes_", None)
        if classes is not None:
            return [str(c) for c in classes]
        meta = self.metadata.get("label_encoder_classes") or []
        return list(meta)


def normalize_request_columns(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Remap request keys to ACI / bundle input column names."""
    out: Dict[str, Any] = {}
    for key, value in payload.items():
        out[REQUEST_COLUMN_ALIASES.get(key, key)] = value
    return out


def align_raw_features(
    df: pd.DataFrame,
    expected_columns: Sequence[str],
    fill_value: float = np.nan,
) -> pd.DataFrame:
    """Ensure DataFrame has exactly the preprocessor's raw input columns.

    Missing columns are filled (default NaN so imputers can run). Extra columns
    are dropped. Column order matches ``expected_columns``.
    """
    aligned = df.copy()
    aligned.columns = [
        REQUEST_COLUMN_ALIASES.get(str(c), str(c)) for c in aligned.columns
    ]
    # If aliasing collided, keep first occurrence
    aligned = aligned.loc[:, ~aligned.columns.duplicated()]

    for col in expected_columns:
        if col not in aligned.columns:
            aligned[col] = fill_value

    return aligned.reindex(columns=list(expected_columns))


def _read_metadata(model_dir: str) -> Dict[str, Any]:
    path = os.path.join(model_dir, "metadata.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_label_encoder(model_dir: str, preprocessor: Any) -> Any:
    le_path = os.path.join(model_dir, "label_encoder.joblib")
    if os.path.isfile(le_path):
        return joblib.load(le_path)
    if hasattr(preprocessor, "label_encoder"):
        return preprocessor.label_encoder
    raise FileNotFoundError(
        f"No label_encoder.joblib in {model_dir} and preprocessor has no "
        "label_encoder attribute"
    )


def _resolve_input_feature_names(
    preprocessor: Any, metadata: Mapping[str, Any]
) -> List[str]:
    if metadata.get("input_feature_names"):
        return list(metadata["input_feature_names"])
    names_in = getattr(preprocessor, "feature_names_in_", None)
    if names_in is not None:
        return [str(c) for c in names_in]
    # Custom preprocessor may store transformed names only; raw alignment skipped.
    feature_names = getattr(preprocessor, "feature_names", None)
    if feature_names is not None:
        return [str(c) for c in feature_names]
    return []


def latest_model_dir(base_path: str = "models") -> str:
    if not os.path.isdir(base_path):
        raise FileNotFoundError(f"Model base path not found: {base_path}")
    model_folders = [
        f
        for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f)) and f.startswith("model_")
    ]
    if not model_folders:
        raise FileNotFoundError(f"No saved models found in {base_path}")
    model_folders.sort(reverse=True)
    return os.path.join(base_path, model_folders[0])


def load_bundle(model_dir: str) -> ModelBundle:
    """Load a model directory using the canonical serve contract."""
    preprocessor_path = os.path.join(model_dir, "preprocessor.joblib")
    if not os.path.isfile(preprocessor_path):
        raise FileNotFoundError(f"Missing preprocessor.joblib in {model_dir}")

    preprocessor = joblib.load(preprocessor_path)
    label_encoder = _resolve_label_encoder(model_dir, preprocessor)
    metadata = _read_metadata(model_dir)

    models: Dict[str, Any] = {}
    for fname in os.listdir(model_dir):
        if not fname.endswith(".joblib"):
            continue
        name = fname[: -len(".joblib")]
        if name in CLASSIFIER_SKIP:
            continue
        models[name] = joblib.load(os.path.join(model_dir, fname))
        logger.info("Loaded classifier %s from %s", name, model_dir)

    input_feature_names = _resolve_input_feature_names(preprocessor, metadata)

    # Fill null metadata fields from live artifacts (legacy bundles).
    if not metadata.get("label_encoder_classes"):
        metadata["label_encoder_classes"] = [
            str(c) for c in getattr(label_encoder, "classes_", [])
        ]
    if not metadata.get("input_feature_names") and input_feature_names:
        metadata["input_feature_names"] = list(input_feature_names)
    metadata.setdefault("contract", CONTRACT_ID)

    return ModelBundle(
        model_dir=model_dir,
        models=models,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        metadata=metadata,
        input_feature_names=input_feature_names,
    )


def load_latest_bundle(base_path: str = "models") -> ModelBundle:
    return load_bundle(latest_model_dir(base_path))


def save_bundle(
    *,
    models: Mapping[str, Any],
    preprocessor: Any,
    label_encoder: Any,
    results: Mapping[str, Any],
    base_path: str = "models",
    feature_names: Optional[Iterable[str]] = None,
    input_feature_names: Optional[Iterable[str]] = None,
    compress: Any = ("zlib", 3),
    evaluation_protocol: str = CONTRACT_ID,
    dropped_columns: Optional[Iterable[str]] = None,
    classification_reports: Optional[Mapping[str, Any]] = None,
    confusion_matrices: Optional[Mapping[str, Any]] = None,
) -> str:
    """Persist artifacts in the canonical layout; return save directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_path, f"model_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    for model_name, model in models.items():
        joblib.dump(
            model,
            os.path.join(save_dir, f"{model_name}.joblib"),
            compress=compress,
        )

    joblib.dump(
        preprocessor,
        os.path.join(save_dir, "preprocessor.joblib"),
        compress=compress,
    )
    joblib.dump(
        label_encoder,
        os.path.join(save_dir, "label_encoder.joblib"),
        compress=compress,
    )

    resolved_input = (
        list(input_feature_names)
        if input_feature_names is not None
        else _resolve_input_feature_names(preprocessor, {})
    )
    resolved_features = (
        list(feature_names)
        if feature_names is not None
        else None
    )

    if classification_reports:
        with open(
            os.path.join(save_dir, "classification_reports.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(classification_reports, f, indent=2)

    if confusion_matrices:
        with open(
            os.path.join(save_dir, "confusion_matrices.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(confusion_matrices, f, indent=2)

    # Strip heavy nested reports from results if present (keep scalar metrics).
    clean_results = {}
    for model_name, metrics in dict(results).items():
        if isinstance(metrics, Mapping):
            clean_results[model_name] = {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in metrics.items()
                if k
                not in {
                    "classification_report",
                    "confusion_matrix",
                }
            }
        else:
            clean_results[model_name] = metrics

    metadata = {
        "timestamp": timestamp,
        "results": clean_results,
        "contract": CONTRACT_ID,
        "evaluation_protocol": evaluation_protocol,
        "dropped_columns": list(dropped_columns) if dropped_columns is not None else None,
        "input_feature_names": resolved_input,
        "feature_names": resolved_features,
        "label_encoder_classes": [str(c) for c in label_encoder.classes_],
        "compression": "zlib level 3",
        "notes": (
            "ISS-06: train-only preprocessor fit; see docs/ISS06_VERIFICATION.md. "
            "Checked-in ~99.9% leaky-protocol scores remain historically accurate on ACI "
            "but use evaluation_protocol for release comparisons."
        ),
    }
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    logger.info("Saved model bundle to %s", save_dir)
    return save_dir


def transform_raw(bundle: ModelBundle, raw_df: pd.DataFrame) -> Any:
    """Align raw request/train columns then run preprocessor.transform.

    Returns a DataFrame with train-time ``feature_names`` when available so
    estimators fitted with named columns do not warn/error on name mismatch.
    """
    if bundle.input_feature_names:
        raw_df = align_raw_features(raw_df, bundle.input_feature_names)
    X = bundle.preprocessor.transform(raw_df)
    names = bundle.metadata.get("feature_names")
    n_cols = getattr(X, "shape", [None, None])[1]
    if names and n_cols is not None and len(names) == n_cols:
        # Prefer explicit train-time names (unprefixed) over get_feature_names_out.
        if not any(str(n).startswith(("num__", "cat__")) for n in names):
            return pd.DataFrame(X, columns=list(names))
        # Legacy metadata may have prefixed names; strip for sklearn estimators
        # that were fitted on the unprefixed training frame.
        stripped = [
            str(n).split("__", 1)[1] if "__" in str(n) else str(n) for n in names
        ]
        return pd.DataFrame(X, columns=stripped)
    return X
