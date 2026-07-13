from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, Field
import pandas as pd
import os
import logging
from typing import Any, Dict, Mapping, Optional

from model_bundle import (
    ModelBundle,
    load_latest_bundle,
    normalize_request_columns,
    transform_raw,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Process-wide artifact cache (ISS-02): loaded once at startup, reused by /predict.
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "models")
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "random_forest")

_bundle: Optional[ModelBundle] = None
_ready: bool = False
_load_error: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once when the app starts."""
    load_artifacts_into_cache(MODEL_BASE_PATH)
    yield


app = FastAPI(
    title="IoT Network Anomaly Detection API",
    description="API for detecting network anomalies in IoT devices",
    version="1.0.0",
    lifespan=lifespan,
)

# Canonical JSON keys for the API docs: ACI-IoT / bundle raw column names (ISS-03).
# Legacy underscores and plural CIC spellings are remapped via normalize_request_columns.
PREDICT_EXAMPLE = {
    "Total Fwd Packet": 10.0,
    "Total Bwd packets": 8.0,
    "Total Length of Fwd Packet": 1500.0,
    "Total Length of Bwd Packet": 1200.0,
    "Flow Duration": 1000000.0,
}


try:
    from pydantic import ConfigDict, model_validator

    class InputData(BaseModel):
        """Core flow features; accepts ACI names and legacy plural/underscore aliases."""

        model_config = ConfigDict(
            populate_by_name=True,
            json_schema_extra={"example": PREDICT_EXAMPLE},
        )

        total_fwd_packet: float = Field(
            ..., alias="Total Fwd Packet", description="Total number of forward packets"
        )
        total_bwd_packets: float = Field(
            ...,
            alias="Total Bwd packets",
            description="Total number of backward packets",
        )
        total_length_of_fwd_packet: float = Field(
            ...,
            alias="Total Length of Fwd Packet",
            description="Total size of forward packets",
        )
        total_length_of_bwd_packet: float = Field(
            ...,
            alias="Total Length of Bwd Packet",
            description="Total size of backward packets",
        )
        flow_duration: float = Field(
            ..., alias="Flow Duration", description="Duration of the flow in microseconds"
        )

        @model_validator(mode="before")
        @classmethod
        def _normalize_legacy_keys(cls, values: Any) -> Any:
            if isinstance(values, Mapping):
                return normalize_request_columns(values)
            return values

        def to_feature_frame(self) -> pd.DataFrame:
            raw = normalize_request_columns(self.model_dump(by_alias=True))
            return pd.DataFrame([raw])

except ImportError:
    from pydantic import root_validator

    class InputData(BaseModel):
        """Core flow features; accepts ACI names and legacy plural/underscore aliases."""

        total_fwd_packet: float = Field(
            ..., alias="Total Fwd Packet", description="Total number of forward packets"
        )
        total_bwd_packets: float = Field(
            ...,
            alias="Total Bwd packets",
            description="Total number of backward packets",
        )
        total_length_of_fwd_packet: float = Field(
            ...,
            alias="Total Length of Fwd Packet",
            description="Total size of forward packets",
        )
        total_length_of_bwd_packet: float = Field(
            ...,
            alias="Total Length of Bwd Packet",
            description="Total size of backward packets",
        )
        flow_duration: float = Field(
            ..., alias="Flow Duration", description="Duration of the flow in microseconds"
        )

        class Config:
            allow_population_by_field_name = True
            schema_extra = {"example": PREDICT_EXAMPLE}

        @root_validator(pre=True)
        def _normalize_legacy_keys(cls, values: Any) -> Any:  # noqa: N805
            if isinstance(values, Mapping):
                return normalize_request_columns(values)
            return values

        def to_feature_frame(self) -> pd.DataFrame:
            raw = normalize_request_columns(self.dict(by_alias=True))
            return pd.DataFrame([raw])


def load_artifacts_into_cache(base_path: str = MODEL_BASE_PATH) -> None:
    """Load artifacts once into module-level cache (ISS-02 / ISS-03)."""
    global _bundle, _ready, _load_error
    try:
        bundle = load_latest_bundle(base_path)
        _bundle = bundle
        _ready = True
        _load_error = None
        logging.info(
            "Artifacts cached for serving (dir=%s, models=%s, contract=%s)",
            bundle.model_dir,
            sorted(bundle.models.keys()),
            bundle.metadata.get("contract"),
        )
    except Exception as e:
        _bundle = None
        _ready = False
        _load_error = str(e)
        logging.error("Failed to cache artifacts at startup: %s", e)


def get_cached_bundle() -> Optional[ModelBundle]:
    return _bundle


@app.get("/health")
async def health():
    """Liveness: process is up (does not require models loaded)."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness: model artifacts are loaded into memory."""
    if not _ready or _bundle is None:
        raise HTTPException(
            status_code=503,
            detail=_load_error or "Model artifacts not loaded",
        )
    return {
        "status": "ready",
        "model_dir": _bundle.model_dir,
        "models": sorted(_bundle.models.keys()),
        "contract": _bundle.metadata.get("contract"),
        "n_input_features": len(_bundle.input_feature_names),
        "n_classes": len(_bundle.label_classes),
    }


@app.post("/predict")
async def predict(input_data: InputData, model_name: Optional[str] = None):
    """FastAPI endpoint for making predictions (uses startup-cached artifacts)."""
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    try:
        if not _ready or _bundle is None:
            raise HTTPException(
                status_code=503,
                detail=_load_error or "Model artifacts not loaded",
            )

        bundle = _bundle
        model = bundle.models.get(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

        input_df = input_data.to_feature_frame()
        X_input = transform_raw(bundle, input_df)

        prediction_numeric = model.predict(X_input)
        prediction_proba = model.predict_proba(X_input)

        attack_type = bundle.label_encoder.inverse_transform(prediction_numeric)[0]
        class_probabilities = {
            label: float(prob)
            for label, prob in zip(bundle.label_encoder.classes_, prediction_proba[0])
        }

        return {
            "prediction": str(attack_type),
            "confidence": float(max(prediction_proba[0])),
            "class_probabilities": class_probabilities,
            "model_used": model_name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
