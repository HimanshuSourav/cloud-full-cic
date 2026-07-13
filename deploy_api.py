from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import logging
from typing import Any, Dict, Mapping, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Process-wide artifact cache (ISS-02): loaded once at startup, reused by /predict.
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "models")
DEFAULT_MODEL_NAME = os.environ.get("DEFAULT_MODEL_NAME", "random_forest")

_models: Dict[str, Any] = {}
_preprocessor: Any = None
_model_dir: Optional[str] = None
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

# Canonical CICFlowMeter / training column names (spaced).
# Legacy underscored API keys are remapped for backward compatibility (ISS-01).
LEGACY_TO_CANONICAL = {
    "Total_Fwd_Packets": "Total Fwd Packets",
    "Total_Backward_Packets": "Total Backward Packets",
    "Total_Length_of_Fwd_Packets": "Total Length of Fwd Packets",
    "Total_Length_of_Bwd_Packets": "Total Length of Bwd Packets",
    "Flow_Duration": "Flow Duration",
}

PREDICT_EXAMPLE = {
    "Total Fwd Packets": 10.0,
    "Total Backward Packets": 8.0,
    "Total Length of Fwd Packets": 1500.0,
    "Total Length of Bwd Packets": 1200.0,
    "Flow Duration": 1000000.0,
}


def normalize_feature_keys(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Map legacy underscored keys to train-time spaced column names."""
    return {LEGACY_TO_CANONICAL.get(key, key): value for key, value in payload.items()}


try:
    from pydantic import ConfigDict, model_validator

    class InputData(BaseModel):
        """IoT flow features; JSON keys match train-time spaced column names."""

        model_config = ConfigDict(
            populate_by_name=True,
            json_schema_extra={"example": PREDICT_EXAMPLE},
        )

        total_fwd_packets: float = Field(
            ..., alias="Total Fwd Packets", description="Total number of forward packets"
        )
        total_backward_packets: float = Field(
            ...,
            alias="Total Backward Packets",
            description="Total number of backward packets",
        )
        total_length_of_fwd_packets: float = Field(
            ...,
            alias="Total Length of Fwd Packets",
            description="Total size of forward packets",
        )
        total_length_of_bwd_packets: float = Field(
            ...,
            alias="Total Length of Bwd Packets",
            description="Total size of backward packets",
        )
        flow_duration: float = Field(
            ..., alias="Flow Duration", description="Duration of the flow in microseconds"
        )

        @model_validator(mode="before")
        @classmethod
        def _normalize_legacy_keys(cls, values: Any) -> Any:
            if isinstance(values, Mapping):
                return normalize_feature_keys(values)
            return values

        def to_feature_frame(self) -> pd.DataFrame:
            raw = self.model_dump(by_alias=True)
            return pd.DataFrame([normalize_feature_keys(raw)])

except ImportError:
    from pydantic import root_validator

    class InputData(BaseModel):
        """IoT flow features; JSON keys match train-time spaced column names."""

        total_fwd_packets: float = Field(
            ..., alias="Total Fwd Packets", description="Total number of forward packets"
        )
        total_backward_packets: float = Field(
            ...,
            alias="Total Backward Packets",
            description="Total number of backward packets",
        )
        total_length_of_fwd_packets: float = Field(
            ...,
            alias="Total Length of Fwd Packets",
            description="Total size of forward packets",
        )
        total_length_of_bwd_packets: float = Field(
            ...,
            alias="Total Length of Bwd Packets",
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
                return normalize_feature_keys(values)
            return values

        def to_feature_frame(self) -> pd.DataFrame:
            raw = self.dict(by_alias=True)
            return pd.DataFrame([normalize_feature_keys(raw)])


class ModelDeployment:
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def load_latest_model(self):
        """Load the latest saved model and preprocessor from disk."""
        try:
            model_folders = [
                f
                for f in os.listdir(self.base_path)
                if os.path.isdir(os.path.join(self.base_path, f))
            ]
            if not model_folders:
                raise FileNotFoundError("No saved models found in the model directory.")

            model_folders.sort(reverse=True)
            latest_model_dir = os.path.join(self.base_path, model_folders[0])
            logging.info(f"Loading models from {latest_model_dir}")

            models = {}
            for f in os.listdir(latest_model_dir):
                if f.endswith(".joblib") and f != "preprocessor.joblib":
                    model_name = f.split(".")[0]
                    # Standalone label encoder is not a classifier (see ISS-03).
                    if model_name == "label_encoder":
                        continue
                    model_path = os.path.join(latest_model_dir, f)
                    models[model_name] = joblib.load(model_path)
                    logging.info(f"Loaded {model_name} model from {model_path}")

            preprocessor_path = os.path.join(latest_model_dir, "preprocessor.joblib")
            preprocessor = joblib.load(preprocessor_path)
            logging.info(f"Loaded preprocessor from {preprocessor_path}")

            return models, preprocessor, latest_model_dir
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise


def load_artifacts_into_cache(base_path: str = MODEL_BASE_PATH) -> None:
    """Load artifacts once into module-level cache (idempotent until process exits)."""
    global _models, _preprocessor, _model_dir, _ready, _load_error
    try:
        models, preprocessor, model_dir = ModelDeployment(base_path).load_latest_model()
        _models = models
        _preprocessor = preprocessor
        _model_dir = model_dir
        _ready = True
        _load_error = None
        logging.info(
            "Artifacts cached for serving (dir=%s, models=%s)",
            model_dir,
            sorted(models.keys()),
        )
    except Exception as e:
        _models = {}
        _preprocessor = None
        _model_dir = None
        _ready = False
        _load_error = str(e)
        logging.error("Failed to cache artifacts at startup: %s", e)


def get_cached_artifacts():
    """Return (models, preprocessor) from the startup cache."""
    return _models, _preprocessor


@app.get("/health")
async def health():
    """Liveness: process is up (does not require models loaded)."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness: model artifacts are loaded into memory."""
    if not _ready:
        raise HTTPException(
            status_code=503,
            detail=_load_error or "Model artifacts not loaded",
        )
    return {
        "status": "ready",
        "model_dir": _model_dir,
        "models": sorted(_models.keys()),
    }


@app.post("/predict")
async def predict(input_data: InputData, model_name: Optional[str] = None):
    """FastAPI endpoint for making predictions (uses startup-cached artifacts)."""
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    try:
        if not _ready or _preprocessor is None:
            raise HTTPException(
                status_code=503,
                detail=_load_error or "Model artifacts not loaded",
            )

        models, preprocessor = get_cached_artifacts()
        model = models.get(model_name)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found.")

        input_df = input_data.to_feature_frame()

        if hasattr(preprocessor, "get_feature_names_out"):
            logging.debug(
                "Feature names after preprocessing: %s",
                preprocessor.get_feature_names_out(),
            )

        X_input = preprocessor.transform(input_df)

        prediction_numeric = model.predict(X_input)
        prediction_proba = model.predict_proba(X_input)

        label_encoder = getattr(preprocessor, "label_encoder", None)
        if label_encoder is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Preprocessor has no label_encoder "
                    "(train/serve contract drift; see ISS-03)."
                ),
            )

        attack_type = label_encoder.inverse_transform(prediction_numeric)[0]

        class_probabilities = {
            label: float(prob)
            for label, prob in zip(label_encoder.classes_, prediction_proba[0])
        }

        return {
            "prediction": attack_type,
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
