"""ISS-03: shared model_bundle contract against the checked-in artifacts."""

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

import deploy_api
from model_bundle import align_raw_features, load_bundle, transform_raw

BUNDLE_DIR = Path("models/model_20250728_222231")


def test_load_bundle_resolves_label_encoder_and_metadata():
    bundle = load_bundle(str(BUNDLE_DIR))
    assert "random_forest" in bundle.models
    assert bundle.label_encoder is not None
    assert len(bundle.label_classes) == 12
    assert len(bundle.input_feature_names) == 78
    assert bundle.metadata.get("label_encoder_classes")
    assert bundle.metadata.get("input_feature_names")
    assert "Benign" in bundle.label_classes


def test_align_and_transform_partial_request():
    bundle = load_bundle(str(BUNDLE_DIR))
    df = pd.DataFrame(
        [
            {
                "Total Fwd Packets": 10.0,  # plural alias
                "Total Backward Packets": 8.0,
                "Total Length of Fwd Packets": 1500.0,
                "Total Length of Bwd Packets": 1200.0,
                "Flow Duration": 1000000.0,
            }
        ]
    )
    X = transform_raw(bundle, df)
    assert X.shape[0] == 1
    assert X.shape[1] > 5


def test_predict_with_real_bundle_returns_label():
    # Use real artifacts via normal lifespan load
    with TestClient(deploy_api.app) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["n_classes"] == 12

        payload = {
            "Total Fwd Packet": 10.0,
            "Total Bwd packets": 8.0,
            "Total Length of Fwd Packet": 1500.0,
            "Total Length of Bwd Packet": 1200.0,
            "Flow Duration": 1000000.0,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["prediction"] in deploy_api.get_cached_bundle().label_classes
        assert body["model_used"] == "random_forest"
        assert 0.0 <= body["confidence"] <= 1.0
