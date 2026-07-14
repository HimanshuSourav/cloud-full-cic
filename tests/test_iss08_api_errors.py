"""ISS-08: actionable HTTP status codes for /predict and /health."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import deploy_api
from model_bundle import ModelBundle

CORE_PAYLOAD = {
    "Total Fwd Packet": 10.0,
    "Total Bwd packets": 8.0,
    "Total Length of Fwd Packet": 1500.0,
    "Total Length of Bwd Packet": 1200.0,
    "Flow Duration": 1000000.0,
}


def _fake_bundle():
    fake_model = MagicMock()
    fake_model.predict.return_value = [1]
    fake_model.predict_proba.return_value = [[0.1, 0.9]]

    fake_pre = MagicMock()
    fake_pre.transform.return_value = [[0.0] * 5]

    fake_le = MagicMock()
    fake_le.inverse_transform.return_value = ["Benign"]
    fake_le.classes_ = ["ARP Spoofing", "Benign"]

    return ModelBundle(
        model_dir="models/fake_model_dir",
        models={"random_forest": fake_model},
        preprocessor=fake_pre,
        label_encoder=fake_le,
        metadata={"contract": "sklearn_column_transformer_v1"},
        input_feature_names=list(CORE_PAYLOAD.keys()),
    )


def test_health_returns_200():
    with patch.object(deploy_api, "load_latest_bundle", return_value=_fake_bundle()):
        with TestClient(deploy_api.app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"


def test_unknown_model_returns_404_with_available():
    with patch.object(deploy_api, "load_latest_bundle", return_value=_fake_bundle()):
        with TestClient(deploy_api.app) as client:
            r = client.post("/predict?model_name=does_not_exist", json=CORE_PAYLOAD)
            assert r.status_code == 404
            detail = r.json()["detail"]
            assert "does_not_exist" in detail
            assert "random_forest" in detail


def test_missing_required_field_returns_422():
    with patch.object(deploy_api, "load_latest_bundle", return_value=_fake_bundle()):
        with TestClient(deploy_api.app) as client:
            bad = dict(CORE_PAYLOAD)
            del bad["Flow Duration"]
            r = client.post("/predict", json=bad)
            assert r.status_code == 422


def test_transform_value_error_returns_400():
    with patch.object(deploy_api, "load_latest_bundle", return_value=_fake_bundle()):
        with patch.object(
            deploy_api, "transform_raw", side_effect=ValueError("column shape mismatch")
        ):
            with TestClient(deploy_api.app) as client:
                r = client.post("/predict", json=CORE_PAYLOAD)
                assert r.status_code == 400
                assert "column shape mismatch" in r.json()["detail"]
