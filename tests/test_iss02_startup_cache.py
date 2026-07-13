"""ISS-02: artifacts load once at startup; /predict reuses the cache."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import deploy_api
from model_bundle import ModelBundle


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
        input_feature_names=[
            "Total Fwd Packet",
            "Total Bwd packets",
            "Total Length of Fwd Packet",
            "Total Length of Bwd Packet",
            "Flow Duration",
        ],
    )


def test_startup_loads_once_predict_reuses_cache():
    bundle = _fake_bundle()
    with patch.object(deploy_api, "load_latest_bundle", return_value=bundle) as load_mock:
        with TestClient(deploy_api.app) as client:
            assert load_mock.call_count == 1
            assert client.get("/health").status_code == 200
            ready = client.get("/ready")
            assert ready.status_code == 200
            assert ready.json()["models"] == ["random_forest"]

            payload = {
                "Total Fwd Packet": 10.0,
                "Total Bwd packets": 8.0,
                "Total Length of Fwd Packet": 1500.0,
                "Total Length of Bwd Packet": 1200.0,
                "Flow Duration": 1000000.0,
            }
            r1 = client.post("/predict", json=payload)
            r2 = client.post("/predict", json=payload)
            assert r1.status_code == 200, r1.text
            assert r2.status_code == 200
            assert r1.json()["prediction"] == "Benign"
            assert load_mock.call_count == 1
