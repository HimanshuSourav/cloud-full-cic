"""ISS-02: artifacts load once at startup; /predict reuses the cache."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import deploy_api


def test_normalize_still_works_with_iss02_module():
    # Smoke: ISS-01 helpers remain importable after lifespan refactor.
    assert deploy_api.normalize_feature_keys({"Total_Fwd_Packets": 1})[
        "Total Fwd Packets"
    ] == 1


def test_startup_loads_once_predict_reuses_cache():
    fake_model = MagicMock()
    fake_model.predict.return_value = [0]
    fake_model.predict_proba.return_value = [[0.9, 0.1]]

    fake_pre = MagicMock()
    fake_pre.transform.return_value = [[0.0]]
    fake_pre.label_encoder.inverse_transform.return_value = ["Benign"]
    fake_pre.label_encoder.classes_ = ["Benign", "Attack"]
    with patch.object(
        deploy_api.ModelDeployment,
        "load_latest_model",
        return_value=(
            {"random_forest": fake_model},
            fake_pre,
            "models/fake_model_dir",
        ),
    ) as load_mock:
        with TestClient(deploy_api.app) as client:
            assert load_mock.call_count == 1
            assert client.get("/health").status_code == 200
            ready = client.get("/ready")
            assert ready.status_code == 200
            assert ready.json()["models"] == ["random_forest"]

            payload = {
                "Total Fwd Packets": 10.0,
                "Total Backward Packets": 8.0,
                "Total Length of Fwd Packets": 1500.0,
                "Total Length of Bwd Packets": 1200.0,
                "Flow Duration": 1000000.0,
            }
            r1 = client.post("/predict", json=payload)
            r2 = client.post("/predict", json=payload)
            assert r1.status_code == 200
            assert r2.status_code == 200
            assert r1.json()["prediction"] == "Benign"
            # Still only the startup load — not per request.
            assert load_mock.call_count == 1
