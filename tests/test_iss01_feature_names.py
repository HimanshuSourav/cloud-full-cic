"""ISS-01: request keys map to train-time spaced feature names."""

from deploy_api import LEGACY_TO_CANONICAL, InputData, normalize_feature_keys


def test_normalize_legacy_keys():
    legacy = {
        "Total_Fwd_Packets": 10.0,
        "Total_Backward_Packets": 8.0,
        "Total_Length_of_Fwd_Packets": 1500.0,
        "Total_Length_of_Bwd_Packets": 1200.0,
        "Flow_Duration": 1000000.0,
    }
    canonical = normalize_feature_keys(legacy)
    assert set(canonical) == set(LEGACY_TO_CANONICAL.values())
    assert "Total_Fwd_Packets" not in canonical
    assert canonical["Total Fwd Packets"] == 10.0


def test_inputdata_spaced_keys_to_feature_frame():
    payload = {
        "Total Fwd Packets": 10.0,
        "Total Backward Packets": 8.0,
        "Total Length of Fwd Packets": 1500.0,
        "Total Length of Bwd Packets": 1200.0,
        "Flow Duration": 1000000.0,
    }
    frame = InputData(**payload).to_feature_frame()
    assert list(frame.columns) == [
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Duration",
    ]
    assert frame.iloc[0]["Total Fwd Packets"] == 10.0


def test_inputdata_legacy_underscore_keys_accepted():
    payload = {
        "Total_Fwd_Packets": 3.0,
        "Total_Backward_Packets": 1.0,
        "Total_Length_of_Fwd_Packets": 100.0,
        "Total_Length_of_Bwd_Packets": 50.0,
        "Flow_Duration": 500.0,
    }
    frame = InputData(**payload).to_feature_frame()
    assert "Total Fwd Packets" in frame.columns
    assert "Total_Fwd_Packets" not in frame.columns
    assert frame.iloc[0]["Flow Duration"] == 500.0
