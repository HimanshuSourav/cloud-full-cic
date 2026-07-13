"""ISS-01 / ISS-03: request keys map to ACI / bundle feature names."""

from deploy_api import InputData
from model_bundle import REQUEST_COLUMN_ALIASES, normalize_request_columns


def test_normalize_legacy_and_plural_keys():
    legacy = {
        "Total_Fwd_Packets": 10.0,
        "Total_Backward_Packets": 8.0,
        "Total_Length_of_Fwd_Packets": 1500.0,
        "Total_Length_of_Bwd_Packets": 1200.0,
        "Flow_Duration": 1000000.0,
    }
    canonical = normalize_request_columns(legacy)
    assert canonical["Total Fwd Packet"] == 10.0
    assert canonical["Total Bwd packets"] == 8.0
    assert "Total_Fwd_Packets" not in canonical


def test_inputdata_aci_keys_to_feature_frame():
    payload = {
        "Total Fwd Packet": 10.0,
        "Total Bwd packets": 8.0,
        "Total Length of Fwd Packet": 1500.0,
        "Total Length of Bwd Packet": 1200.0,
        "Flow Duration": 1000000.0,
    }
    frame = InputData(**payload).to_feature_frame()
    assert list(frame.columns) == [
        "Total Fwd Packet",
        "Total Bwd packets",
        "Total Length of Fwd Packet",
        "Total Length of Bwd Packet",
        "Flow Duration",
    ]


def test_inputdata_legacy_underscore_keys_accepted():
    payload = {
        "Total_Fwd_Packets": 3.0,
        "Total_Backward_Packets": 1.0,
        "Total_Length_of_Fwd_Packets": 100.0,
        "Total_Length_of_Bwd_Packets": 50.0,
        "Flow_Duration": 500.0,
    }
    frame = InputData(**payload).to_feature_frame()
    assert frame.iloc[0]["Total Fwd Packet"] == 3.0
    assert frame.iloc[0]["Flow Duration"] == 500.0
    assert "Total_Fwd_Packets" not in frame.columns


def test_plural_aliases_in_map():
    assert REQUEST_COLUMN_ALIASES["Total Fwd Packets"] == "Total Fwd Packet"
    assert REQUEST_COLUMN_ALIASES["Total Backward Packets"] == "Total Bwd packets"
