import pandas as pd

from src.data import FEATURES, load_raw, to_metric, validate
from src.config import ROOT


def test_raw_file_has_252_rows(params):
    raw = load_raw(ROOT / params["data"]["raw_path"])
    assert len(raw) == 252
    assert raw.shape[1] == 15


def test_validation_removes_impossible_records(clean_data):
    df, report = clean_data
    reasons = " ".join(r["reason"] for r in report["removed"])
    assert "height_cm" in reasons          # the 74 cm tall man (row 42)
    assert "body_fat" in reasons           # the 0 % body fat man (row 182)
    assert report["rows_out"] == len(df)
    assert 240 <= len(df) < 252


def test_clean_data_has_no_missing_values_and_all_features(clean_data):
    df, _ = clean_data
    assert not df[FEATURES].isna().any().any()
    assert df["height_cm"].min() >= 140


def test_validate_catches_missing_and_duplicates(params):
    row = {"age": 30, "weight_kg": 80, "height_cm": 178, "neck_cm": 38, "waist_cm": 90,
           "hip_cm": 99, "density": 1.06, "body_fat": round(495 / 1.06 - 450, 1)}
    df = pd.DataFrame([row, row, {**row, "age": None}])
    clean, report = validate(df, params["validation"])
    assert len(clean) == 1
    assert report["rows_removed"] == 2


def test_units_are_converted(params):
    raw = load_raw(ROOT / params["data"]["raw_path"]).head(1)
    m = to_metric(raw)
    assert abs(m["weight_kg"][0] - raw["weight_lb"][0] * 0.4536) < 0.1
    assert abs(m["height_cm"][0] - raw["height_in"][0] * 2.54) < 0.1
