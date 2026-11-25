import pytest
from fastapi.testclient import TestClient

from api.main import app, load_model

client = TestClient(app)


@pytest.fixture(autouse=True, scope="module")
def _load(trained_model):
    load_model()   # the test model is written by conftest.py before this runs


GOOD = {"age": 30, "weight_kg": 80, "height_cm": 178, "neck_cm": 38, "waist_cm": 90, "hip_cm": 99}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["model_name"] == "bodyfat-model"


def test_predict_valid():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    d = r.json()
    assert 3 <= d["body_fat_pct"] <= 50
    assert "navy_formula_pct" in d


def test_predict_rejects_impossible_input():
    assert client.post("/predict", json={**GOOD, "height_cm": 40}).status_code == 422
    assert client.post("/predict", json={**GOOD, "waist_cm": 30, "neck_cm": 38}).status_code == 422
    assert client.post("/predict", json={"age": 30}).status_code == 422


def test_drift_detected_for_shifted_users():
    heavy = {**GOOD, "age": 65, "weight_kg": 120, "waist_cm": 125, "hip_cm": 120}
    for _ in range(25):
        client.post("/predict", json=heavy)
    d = client.get("/monitoring/drift").json()
    assert d["status"] == "drift"
    assert "weight_kg" in d["drifted_features"]
