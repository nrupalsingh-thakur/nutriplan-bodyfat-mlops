"""Stage 5 - Serving: REST API for the production model.

Run:  uvicorn api.main:app --port 8000      (docs at /docs)
"""
import csv
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from src.config import LOG_DIR, MODEL_DIR
from src.data import FEATURES
from src.monitor import drift_report
from src.navy import navy_body_fat_male

app = FastAPI(
    title="NutriPlan Body Fat Prediction API",
    description="Predicts body fat % from tape measurements. Served model comes from the MLflow registry.",
    version="1.0.0",
)
STATIC = Path(__file__).parent / "static"
_lock = threading.Lock()
state: dict = {"model": None, "meta": None}


def load_model():
    path = MODEL_DIR / "model.joblib"
    if path.exists():
        state["model"] = joblib.load(path)
        state["meta"] = json.loads((MODEL_DIR / "metadata.json").read_text())


load_model()


class Measurements(BaseModel):
    age: int = Field(..., ge=18, le=90, examples=[30])
    weight_kg: float = Field(..., ge=40, le=200, examples=[80])
    height_cm: float = Field(..., ge=140, le=210, examples=[178])
    neck_cm: float = Field(..., ge=25, le=60, examples=[38])
    waist_cm: float = Field(..., ge=55, le=160, examples=[90])
    hip_cm: float = Field(..., ge=70, le=160, examples=[99])

    @model_validator(mode="after")
    def waist_bigger_than_neck(self):
        if self.waist_cm <= self.neck_cm:
            raise ValueError("waist_cm must be larger than neck_cm")
        return self


def log_prediction(row: dict):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    f = LOG_DIR / "predictions.csv"
    with _lock:
        new = not f.exists()
        with open(f, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health():
    return {"status": "ok" if state["model"] else "no_model", "model_loaded": state["model"] is not None}


@app.get("/model-info")
def model_info():
    if not state["meta"]:
        raise HTTPException(503, "No model loaded yet. Run the training pipeline first.")
    return state["meta"]


@app.post("/predict")
def predict(m: Measurements):
    if not state["model"]:
        raise HTTPException(503, "No model loaded yet. Run the training pipeline first.")
    x = pd.DataFrame([m.model_dump()])[FEATURES]
    pred = float(state["model"].predict(x)[0])
    pred = min(max(pred, 3.0), 50.0)
    navy = float(navy_body_fat_male(m.height_cm, m.neck_cm, m.waist_cm))
    meta = state["meta"]
    log_prediction({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **m.model_dump(),
        "prediction": round(pred, 2),
        "model_version": meta["model_version"],
    })
    return {
        "body_fat_pct": round(pred, 1),
        "expected_error_pct": meta["metrics"]["test_mae"],
        "navy_formula_pct": round(navy, 1),
        "model_name": meta["model_name"],
        "model_version": meta["model_version"],
        "model_type": meta["model_type"],
    }


@app.get("/monitoring/drift")
def monitoring_drift():
    if not state["meta"]:
        raise HTTPException(503, "No model loaded yet.")
    return drift_report()


@app.post("/admin/reload-model")
def reload_model():
    """Pick up a newly trained model without restarting the server."""
    load_model()
    return health()
