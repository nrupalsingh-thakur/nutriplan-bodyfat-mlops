"""Test setup: point model/log folders to a temp dir and put a small trained model there."""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp())
os.environ["MODEL_DIR"] = str(TMP / "models")
os.environ["LOG_DIR"] = str(TMP / "logs")
os.environ["REPORT_DIR"] = str(TMP / "reports")

import joblib  # noqa: E402
import pytest  # noqa: E402

from src.config import load_params  # noqa: E402
from src.data import FEATURES, TARGET, prepare_data  # noqa: E402


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture(scope="session")
def clean_data(params, tmp_path_factory):
    p = {**params, "data": {**params["data"],
                            "clean_path": str(tmp_path_factory.mktemp("d") / "clean.csv")}}
    df, report = prepare_data(p, ROOT)
    return df, report


@pytest.fixture(scope="session", autouse=True)
def trained_model(clean_data):
    from src.train import build_pipeline
    df, report = clean_data
    pipe = build_pipeline("linear_regression", {}, 42).fit(df[FEATURES], df[TARGET])
    d = Path(os.environ["MODEL_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, d / "model.joblib")
    (d / "metadata.json").write_text(json.dumps({
        "model_name": "bodyfat-model", "model_version": 0, "model_type": "linear_regression",
        "run_id": "test", "trained_at": "test", "data_md5": report["data_md5"], "features": FEATURES,
        "metrics": {"test_mae": 3.5, "test_r2": 0.7}, "navy_baseline": {"test_mae": 4.0},
    }))
    stats = {c: {"mean": float(df[c].mean()), "std": float(df[c].std())} for c in FEATURES}
    (d / "training_stats.json").write_text(json.dumps(stats))
    return pipe
