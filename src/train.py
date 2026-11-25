"""The training pipeline: data -> features -> train -> track -> quality gate -> register -> export.

Run:  python -m src.train
"""
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import matplotlib
import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_DIR, ROOT, load_params
from src.data import FEATURES, TARGET, prepare_data
from src.features import feature_step
from src.navy import navy_body_fat_male

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL_TYPES = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def build_pipeline(name: str, hp: dict, seed: int) -> Pipeline:
    cls = MODEL_TYPES[name]
    if "random_state" in cls().get_params():
        hp = {**hp, "random_state": seed}
    return Pipeline([
        ("features", feature_step()),
        ("scale", StandardScaler()),
        ("model", cls(**hp)),
    ])


def metrics(y_true, y_pred) -> dict:
    return {
        "test_mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "test_rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "test_r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def scatter_plot(y_true, y_pred, title: str, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.7)
    lims = [0, 50]
    ax.plot(lims, lims, "--", color="grey")
    ax.set(xlim=lims, ylim=lims, xlabel="Measured body fat %", ylabel="Predicted body fat %", title=title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> int:
    params = load_params()
    seed = params["data"]["random_state"]
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["training"]["experiment_name"])
    print(f"MLflow tracking at: {tracking_uri}")

    # ---------- 1. Data ----------
    df, report = prepare_data(params, ROOT)
    print(f"\n[DATA] {report['rows_in']} rows read, {report['rows_removed']} removed by validation, "
          f"{report['rows_out']} kept  (data md5 {report['data_md5'][:10]})")
    for r in report["removed"]:
        print(f"   - row {r['row']}: {r['reason']}")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["data"]["test_size"], random_state=seed)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_DIR / "_tmp"
    tmp.mkdir(exist_ok=True)
    (tmp / "validation_report.json").write_text(json.dumps(report, indent=2))

    common_params = {"data_md5": report["data_md5"], "rows_train": len(X_train), "rows_test": len(X_test)}

    # ---------- 2. Baseline: the Navy formula NutriPlan uses today ----------
    navy_pred = navy_body_fat_male(X_test["height_cm"], X_test["neck_cm"], X_test["waist_cm"])
    navy = metrics(y_test, navy_pred)
    with mlflow.start_run(run_name="navy_formula_baseline"):
        mlflow.set_tag("kind", "baseline")
        mlflow.log_params(common_params)
        mlflow.log_metrics(navy)
        scatter_plot(y_test, navy_pred, "Navy formula", tmp / "pred_vs_actual.png")
        mlflow.log_artifact(str(tmp / "pred_vs_actual.png"))
    print(f"\n[BASELINE] Navy formula: MAE {navy['test_mae']}  RMSE {navy['test_rmse']}  R2 {navy['test_r2']}")

    # ---------- 3. Train + track every candidate ----------
    results = []
    cv = params["training"]["cv_folds"]
    print("\n[TRAIN] model                cv_mae  test_mae  test_rmse  test_r2")
    for name, hp in params["training"]["models"].items():
        hp = hp or {}
        pipe = build_pipeline(name, hp, seed)
        cv_mae = float(-cross_val_score(pipe, X_train, y_train, cv=cv,
                                        scoring="neg_mean_absolute_error").mean())
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        m = {"cv_mae": round(cv_mae, 3), **metrics(y_test, pred)}

        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tag("kind", "candidate")
            mlflow.log_params({"model_type": name, **hp, **common_params})
            mlflow.log_metrics(m)
            scatter_plot(y_test, pred, name, tmp / "pred_vs_actual.png")
            mlflow.log_artifact(str(tmp / "pred_vs_actual.png"))
            mlflow.log_artifact(str(tmp / "validation_report.json"))
            mlflow.sklearn.log_model(pipe, "model",
                                     signature=infer_signature(X_train, pipe.predict(X_train)),
                                     input_example=X_train.head(3))
            results.append({"name": name, "run_id": run.info.run_id, "pipe": pipe, **m})
        print(f"        {name:<20} {m['cv_mae']:>6}  {m['test_mae']:>8}  {m['test_rmse']:>9}  {m['test_r2']:>7}")

    # Pick the winner on cross-validation error (the test set stays an honest final check).
    best = min(results, key=lambda r: r["cv_mae"])
    print(f"\n[SELECT] best model by CV MAE: {best['name']} (test MAE {best['test_mae']})")

    # ---------- 4. Quality gate ----------
    gate = params["quality_gate"]
    checks = {
        f"test MAE {best['test_mae']} <= {gate['max_mae']}": best["test_mae"] <= gate["max_mae"],
    }
    if gate["must_beat_navy"]:
        checks[f"test MAE {best['test_mae']} < Navy MAE {navy['test_mae']}"] = best["test_mae"] < navy["test_mae"]
    print("[GATE]")
    for text, ok in checks.items():
        print(f"   {'PASS' if ok else 'FAIL'}  {text}")
    if not all(checks.values()):
        print("\nQuality gate FAILED - model NOT registered, current production model kept.")
        return 1

    # ---------- 5. Register in the MLflow Model Registry ----------
    reg_name = params["training"]["registered_model_name"]
    mv = mlflow.register_model(f"runs:/{best['run_id']}/model", reg_name)
    client = MlflowClient()
    client.set_registered_model_alias(reg_name, "production", mv.version)
    client.set_model_version_tag(reg_name, mv.version, "beats_navy_by",
                                 f"{navy['test_mae'] - best['test_mae']:.2f}")
    print(f"\n[REGISTRY] {reg_name} version {mv.version} -> alias 'production'")

    # ---------- 6. Export for the serving API ----------
    joblib.dump(best["pipe"], MODEL_DIR / "model.joblib")
    meta = {
        "model_name": reg_name,
        "model_version": int(mv.version),
        "model_type": best["name"],
        "run_id": best["run_id"],
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_md5": report["data_md5"],
        "features": FEATURES,
        "metrics": {k: best[k] for k in ("cv_mae", "test_mae", "test_rmse", "test_r2")},
        "navy_baseline": navy,
        "all_candidates": [{k: r[k] for k in ("name", "cv_mae", "test_mae", "test_r2")} for r in results],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    stats = {c: {"mean": float(X_train[c].mean()), "std": float(X_train[c].std())} for c in FEATURES}
    (MODEL_DIR / "training_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[EXPORT] {MODEL_DIR / 'model.joblib'} + metadata.json + training_stats.json")
    print("\nPipeline finished successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
