"""Stage 6 - Monitoring: is the live data still like the training data?

The API writes every prediction to logs/predictions.csv.
This compares the average of each input in the log with the training average.
If a mean moved more than `drift_threshold` standard deviations -> DRIFT -> retrain.

Run:  python -m src.monitor
"""
import json
import sys

import pandas as pd

from src.config import LOG_DIR, MODEL_DIR, REPORT_DIR, load_params
from src.data import FEATURES


def drift_report(params: dict | None = None) -> dict:
    params = params or load_params()
    cfg = params["monitoring"]
    log_file = LOG_DIR / "predictions.csv"
    if not log_file.exists():
        return {"status": "no_data", "message": "No predictions logged yet.", "samples": 0}

    live = pd.read_csv(log_file)
    if len(live) < cfg["min_samples"]:
        return {"status": "not_enough_data", "samples": int(len(live)),
                "message": f"Need at least {cfg['min_samples']} predictions, have {len(live)}."}

    stats = json.loads((MODEL_DIR / "training_stats.json").read_text())
    features = {}
    for c in FEATURES:
        shift = abs(live[c].mean() - stats[c]["mean"]) / stats[c]["std"]
        features[c] = {
            "train_mean": round(stats[c]["mean"], 1),
            "live_mean": round(float(live[c].mean()), 1),
            "shift_in_std": round(float(shift), 2),
            "drift": bool(shift > cfg["drift_threshold"]),
        }
    drifted = [c for c, v in features.items() if v["drift"]]
    return {
        "status": "drift" if drifted else "ok",
        "samples": int(len(live)),
        "threshold_std": cfg["drift_threshold"],
        "drifted_features": drifted,
        "features": features,
        "advice": "Retrain the model with recent data." if drifted else "No action needed.",
    }


def main() -> int:
    rep = drift_report()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "drift_report.json").write_text(json.dumps(rep, indent=2))
    print(f"Status: {rep['status'].upper()}   samples: {rep['samples']}")
    for c, v in rep.get("features", {}).items():
        flag = "DRIFT" if v["drift"] else "ok"
        print(f"  {c:<10} train {v['train_mean']:>6}  live {v['live_mean']:>6}  "
              f"shift {v['shift_in_std']:>5} std  {flag}")
    if "advice" in rep:
        print(rep["advice"])
    print(f"Saved {REPORT_DIR / 'drift_report.json'}")
    return 2 if rep["status"] == "drift" else 0


if __name__ == "__main__":
    sys.exit(main())
