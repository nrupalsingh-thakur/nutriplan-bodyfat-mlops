"""Stage 1 - Data: ingest the raw file, convert units, validate and clean.

Raw source: "Body Fat" dataset (Johnson 1996, StatLib, Carnegie Mellon University).
252 adult men; body fat measured by underwater weighing (the lab gold standard).
"""
import hashlib
from pathlib import Path

import pandas as pd

RAW_COLUMNS = [
    "density", "body_fat", "age", "weight_lb", "height_in", "neck", "chest",
    "abdomen", "hip", "thigh", "knee", "ankle", "biceps", "forearm", "wrist",
]

# The six inputs NutriPlan already asks the user for (tape measure + scale).
FEATURES = ["age", "weight_kg", "height_cm", "neck_cm", "waist_cm", "hip_cm"]
TARGET = "body_fat"


def file_md5(path: Path) -> str:
    """Fingerprint of the data file. Logged with every run = simple data versioning."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    """Read the StatLib text file: keep only lines that hold exactly 15 numbers."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) != len(RAW_COLUMNS):
            continue
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def to_metric(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert pounds/inches to kg/cm and rename abdomen -> waist (NutriPlan naming)."""
    return pd.DataFrame({
        "age": raw["age"],
        "weight_kg": (raw["weight_lb"] * 0.453592).round(1),
        "height_cm": (raw["height_in"] * 2.54).round(1),
        "neck_cm": raw["neck"],
        "waist_cm": raw["abdomen"],
        "hip_cm": raw["hip"],
        "density": raw["density"],
        "body_fat": raw["body_fat"],
    })


def validate(df: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, dict]:
    """Apply the validation rules. Returns the clean data and a report of what was removed."""
    report = {"rows_in": int(len(df)), "removed": []}
    keep = pd.Series(True, index=df.index)

    missing = df.isna().any(axis=1)
    for i in df.index[missing]:
        report["removed"].append({"row": int(i) + 1, "reason": "missing value"})
    keep &= ~missing

    dup = df.duplicated()
    for i in df.index[dup & keep]:
        report["removed"].append({"row": int(i) + 1, "reason": "duplicate record"})
    keep &= ~dup

    for col, (lo, hi) in rules["ranges"].items():
        bad = ~df[col].between(lo, hi) & keep
        for i in df.index[bad]:
            report["removed"].append({
                "row": int(i) + 1,
                "reason": f"{col}={df.at[i, col]} outside [{lo}, {hi}]",
            })
        keep &= ~bad

    siri = 495 / df["density"] - 450
    bad = ((siri - df["body_fat"]).abs() > rules["density_tolerance"]) & keep
    for i in df.index[bad]:
        report["removed"].append({
            "row": int(i) + 1,
            "reason": f"body_fat={df.at[i, 'body_fat']} disagrees with density (Siri gives {siri[i]:.1f})",
        })
    keep &= ~bad

    clean = df[keep].reset_index(drop=True)
    report["rows_out"] = int(len(clean))
    report["rows_removed"] = report["rows_in"] - report["rows_out"]
    return clean, report


def prepare_data(params: dict, root: Path) -> tuple[pd.DataFrame, dict]:
    """Full data stage: ingest -> metric units -> validate -> save clean CSV."""
    raw_path = root / params["data"]["raw_path"]
    df = to_metric(load_raw(raw_path))
    clean, report = validate(df, params["validation"])
    report["data_md5"] = file_md5(raw_path)

    out = root / params["data"]["clean_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False)
    return clean, report
