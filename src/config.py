"""Load params.yaml and resolve project paths."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT / "logs"))
REPORT_DIR = Path(os.getenv("REPORT_DIR", ROOT / "reports"))


def load_params(path: Path | None = None) -> dict:
    with open(path or ROOT / "params.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
