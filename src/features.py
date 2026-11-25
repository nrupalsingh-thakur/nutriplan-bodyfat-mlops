"""Stage 2 - Features: the same transformation runs in training and in the API.

It is placed inside the scikit-learn Pipeline, so the saved model file contains it.
That removes "training/serving skew" - a classic production bug.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from src.data import FEATURES


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(X, columns=FEATURES).copy()
    X["bmi"] = X["weight_kg"] / (X["height_cm"] / 100) ** 2
    X["waist_to_height"] = X["waist_cm"] / X["height_cm"]
    # The Navy formula is built on log(waist - neck); give the model the same signal.
    X["log_waist_minus_neck"] = np.log10((X["waist_cm"] - X["neck_cm"]).clip(lower=1))
    return X


def feature_step() -> FunctionTransformer:
    return FunctionTransformer(add_features, validate=False)
