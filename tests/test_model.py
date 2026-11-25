import pandas as pd

from src.data import FEATURES
from src.features import add_features
from src.navy import navy_body_fat_male


def test_navy_formula_known_value():
    # 178 cm, neck 38, waist 90 -> about 20 % body fat
    assert 18 < float(navy_body_fat_male(178, 38, 90)) < 22


def test_engineered_features():
    X = pd.DataFrame([[30, 80, 200, 38, 90, 99]], columns=FEATURES)
    out = add_features(X)
    assert round(out["bmi"][0], 1) == 20.0
    assert round(out["waist_to_height"][0], 2) == 0.45


def test_model_predicts_sensible_range(trained_model):
    lean = pd.DataFrame([[25, 70, 180, 39, 76, 92]], columns=FEATURES)
    heavy = pd.DataFrame([[50, 110, 175, 42, 118, 112]], columns=FEATURES)
    p_lean, p_heavy = trained_model.predict(lean)[0], trained_model.predict(heavy)[0]
    assert 3 < p_lean < 20
    assert 25 < p_heavy < 50
    assert p_heavy > p_lean
