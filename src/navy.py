"""Baseline: U.S. Navy tape formula for men (Hodgdon & Beckett 1984).

This is exactly what NutriPlan uses today (backend/app/services/body.py).
A new model is only worth deploying if it beats this baseline.
"""
import numpy as np


def navy_body_fat_male(height_cm, neck_cm, waist_cm):
    height_cm, neck_cm, waist_cm = map(np.asarray, (height_cm, neck_cm, waist_cm))
    return 495 / (1.0324 - 0.19077 * np.log10(waist_cm - neck_cm)
                  + 0.15456 * np.log10(height_cm)) - 450
