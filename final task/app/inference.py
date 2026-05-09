from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


FEATURE_NAMES = ["area_m2", "rooms", "floor", "building_age"]
MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "linear_model.pkl"


def load_model(model_path: Path = MODEL_PATH):
    with model_path.open("rb") as file_obj:
        return pickle.load(file_obj)


def validate_features(features: Sequence[float]) -> pd.DataFrame:
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"Expected {len(FEATURE_NAMES)} features, got {len(features)}")

    arr = np.asarray(features, dtype=float)
    if np.isnan(arr).any():
        raise ValueError("Input contains NaN values")
    return pd.DataFrame([arr], columns=FEATURE_NAMES)


def predict_price(features: Sequence[float]) -> float:
    model = load_model()
    rows = validate_features(features)
    pred = model.predict(rows)[0]
    return float(pred)
