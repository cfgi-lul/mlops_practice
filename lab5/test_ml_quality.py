from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def _resolve_path(filename: str) -> Path:
    """Find artifacts saved from notebook execution."""
    candidates = [Path(filename), Path("lab5") / filename]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Artifact not found: {filename}")


def _load_artifacts():
    model_path = _resolve_path("linear_model.pkl")
    data_path = _resolve_path("datasets.npz")
    model = joblib.load(model_path)
    data = np.load(data_path)
    return model, data


def _metrics(model, X, y):
    y_pred = model.predict(X)
    return {
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
    }


def test_quality_on_training_clean_dataset():
    """Model should be accurate on clean dataset used for training."""
    model, data = _load_artifacts()
    score = _metrics(model, data["X1"], data["y1"])

    assert score["r2"] > 0.95
    assert score["rmse"] < 1.0


def test_noisy_dataset_has_significant_quality_drop():
    """Noisy dataset should show quality degradation vs training clean dataset."""
    model, data = _load_artifacts()

    clean_score = _metrics(model, data["X1"], data["y1"])
    noisy_score = _metrics(model, data["X_noisy"], data["y_noisy"])

    assert noisy_score["r2"] < clean_score["r2"] - 0.25
    assert noisy_score["rmse"] > clean_score["rmse"] * 5


def test_noisy_dataset_meets_clean_quality_requirements():
    """This test should fail and reveal the noisy data issue."""
    model, data = _load_artifacts()
    noisy_score = _metrics(model, data["X_noisy"], data["y_noisy"])

    # Same quality bar as for clean data: noisy dataset should violate it.
    assert noisy_score["r2"] > 0.95
    assert noisy_score["rmse"] < 1.0
