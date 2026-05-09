from pathlib import Path

import pytest

from app.inference import MODEL_PATH, predict_price, validate_features


def test_model_file_exists():
    assert Path(MODEL_PATH).exists()


def test_validate_features_shape():
    arr = validate_features([50, 2, 5, 10])
    assert arr.shape == (1, 4)


def test_validate_features_rejects_bad_length():
    with pytest.raises(ValueError):
        validate_features([50, 2, 5])


def test_predict_price_returns_float():
    prediction = predict_price([60, 2, 5, 8])
    assert isinstance(prediction, float)
