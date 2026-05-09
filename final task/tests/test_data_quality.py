from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
EXPECTED_COLUMNS = ["area_m2", "rooms", "floor", "building_age", "price"]


def _datasets() -> list[Path]:
    return [TRAIN_PATH, TEST_PATH]


def test_dataset_exists():
    for path in _datasets():
        assert path.exists()


def test_dataset_schema():
    for path in _datasets():
        df = pd.read_csv(path)
        assert list(df.columns) == EXPECTED_COLUMNS


def test_dataset_has_no_missing_values():
    for path in _datasets():
        df = pd.read_csv(path)
        assert not df.isna().any().any()


def test_value_ranges_are_reasonable():
    for path in _datasets():
        df = pd.read_csv(path)
        assert df["area_m2"].between(10, 500).all()
        assert df["rooms"].between(1, 10).all()
        assert df["floor"].between(1, 50).all()
        assert df["building_age"].between(0, 150).all()
        assert df["price"].gt(0).all()


def test_noise_present_in_targets():
    train_df = pd.read_csv(TRAIN_PATH)
    # We expect natural scatter in synthetic prices due to normal noise.
    assert train_df["price"].std() > 0
