from pathlib import Path

import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_input.csv"
EXPECTED_COLUMNS = ["area_m2", "rooms", "floor", "building_age"]


def test_dataset_exists():
    assert DATA_PATH.exists()


def test_dataset_schema():
    df = pd.read_csv(DATA_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_dataset_has_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert not df.isna().any().any()


def test_value_ranges_are_reasonable():
    df = pd.read_csv(DATA_PATH)
    assert df["area_m2"].between(10, 500).all()
    assert df["rooms"].between(1, 10).all()
    assert df["floor"].between(1, 50).all()
    assert df["building_age"].between(0, 150).all()
