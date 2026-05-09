#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["area_m2", "rooms", "floor", "building_age"]
TARGET_COLUMN = "price"


def generate_dataset(size: int, noise_std: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    area = rng.uniform(25, 180, size)
    rooms = rng.integers(1, 6, size)
    floor = rng.integers(1, 21, size)
    building_age = rng.uniform(0, 50, size)
    noise = rng.normal(loc=0.0, scale=noise_std, size=size)

    # Synthetic target with normal distributed noise.
    price = 1500 * area + 14000 * rooms + 2000 * floor - 900 * building_age + 20000 + noise

    return pd.DataFrame(
        {
            "area_m2": np.round(area, 2),
            "rooms": rooms.astype(float),
            "floor": floor.astype(float),
            "building_age": np.round(building_age, 2),
            "price": np.round(price, 2),
        }
    )


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1] / "data"
    base_dir.mkdir(parents=True, exist_ok=True)

    train_df = generate_dataset(size=300, noise_std=8000.0, seed=42)
    test_df = generate_dataset(size=120, noise_std=8000.0, seed=43)

    train_df.to_csv(base_dir / "train.csv", index=False)
    test_df.to_csv(base_dir / "test.csv", index=False)
    print("Generated datasets: data/train.csv, data/test.csv")


if __name__ == "__main__":
    main()
