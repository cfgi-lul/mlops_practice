#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd

DEFAULT_SEED = 42


def generate_temperature_series(
    n_days: int,
    base_temp: float,
    amplitude: float,
    trend: float = 0,
    noise_std: float = 0,
    anomaly_days: list = None,
    anomaly_shift: float = 0,
) -> pd.DataFrame:
    rng = np.random.RandomState(DEFAULT_SEED)
    t = np.arange(n_days)
    # годовая сезонность + недельная сезонность
    temp = base_temp + amplitude * (
        np.sin(2 * np.pi * t / 365) + 0.3 * np.sin(2 * np.pi * t / 7)
    )
    temp += trend * t / n_days
    if noise_std > 0:
        temp += rng.normal(0, noise_std, n_days)
    if anomaly_days is not None and anomaly_shift is not None:
        temp = temp.astype(float)
        shifts = np.atleast_1d(anomaly_shift)
        for i, day in enumerate(anomaly_days):
            if 0 <= day < n_days and i < len(shifts):
                temp[day] += shifts[i]
    return pd.DataFrame({"day": t, "temperature": temp})


def main():
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)

    rng = np.random.RandomState(DEFAULT_SEED)
    n_days = 365 * 2  # 2 года

    # чистые данные
    df1 = generate_temperature_series(n_days, base_temp=10.0, amplitude=12.0)
    df1.iloc[: int(0.7 * n_days)].to_csv("train/temperature_clean.csv", index=False)
    df1.iloc[int(0.7 * n_days) :].to_csv("test/temperature_clean.csv", index=False)

    # данные с шумом
    df2 = generate_temperature_series(
        n_days, base_temp=8.0, amplitude=14.0, noise_std=2.0
    )
    df2.iloc[: int(0.7 * n_days)].to_csv("train/temperature_noisy.csv", index=False)
    df2.iloc[int(0.7 * n_days) :].to_csv("test/temperature_noisy.csv", index=False)

    # данные с аномалиями
    anomalies = [100, 200, 400, 500]
    df3 = generate_temperature_series(
        n_days,
        base_temp=11.0,
        amplitude=10.0,
        anomaly_days=anomalies,
        anomaly_shift=rng.uniform(8, 15, size=len(anomalies)),
    )
    df3.iloc[: int(0.7 * n_days)].to_csv("train/temperature_anomalies.csv", index=False)
    df3.iloc[int(0.7 * n_days) :].to_csv("test/temperature_anomalies.csv", index=False)

    # данные с трендом и шумом
    df4 = generate_temperature_series(
        n_days, base_temp=9.0, amplitude=11.0, trend=1.5, noise_std=0.5
    )
    df4.iloc[: int(0.7 * n_days)].to_csv("train/temperature_trend.csv", index=False)
    df4.iloc[int(0.7 * n_days) :].to_csv("test/temperature_trend.csv", index=False)

    print("Datasets created")


if __name__ == "__main__":
    main()
