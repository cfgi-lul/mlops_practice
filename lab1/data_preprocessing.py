#!/usr/bin/env python3

import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler


def main():
    train_dir = "train"
    test_dir = "test"
    feature_cols = ["day", "temperature"]

    # training data for training scaler
    train_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
    if not train_files:
        raise FileNotFoundError(f"No CSV files. Run data_creation.py")

    train_dfs = [pd.read_csv(f) for f in train_files]
    train_all = pd.concat(train_dfs, axis=0, ignore_index=True)
    X_train = train_all[feature_cols]

    # train scaler on train
    scaler = StandardScaler()
    scaler.fit(X_train)

    # transform train
    for path in train_files:
        df = pd.read_csv(path)
        df[feature_cols] = scaler.transform(df[feature_cols])
        df.to_csv(path, index=False)

    # transform test
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    for path in test_files:
        df = pd.read_csv(path)
        df[feature_cols] = scaler.transform(df[feature_cols])
        df.to_csv(path, index=False)

    print("Preprocessing done")


if __name__ == "__main__":
    main()
