#!/usr/bin/env python3

import os
import glob
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression


def main():
    train_dir = "train"
    feature_col = "day"
    target_col = "temperature"
    model_path = "model.pkl"

    train_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
    if not train_files:
        raise FileNotFoundError(
            f"No CSV files in {train_dir}. Run data_creation.py and data_preprocessing.py first."
        )

    train_dfs = [pd.read_csv(f) for f in train_files]
    train_all = pd.concat(train_dfs, axis=0, ignore_index=True)
    X_train = train_all[[feature_col]]
    y_train = train_all[target_col]

    model = LinearRegression()
    model.fit(X_train, y_train)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("Model trained and saved")


if __name__ == "__main__":
    main()
