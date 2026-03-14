#!/usr/bin/env python3

import os
import glob
import pickle
import pandas as pd


def main():
    test_dir = "test"
    feature_col = "day"
    target_col = "temperature"
    model_path = "model.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    test_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))
    if not test_files:
        raise FileNotFoundError(f"No CSV files")

    test_dfs = [pd.read_csv(f) for f in test_files]
    test_all = pd.concat(test_dfs, axis=0, ignore_index=True)
    X_test = test_all[[feature_col]]
    y_test = test_all[target_col]

    score = model.score(X_test, y_test)

    print(f"Model test accuracy is: {score:.3f}")


if __name__ == "__main__":
    main()
