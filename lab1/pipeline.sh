#!/usr/bin/env bash
set -e # exit on error

# dependencies
python3 -m pip install -q numpy pandas scikit-learn

# dataset creation
python3 data_creation.py

# preprocessing
python3 data_preprocessing.py

# train and save
python3 model_preparation.py

# evaluate on test
python3 model_testing.py
