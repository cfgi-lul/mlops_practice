import os
import pandas as pd
from catboost.datasets import titanic

# Папка, где лежит скрипт
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папка datasets внутри lab4
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
os.makedirs(DATASETS_DIR, exist_ok=True)

# Загрузка исходного датасета Titanic
df, _ = titanic()
df.to_csv(os.path.join(DATASETS_DIR, "titanic.csv"), index=False)

print("Исходный датасет создан")