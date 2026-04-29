import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Чтение датасета с заполненными пропусками
df = pd.read_csv(os.path.join(DATASETS_DIR, "titanic_filled.csv"))

# One-hot encoding для колонки Sex
df = pd.get_dummies(df, columns=['Sex'])

# Сохраняем итоговый датасет
df.to_csv(os.path.join(DATASETS_DIR, "titanic_encoded.csv"), index=False)

print("One-hot encoding выполнен, файл сохранён")