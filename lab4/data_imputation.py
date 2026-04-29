import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Чтение модифицированного датасета
df = pd.read_csv(os.path.join(DATASETS_DIR, "titanic_modified.csv"))

# Заполнение пропущенных значений в Age средним
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Сохраняем датасет с заполненными значениями
df.to_csv(os.path.join(DATASETS_DIR, "titanic_filled.csv"), index=False)

print("Пропущенные значения заполнены и сохранены")