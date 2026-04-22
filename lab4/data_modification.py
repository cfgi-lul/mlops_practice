import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

# Чтение исходного датасета
df = pd.read_csv(os.path.join(DATASETS_DIR, "titanic.csv"))

# Оставляем нужные колонки
df = df[['Pclass', 'Sex', 'Age', 'Survived']]

# Сохраняем модифицированный датасет
df.to_csv(os.path.join(DATASETS_DIR, "titanic_modified.csv"), index=False)

print("Датасет модифицирован и сохранён")