import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv("datasets/train/preprocessed_train_data.csv")

# Разделение на train и test
y_train = train_data["Survived"]
X_train = train_data.drop(columns="Survived")

# Обучение модели
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Сохранение
with open('model.pkl', 'wb') as f:
    pickle.dump(lr, f)
