# Лабораторная работа 3 — MLOps (Docker)

Приложение разработано с использованием:

* Streamlit — для создания веб-интерфейса
* TensorFlow — для работы с моделью

В основе лежит нейронная сеть EfficientNetB0.

---

## Функциональность

* Загрузка изображения пользователем
* Классификация изображения с помощью модели
* Отображение результата в веб-интерфейсе

---

## 🚀 Запуск через Docker

### 1. Сборка образа

```bash
docker build -t lab3-app:1.0 .
```

### 2. Запуск контейнера

```bash
docker run -p 8501:8501 lab3-app:1.0
```

### 3. Открыть в браузере

```
http://localhost:8501
```

---

## 🧩 Запуск через Docker Compose

### 1. Запуск

```bash
docker compose up --build
```

### 2. Остановка

```bash
docker compose down
```

---
## Автоматическая сборка и публикация образа

При push в основную ветку GitHub Actions автоматически:
- собирает Docker image
- публикует его в Docker Hub
- присваивает тег по SHA коммита и имени ветки
---

## 📁 Структура проекта

```
lab3/
│── image_classification.py
│── requirements.txt
│── Dockerfile
│── docker-compose.yml
│── README.md
```