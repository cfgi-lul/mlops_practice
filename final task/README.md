## Final task
<details>

### Цель проекта: разработать конвеер машинного обучения data-продукта (Web или API приложение).

Команда проекта. Проект выполняется в команде из 3-4 человека.

Требования к реализации проекта:
1. Исходные коды проекта должны находиться в репозитории GitHub.
2. Проект оркестируется с помощью ci/cd (jenkins или gitlab).
3. Датасеты версионируются с помощью dvc и синхронизируются с удалённым хранилищем.
4. Разработка возможностей приложения должна проводиться в отдельных ветках, наборы фичей и версии данных тоже.
5. В коневеере запускаются не только модульные тесты, но и проверка тестами на качество данных.
6. Итоговое приложение реализуется в виде образа docker. Сборка образа происходит в конвеере.
7. В проекте может использоваться предварительно обученная модель. Обучать собственную модель не требуется.

</details>


# Final Task: Streamlit ML data product

## What is implemented
- Web application on Streamlit (`main.py`) for house price prediction.
- Pretrained linear regression model in `model/linear_model.pkl`.
- Versioned train/test datasets in `data/train.csv` and `data/test.csv` via DVC metadata.
- Unit tests + data quality tests in `tests/`.
- Docker image build via `Dockerfile`.
- GitLab CI pipeline in repository root (`.gitlab-ci.yml`).
- Dataset generator with normally distributed noise in `scripts/generate_datasets.py`.

## Local run
```bash
cd "final task"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

## Run tests
```bash
cd "final task"
PYTHONPATH=. pytest -q
```

## Docker run
```bash
cd "final task"
docker build -t final-task-streamlit .
docker run --rm -p 8501:8501 final-task-streamlit
```

## DVC notes
Example commands for local DVC remote:
```bash
cd "/Users/cfgi/Desktop/ueba/Мдааагистратура/2сем/mlops/mlops_practice"
dvc init -f
dvc remote add -d localremote ../dvc-storage-final-task
python "final task/scripts/generate_datasets.py"
dvc add "final task/data/train.csv" "final task/data/test.csv"
git add "final task/data/train.csv.dvc" "final task/data/test.csv.dvc" .dvc/config
```