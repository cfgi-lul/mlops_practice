# Final Task: Streamlit ML data product

## What is implemented
- Web application on Streamlit (`main.py`) for house price prediction.
- Pretrained linear regression model in `model/linear_model.pkl`.
- Versioned demo dataset in `data/sample_input.csv` via DVC metadata.
- Unit tests + data quality tests in `tests/`.
- Docker image build via `Dockerfile`.
- GitLab CI pipeline in repository root (`.gitlab-ci.yml`).

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
cd "final task"
dvc init
dvc remote add -d localremote ../dvc-storage-final-task
dvc add data/sample_input.csv
git add data/sample_input.csv.dvc .dvc/config .gitignore
```