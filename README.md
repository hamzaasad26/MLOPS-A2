# Titanic MLOps Pipeline — Airflow + MLflow

End-to-end ML pipeline that orchestrates data ingestion, validation, feature engineering,
model training, evaluation, and registration using **Apache Airflow 2.9** and **MLflow 2.13**.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Docker Desktop | ≥ 4.x |
| Docker Compose | ≥ 2.x |
| Git | any |

---

## Project Structure

```
Assignment2/
├── dags/
│   └── mlops_airflow_mlflow_pipeline.py   ← Main DAG
├── data/
│   └── Titanic-Dataset.csv                ← Dataset (download below)
├── logs/                                  ← Airflow logs (auto-created)
├── plugins/                               ← Empty (required by Airflow)
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Download the Dataset

Download **Titanic-Dataset.csv** from  
https://www.kaggle.com/datasets/yasserh/titanic-dataset/data  
and place it in the `data/` folder.

### 2. Create required folders & set UID

**Windows (PowerShell):**
```powershell
mkdir -Force dags, logs, plugins, data
echo "AIRFLOW_UID=50000" > .env
```

**Linux / macOS:**
```bash
mkdir -p dags logs plugins data
echo "AIRFLOW_UID=$(id -u)" > .env
```

### 3. Copy the DAG

```powershell
# Windows
Copy-Item dags\mlops_airflow_mlflow_pipeline.py dags\
```

### 4. Start the stack

```bash
docker compose up airflow-init   # one-time DB setup (waits ~60 s)
docker compose up -d             # start all services
```

Services:
| Service | URL | Credentials |
|---------|-----|-------------|
| Airflow Webserver | http://localhost:8080 | admin / admin |
| MLflow UI | http://localhost:5000 | – |

### 5. Trigger the DAG

Open http://localhost:8080 → toggle **mlops_airflow_mlflow_pipeline** ON → click ▶ **Trigger DAG**.

---

## Running Multiple Experiments (Task 10)

Override environment variables to change hyperparameters between runs.

**Run 1 – Random Forest (default)**
```bash
docker compose up -d   # default: MODEL_TYPE=RandomForest, N_ESTIMATORS=100, MAX_DEPTH=5
```

**Run 2 – Random Forest (deeper)**
```bash
MODEL_TYPE=RandomForest N_ESTIMATORS=200 MAX_DEPTH=10 docker compose up -d
```

**Run 3 – Logistic Regression**
```bash
MODEL_TYPE=LogisticRegression C_PARAM=0.5 docker compose up -d
```

On Windows PowerShell, set env vars in `.env` file or via System Properties before
running `docker compose up -d`.

---

## Viewing Results

- **Airflow graph view**: http://localhost:8080 → DAGs → *mlops_airflow_mlflow_pipeline* → Graph
- **MLflow experiments**: http://localhost:5000 → Experiments → *Titanic_Survival_Prediction*
- **Model registry**: http://localhost:5000 → Models → *TitanicSurvivalModel*

---

## Stopping

```bash
docker compose down            # stop containers, keep data volumes
docker compose down -v         # stop + wipe all volumes (fresh start)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Permission denied` on logs/dags | Run `echo "AIRFLOW_UID=$(id -u)" > .env` (Linux/Mac) or set `AIRFLOW_UID=50000` in `.env` (Windows) |
| MLflow connection refused | Wait ~30 s for the `mlflow` container to pass its healthcheck before triggering the DAG |
| `FileNotFoundError: Titanic-Dataset.csv` | Ensure the CSV is in the `./data/` folder |
| Packages not found in DAG | The `_PIP_ADDITIONAL_REQUIREMENTS` env var installs them on first run – allow ~2 min |
