"""
Titanic Survival Prediction - End-to-End MLOps Pipeline
Apache Airflow DAG + MLflow Experiment Tracking
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# ── Globals ──────────────────────────────────────────────────────────────────
DATASET_PATH = "/opt/airflow/data/titanic.csv"
MLFLOW_TRACKING_URI = "http://mlflow:5000"

# Hyperparameters – change these per run to compare experiments
MODEL_TYPE      = os.getenv("MODEL_TYPE",       "RandomForest")   # "LogisticRegression" | "RandomForest"
N_ESTIMATORS    = int(os.getenv("N_ESTIMATORS",  "100"))
MAX_DEPTH       = int(os.getenv("MAX_DEPTH",     "5"))
C_PARAM         = float(os.getenv("C_PARAM",     "1.0"))
ACCURACY_THRESHOLD = 0.80

logger = logging.getLogger(__name__)

# ── Default args ──────────────────────────────────────────────────────────────
default_args = {
    "owner": "mlops_student",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(seconds=10),
}

# ══════════════════════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ── Task 2 : Data Ingestion ───────────────────────────────────────────────────
def ingest_data(**context):
    """Load CSV, print shape, log missing values, push path via XCom."""
    logger.info("=== Task 2 : Data Ingestion ===")

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}. "
            "Make sure Titanic-Dataset.csv is in the ./data folder."
        )

    df = pd.read_csv(DATASET_PATH)

    logger.info("Dataset shape: %s", df.shape)
    print(f"[Ingestion] Dataset shape: {df.shape}")

    missing = df.isnull().sum()
    missing_nonzero = missing[missing > 0]
    logger.info("Missing values per column:\n%s", missing_nonzero.to_string())
    print(f"[Ingestion] Missing values:\n{missing_nonzero.to_string()}")

    # Push path for downstream tasks
    context["ti"].xcom_push(key="dataset_path", value=DATASET_PATH)
    logger.info("Dataset path pushed to XCom: %s", DATASET_PATH)


# ── Task 3 : Data Validation ──────────────────────────────────────────────────
def validate_data(**context):
    """
    Check missing % for Age & Embarked.
    On the FIRST attempt we intentionally raise to demonstrate retry behaviour.
    """
    logger.info("=== Task 3 : Data Validation ===")

    dataset_path = context["ti"].xcom_pull(
        key="dataset_path", task_ids="ingest_data"
    )
    df = pd.read_csv(dataset_path)

    # ── Intentional failure on attempt 1 (retry demo) ────────────────────────
    # try_number is 1 on first attempt, 2 on first retry — persists across processes
    try_number = context["ti"].try_number
    logger.info("Validation try_number: %d", try_number)

    if try_number == 1:
        logger.warning("INTENTIONAL FAILURE on attempt 1 to demonstrate retry.")
        raise RuntimeError(
            "Intentional failure on first attempt – Airflow will retry automatically."
        )

    logger.info("Validation attempt %d (retry succeeded)", try_number)

    # ── Real validation ───────────────────────────────────────────────────────
    total = len(df)
    for col in ["Age", "Embarked"]:
        pct = df[col].isnull().mean() * 100
        logger.info("Missing %% in %-10s: %.2f%%", col, pct)
        if pct > 30:
            raise ValueError(
                f"Column '{col}' has {pct:.2f}%% missing values – exceeds 30%% threshold!"
            )

    logger.info("Validation passed.")


# ── Task 4a : Handle Missing Values ──────────────────────────────────────────
def handle_missing_values(**context):
    """Fill Age (median) and Embarked (mode). Runs in PARALLEL with feature_engineering."""
    logger.info("=== Task 4a : Handle Missing Values ===")

    dataset_path = context["ti"].xcom_pull(
        key="dataset_path", task_ids="ingest_data"
    )
    df = pd.read_csv(dataset_path)

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
    df["Fare"].fillna(df["Fare"].median(), inplace=True)

    out_path = "/opt/airflow/data/titanic_missing_handled.csv"
    df.to_csv(out_path, index=False)

    context["ti"].xcom_push(key="missing_handled_path", value=out_path)
    logger.info("Missing values handled. Saved to %s", out_path)


# ── Task 4b : Feature Engineering ────────────────────────────────────────────
def feature_engineering(**context):
    """Create FamilySize & IsAlone. Runs in PARALLEL with handle_missing_values."""
    logger.info("=== Task 4b : Feature Engineering ===")

    dataset_path = context["ti"].xcom_pull(
        key="dataset_path", task_ids="ingest_data"
    )
    df = pd.read_csv(dataset_path)

    # Fill Age/Embarked here too so we have a self-contained artefact
    df["Age"].fillna(df["Age"].median(), inplace=True)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"]    = (df["FamilySize"] == 1).astype(int)

    logger.info("FamilySize stats:\n%s", df["FamilySize"].describe().to_string())
    logger.info("IsAlone distribution:\n%s", df["IsAlone"].value_counts().to_string())

    out_path = "/opt/airflow/data/titanic_features.csv"
    df.to_csv(out_path, index=False)

    context["ti"].xcom_push(key="features_path", value=out_path)
    logger.info("Feature engineering complete. Saved to %s", out_path)


# ── Task 5 : Data Encoding ────────────────────────────────────────────────────
def encode_data(**context):
    """Merge parallel outputs, encode Sex/Embarked, drop irrelevant columns."""
    logger.info("=== Task 5 : Data Encoding ===")

    ti = context["ti"]
    missing_path  = ti.xcom_pull(key="missing_handled_path", task_ids="handle_missing_values")
    features_path = ti.xcom_pull(key="features_path",        task_ids="feature_engineering")

    df_m = pd.read_csv(missing_path)
    df_f = pd.read_csv(features_path)

    # Merge the two parallel artefacts on PassengerId
    df = df_m.merge(df_f[["PassengerId", "FamilySize", "IsAlone"]], on="PassengerId")

    # Encode
    df["Sex"]      = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # Drop irrelevant columns
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    out_path = "/opt/airflow/data/titanic_encoded.csv"
    df.to_csv(out_path, index=False)

    ti.xcom_push(key="encoded_path", value=out_path)
    logger.info("Encoding complete. Final columns: %s", df.columns.tolist())


# ── Task 6 : Model Training with MLflow ──────────────────────────────────────
def train_model(**context):
    """Train model, log params & artefacts to MLflow."""
    import mlflow
    import mlflow.sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    logger.info("=== Task 6 : Model Training ===")

    ti = context["ti"]
    encoded_path = ti.xcom_pull(key="encoded_path", task_ids="encode_data")

    df = pd.read_csv(encoded_path).dropna()
    X  = df.drop(columns=["Survived"])
    y  = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Titanic_Survival_Prediction")

    # Force all artifact uploads through the HTTP tracking server (server mode)
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
    os.environ["GIT_PYTHON_REFRESH"] = "quiet"
    os.environ["MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD"] = "true"
    mlflow.environment_variables.MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING.set(False)

    with mlflow.start_run(run_name=f"{MODEL_TYPE}_run_{datetime.now().strftime('%H%M%S')}") as run:
        mlflow.log_param("model_type",    MODEL_TYPE)
        mlflow.log_param("test_size",     0.2)
        mlflow.log_param("random_state",  42)
        mlflow.log_param("dataset_size",  len(df))
        mlflow.log_param("n_features",    X.shape[1])

        if MODEL_TYPE == "RandomForest":
            mlflow.log_param("n_estimators", N_ESTIMATORS)
            mlflow.log_param("max_depth",    MAX_DEPTH)
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                max_depth=MAX_DEPTH,
                random_state=42,
            )
        else:
            mlflow.log_param("C", C_PARAM)
            model = LogisticRegression(C=C_PARAM, max_iter=500, random_state=42)

        model.fit(X_train, y_train)

        mlflow.sklearn.log_model(model, artifact_path="model")
        mlflow.log_param("dataset_size_train", len(X_train))
        mlflow.log_param("dataset_size_test",  len(X_test))

        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)

    # Push for downstream tasks
    ti.xcom_push(key="run_id",    value=run_id)
    ti.xcom_push(key="X_test",    value=X_test.to_json())
    ti.xcom_push(key="y_test",    value=y_test.to_json(orient="records"))
    ti.xcom_push(key="model_type", value=MODEL_TYPE)

    # Persist test split for evaluation
    X_test.to_csv("/opt/airflow/data/X_test.csv", index=False)
    y_test.to_csv("/opt/airflow/data/y_test.csv", index=False)


# ── Task 7 : Model Evaluation ────────────────────────────────────────────────
def evaluate_model(**context):
    """Compute metrics, log to MLflow, push accuracy via XCom."""
    import mlflow
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
    )

    logger.info("=== Task 7 : Model Evaluation ===")

    ti     = context["ti"]
    run_id = ti.xcom_pull(key="run_id",    task_ids="train_model")

    X_test = pd.read_csv("/opt/airflow/data/X_test.csv")
    y_test = pd.read_csv("/opt/airflow/data/y_test.csv").squeeze()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

    with mlflow.start_run(run_id=run_id):
        # Reload model from MLflow
        model_uri = f"runs:/{run_id}/model"
        model     = mlflow.sklearn.load_model(model_uri)

        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test,  y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test,    y_pred, zero_division=0)
        f1   = f1_score(y_test,        y_pred, zero_division=0)

        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)

        logger.info("Accuracy : %.4f", acc)
        logger.info("Precision: %.4f", prec)
        logger.info("Recall   : %.4f", rec)
        logger.info("F1-score : %.4f", f1)

    ti.xcom_push(key="accuracy", value=acc)


# ── Task 8 : Branching Logic ──────────────────────────────────────────────────
def branch_on_accuracy(**context):
    """Route to register_model or reject_model based on accuracy threshold."""
    ti       = context["ti"]
    accuracy = ti.xcom_pull(key="accuracy", task_ids="evaluate_model")

    logger.info("Accuracy = %.4f  |  Threshold = %.2f", accuracy, ACCURACY_THRESHOLD)

    if accuracy >= ACCURACY_THRESHOLD:
        logger.info("Accuracy meets threshold → registering model.")
        return "register_model"
    else:
        logger.info("Accuracy below threshold → rejecting model.")
        return "reject_model"


# ── Task 9a : Model Registration ──────────────────────────────────────────────
def register_model(**context):
    """Register the approved model in MLflow Model Registry."""
    import mlflow

    logger.info("=== Task 9a : Register Model ===")

    ti     = context["ti"]
    run_id = ti.xcom_pull(key="run_id", task_ids="train_model")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

    model_uri  = f"runs:/{run_id}/model"
    model_name = "TitanicSurvivalModel"

    registered = mlflow.register_model(model_uri=model_uri, name=model_name)

    logger.info(
        "Model registered: name=%s  version=%s",
        registered.name,
        registered.version,
    )

    # Optionally transition to Staging
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    client.transition_model_version_stage(
        name=model_name,
        version=registered.version,
        stage="Staging",
    )
    logger.info("Model transitioned to Staging.")


# ── Task 9b : Model Rejection ─────────────────────────────────────────────────
def reject_model(**context):
    """Log rejection reason to MLflow."""
    import mlflow

    logger.info("=== Task 9b : Reject Model ===")

    ti       = context["ti"]
    run_id   = ti.xcom_pull(key="run_id",    task_ids="train_model")
    accuracy = ti.xcom_pull(key="accuracy",  task_ids="evaluate_model")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

    with mlflow.start_run(run_id=run_id):
        rejection_reason = (
            f"Model rejected: accuracy={accuracy:.4f} < threshold={ACCURACY_THRESHOLD}"
        )
        mlflow.set_tag("rejection_reason", rejection_reason)
        mlflow.set_tag("model_status", "REJECTED")

    logger.warning(rejection_reason)


# ══════════════════════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════════════════════

with DAG(
    dag_id="mlops_airflow_mlflow_pipeline",
    default_args=default_args,
    description="Titanic Survival Prediction – Airflow + MLflow Pipeline",
    schedule_interval=None,          # manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "titanic", "mlflow"],
) as dag:

    # ── Start sentinel ────────────────────────────────────────────────────────
    start = EmptyOperator(task_id="start")

    # ── Task 2 ────────────────────────────────────────────────────────────────
    t_ingest = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    # ── Task 3 (with retries) ─────────────────────────────────────────────────
    t_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        retries=2,
        retry_delay=timedelta(seconds=10),
    )

    # ── Task 4 – PARALLEL ─────────────────────────────────────────────────────
    t_missing  = PythonOperator(
        task_id="handle_missing_values",
        python_callable=handle_missing_values,
    )

    t_features = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    # ── Task 5 ────────────────────────────────────────────────────────────────
    t_encode = PythonOperator(
        task_id="encode_data",
        python_callable=encode_data,
    )

    # ── Task 6 ────────────────────────────────────────────────────────────────
    t_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    # ── Task 7 ────────────────────────────────────────────────────────────────
    t_evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
    )

    # ── Task 8 – Branch ───────────────────────────────────────────────────────
    t_branch = BranchPythonOperator(
        task_id="branch_on_accuracy",
        python_callable=branch_on_accuracy,
    )

    # ── Task 9 ────────────────────────────────────────────────────────────────
    t_register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    t_reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model,
    )

    # ── End sentinel (runs regardless of branch) ─────────────────────────────
    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    # ══════════════════════════════════════════════════════════════════════════
    # DEPENDENCY GRAPH
    #
    #  start
    #    │
    #  ingest_data
    #    │
    #  validate_data
    #    ├─────────────────────┐
    #  handle_missing_values  feature_engineering   ← PARALLEL
    #    └─────────────────────┘
    #           │
    #       encode_data
    #           │
    #       train_model
    #           │
    #      evaluate_model
    #           │
    #     branch_on_accuracy
    #        ┌──┴──┐
    #  register   reject
    #        └──┬──┘
    #          end
    # ══════════════════════════════════════════════════════════════════════════
    start >> t_ingest >> t_validate
    t_validate >> [t_missing, t_features]   # parallel fan-out
    [t_missing, t_features] >> t_encode     # fan-in
    t_encode >> t_train >> t_evaluate >> t_branch
    t_branch >> [t_register, t_reject]
    [t_register, t_reject] >> end