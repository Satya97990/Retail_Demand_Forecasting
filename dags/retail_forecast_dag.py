"""Airflow DAG demonstrating scheduled validation, model training, monitoring, and forecasting.

Deploy this DAG only after mounting the repository into the Airflow worker/scheduler at
/opt/airflow/project (or setting RETAIL_PROJECT_ROOT to the mounted path) and installing the
project package in the Airflow runtime. See docs/architecture.md for setup notes.
"""
from __future__ import annotations

import os
from datetime import datetime

try:
    from airflow.decorators import dag
    from airflow.operators.bash import BashOperator
except ImportError as exc:  # Keep static tooling and unit tests usable without Airflow installed.
    raise ImportError("Airflow is optional. Install Apache Airflow to load this DAG.") from exc

PROJECT_ROOT = os.getenv("RETAIL_PROJECT_ROOT", "/opt/airflow/project")
TRAIN_PATH = os.getenv("RETAIL_TRAIN_PATH", f"{PROJECT_ROOT}/data/raw/train.csv")
TEST_PATH = os.getenv("RETAIL_TEST_PATH", f"{PROJECT_ROOT}/data/raw/test.csv")
ENV_PREFIX = f"cd {PROJECT_ROOT} && export PYTHONPATH={PROJECT_ROOT}/src:{PROJECT_ROOT}:$PYTHONPATH && "


@dag(
    dag_id="retail_demand_forecasting_pipeline",
    description="Validate data, train/evaluate a model, screen drift, and generate a submission file.",
    start_date=datetime(2024, 1, 1),
    schedule="0 5 * * 1",
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "ai-engineering", "retries": 2},
    tags=["retail", "forecasting", "mlops"],
)
def retail_demand_forecasting_pipeline():
    validate = BashOperator(
        task_id="validate_data",
        bash_command=(
            ENV_PREFIX
            + "python -m retail_forecasting.validate_cli "
            + f"--train-path {TRAIN_PATH} --test-path {TEST_PATH}"
        ),
    )
    train = BashOperator(
        task_id="train_and_evaluate",
        bash_command=(ENV_PREFIX + f"python -m retail_forecasting.train --train-path {TRAIN_PATH}"),
    )
    monitor = BashOperator(
        task_id="historical_drift_screen",
        bash_command=(
            ENV_PREFIX
            + f"python -m retail_forecasting.monitoring --train-path {TRAIN_PATH}"
        ),
    )
    forecast = BashOperator(
        task_id="generate_forecasts",
        bash_command=(
            ENV_PREFIX
            + "python -m retail_forecasting.predict "
            + f"--train-path {TRAIN_PATH} --test-path {TEST_PATH}"
        ),
    )
    validate >> train >> [monitor, forecast]


retail_demand_forecasting_pipeline()
