# Retail Demand Forecasting

An end-to-end retail demand forecasting and MLOps portfolio project built around Kaggle's **Store Item Demand Forecasting Challenge**. The repository demonstrates a chronological forecasting workflow, engineered time-series features, MLflow experiment tracking, data validation, recursive prediction, an API, orchestration, CI tests, and drift screening.

> **Dataset note:** The original Kaggle competition files are kept in `data/raw/` in this local project bundle so it runs immediately after extraction. Git is configured to ignore these CSVs. Do not force-add or redistribute the competition data; anyone reproducing this project should download it directly from Kaggle and follow the competition's rules: [competition page](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) and [rules](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/rules).

## What is implemented

- Data contract checks for columns, types, missing values, duplicate keys, negative target values, and train/test item-store coverage.
- Calendar, lag (1/7/14/28 days), and shifted rolling-window features (7/14/28-day means and 7-day standard deviation).
- A leakage-aware, **90-day chronological holdout**. The validation forecast is recursive: on later dates, the model receives its own earlier predictions rather than actual values from the holdout period.
- A seasonal-naive 7-day baseline evaluated over the same validation window.
- A scikit-learn `HistGradientBoostingRegressor`, retrained on all eligible historical rows after validation.
- MAE, RMSE, WAPE, sMAPE, and forecast bias metrics.
- MLflow experiment tracking (optional at runtime; enabled by default by the pipeline when MLflow is installed).
- Saved model artifact and metadata, Kaggle-style `submission.csv`, data profile, historical PSI drift screening, and plots.
- FastAPI `/health` and `/forecast` endpoints; recursive forecast generation for one store/item pair.
- A Docker image and Docker Compose setup for the API and MLflow tracking server.
- An Apache Airflow DAG illustrating scheduled validation, model training, drift screening, and forecast generation.
- Automated unit/API tests and a GitHub Actions CI workflow.

## Dataset

The training file contains historical daily sales with columns `date`, `store`, `item`, and `sales`. The competition test file contains `id`, `date`, `store`, and `item`, with sales values omitted. The test period follows the training history and spans 90 days.

Default paths:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

If these files are not present after cloning, accept the rules on Kaggle and download the data directly into `data/raw/`. Using the Kaggle CLI:

```bash
python -m pip install kaggle
# Configure Kaggle API credentials locally; never commit kaggle.json.
kaggle competitions download -c demand-forecasting-kernels-only -p data/raw
python -m zipfile -e data/raw/demand-forecasting-kernels-only.zip data/raw
```

The CLI download may require an API token and prior acceptance of the competition rules.

## Quick start

Python 3.10, 3.11, 3.12, or 3.13 is supported for the core project. Airflow requires a separately selected Airflow-compatible Python/runtime version. From the repository root:

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m retail_forecasting.pipeline
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -m retail_forecasting.pipeline
```

The full run builds features, performs the validation experiment, fits the final model, generates test-period predictions, creates a drift report, and writes plots. It can take several minutes on a laptop because it processes 913,000 rows and forecasts a 90-day horizon recursively. To skip MLflow tracking for a quick run, add `--no-mlflow`.

## Run individual stages

```bash
# 1. Validate schema and dataset coverage
python -m retail_forecasting.validate_cli

# 2. Train, validate over the final 90 days, save model.joblib and metadata
python -m retail_forecasting.train

# 3. Generate test forecasts into data/processed/submission.csv
python -m retail_forecasting.predict

# 4. Run historical feature-drift screening
python -m retail_forecasting.monitoring

# 5. Execute unit tests
pytest
```

## Main outputs

| Path | Purpose |
|---|---|
| `models/model.joblib` | Trained final model artifact (generated locally; Git-ignored) |
| `models/metadata.json` | Feature names, training period, validation design, and metrics |
| `reports/validation_metrics.json` | ML model and seasonal-naive validation metrics |
| `reports/dataset_profile.json` | Dataset coverage summary |
| `reports/historical_drift_report.json` | PSI-based historical distribution-shift screening |
| `reports/figures/` | Time series, month patterns, validation forecast, and model-comparison charts |
| `data/processed/validation_predictions.csv` | Per-item validation predictions and actuals (Git-ignored) |
| `data/processed/submission.csv` | `id,sales` predictions for the competition test period (Git-ignored) |
| `mlruns/` or configured tracking URI | MLflow experiment runs and model artifacts (Git-ignored) |

## Explore results

Open the charts in `reports/figures/` and the structured metrics in `reports/validation_metrics.json`. The run packaged with this project was evaluated on the **90-day holdout from 2017-10-03 through 2017-12-31**, with predictions generated recursively for the full horizon.

| Metric | HistGradientBoosting model | Seasonal-naive baseline |
|---|---:|---:|
| MAE | 6.124 | 11.649 |
| RMSE | 7.943 | 15.413 |
| WAPE | 11.210% | 21.322% |
| sMAPE | 12.902% | 22.270% |
| Mean prediction bias | +0.227 | +6.164 |

On this holdout and run, the ML model outperformed the weekly seasonal-naive baseline on the listed metrics. These are validation results—not measured performance on the competition test set, whose sales labels are not provided. Re-run the pipeline to reproduce metrics in your own environment. See `reports/project_results.md` for additional context.

### MLflow UI

The pipeline logs to a local `mlruns/` directory by default if MLflow is installed. After a pipeline run:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` in a browser. To use a remote tracking server, set `MLFLOW_TRACKING_URI` before running the pipeline.

## API

Train the model first, then start the API from the project root:

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Health: `GET http://127.0.0.1:8000/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`
- Forecast: `POST http://127.0.0.1:8000/forecast`

A ready-to-run request with 35 consecutive historical observations is included in `examples/forecast_request.json`. Send it after starting the API:

```bash
curl -X POST http://127.0.0.1:8000/forecast \
  -H "Content-Type: application/json" \
  --data @examples/forecast_request.json
```

The endpoint forecasts future days recursively and clips negative predictions to zero. The API requires at least 28 sorted, consecutive daily history points so its lag and rolling features are defined.

## Docker

After generating `models/model.joblib` locally:

```bash
docker compose up --build
```

The API is available on port 8000 and the MLflow tracking server on port 5000. Docker is optional; the local Python workflow is sufficient to run and test the core project. Do not publish secrets or bind cloud credentials into the image.

## Airflow orchestration

`dags/retail_forecast_dag.py` provides a weekly DAG. To run it, use an Airflow environment with this repository mounted at `/opt/airflow/project`, install this project in the worker/scheduler environment, and provide the competition files to the runtime. Set `RETAIL_PROJECT_ROOT`, `RETAIL_TRAIN_PATH`, and `RETAIL_TEST_PATH` if your mount paths differ. Airflow itself is intentionally not part of the basic `pip install` because its dependency constraints are platform-specific; use the [official Airflow installation guidance](https://airflow.apache.org/docs/apache-airflow/stable/installation/index.html).

## Design choices and limitations

1. **Forecast horizon:** the competition test period is 90 days, so validation uses the last 90 dates to mimic the task.
2. **Temporal integrity:** validation does not shuffle dates. Forecasting later holdout days uses earlier model predictions, not holdout actuals.
3. **Baseline:** the seasonal-naive method predicts using the value seven days earlier, and also updates its history recursively.
4. **Feature availability:** the dataset contains only sales, store, item, and date. It does not contain inventory, prices, promotions, holidays, or weather. Therefore this project forecasts sales as a demand proxy; it does not claim to optimize inventory directly.
5. **Monitoring:** PSI is a screening heuristic for feature-distribution change. It is not, by itself, proof that model performance has deteriorated or that retraining is needed.
6. **Production status:** local MLflow, API, Docker, and Airflow code are implemented. A real cloud deployment, access control, production alerting, and integration with enterprise data sources require environment-specific configuration and have not been performed by this repository build.

## Repository layout

```text
Retail_Demand_Forecasting/
├── app/main.py
├── dags/retail_forecast_dag.py
├── data/
│   ├── raw/                 # local Kaggle data; ignored by Git
│   └── processed/           # generated local outputs; ignored by Git
├── docker/mlflow/Dockerfile
├── docs/
├── examples/forecast_request.json
├── notebooks/01_exploratory_analysis.ipynb
├── reports/figures/
├── models/                  # generated artifacts; model binaries ignored by Git
├── src/retail_forecasting/
│   ├── analysis.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── forecast.py
│   ├── metrics.py
│   ├── modeling.py
│   ├── monitoring.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── train.py
│   └── validate_cli.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Interview framing

Be precise about the difference between the work implemented here and a production cloud rollout. A strong project defense should cover the time-based validation, recursive inference, why a seasonal baseline is needed, experiment/model versioning, task retries and idempotency, serving vs batch forecasts, and how to respond to drift alerts.
