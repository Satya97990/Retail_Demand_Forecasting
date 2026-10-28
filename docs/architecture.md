# Architecture and operational design

## Flow

1. **Data contract** — validate expected schema, parse dates and numeric columns, reject missing or duplicate observations, and verify test store/item pairs exist in the history.
2. **Feature generation** — calendar features plus lagged sales and shifted rolling summaries. Shifted windows ensure the target day cannot appear in its own features.
3. **Training and validation** — train a tree-boosting model on an earlier period; forecast the last 90 days recursively. Compare against a seasonal-naive baseline on the exact same dates.
4. **Final fit** — after validation, refit the final model on every eligible historical row and save the artifact plus metadata.
5. **Forecasting** — recursively predict the 90-day competition test period. Every forecast day is appended to history before the next date is predicted.
6. **Serving** — FastAPI accepts a store/item identifier and at least 28 consecutive history points, and produces a future horizon of up to 90 days.
7. **Orchestration and monitoring** — the Airflow DAG sequences validation, training, historical drift screening, and forecast generation. PSI results are a signal to investigate, not an automatic retraining trigger.

## Reliability considerations

- The ingestion contract fails fast on malformed files rather than silently dropping rows.
- Validation is chronological; no random split is used for final holdout assessment.
- Forecast recursion avoids using actual sales from later forecast dates.
- `submission.csv` is validated for row count, unique IDs, and non-negative finite predictions.
- Tasks should be safe to rerun because outputs are rewritten deterministically. For production pipelines writing to shared data stores, use partitioned/versioned output paths and atomic promotion.
- Model and feature versions should be deployed together. A changed feature schema should fail tests before serving.
- For production cloud use, add IAM/service accounts, secret management, secure artifact storage, API authentication/authorization, alert routing, cost controls, and an approved model promotion process.

## Tool responsibility map

| Component | Responsibility |
|---|---|
| pandas / scikit-learn | Data transformation and modelling |
| MLflow | Experiment/run history, parameters, metrics, and model artifacts; registry registration can be requested when supported by the configured backend |
| Airflow | Scheduling and dependency/retry management for jobs |
| FastAPI | Online prediction interface |
| Docker | Repeatable application runtime |
| GitHub Actions | Automated tests on pushes and pull requests |
| PSI monitor | Historical feature-distribution shift screening |

## Local MLflow vs model registry

The local file-backed default is suitable for experiment tracking. To use an MLflow Model Registry, configure a backend that supports registry operations, then run training with `--register-model`. If registration fails, it is not considered a successful deployment; the local model file remains available and the error is logged.
