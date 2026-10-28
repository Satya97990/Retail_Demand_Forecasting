# Interview defense guide — Retail Demand Forecasting

Use this document to prepare, but answer from your actual run results. Do not memorize claims that the model beats the baseline unless the metrics confirm it.

## 90-second project explanation

"I built an end-to-end retail demand forecasting pipeline using the Store Item Demand Forecasting dataset. The data has daily sales by store and item. I validate the input schema and coverage, generate calendar features and leakage-safe lag/rolling features, and train a HistGradientBoostingRegressor. To mimic the actual 90-day forecasting task, I hold out the final 90 days and generate predictions recursively, using previous predictions for later forecast dates. I compare the model to a seasonal-naive baseline with MAE, RMSE, WAPE, and sMAPE. After validation, I refit the final model on all eligible historical data, track experiments with MLflow, generate a Kaggle-format forecast file, expose an API through FastAPI, containerize it with Docker, and provide Airflow orchestration, automated tests, and a feature-drift screening report. The current version runs locally; cloud security and deployment need environment-specific setup." 

## Questions and answer points

### Why forecasting, and what is the target?
The target is `sales` for each date/store/item combination. It is a demand proxy, not necessarily unconstrained demand because the dataset does not identify stockouts. Forecasts can support replenishment decisions but are not a complete inventory optimizer.

### Why is time-based validation needed?
Random splitting can train on dates later than validation dates and does not mimic forecasting future periods. The project holds out the final 90 days and forecasts them forward in order.

### What is recursive forecasting?
For the first holdout date, features use only available historical sales. The prediction is then appended to history, and the next date's lag features use that prediction. The process repeats over the full horizon. This is more realistic than using actual holdout sales to construct later-day lag features.

### Why use lag features and rolling means?
Sales have temporal dependence and can display weekly or longer patterns. Lag features represent recent and seasonal history; rolling means summarize recent levels. Every rolling feature is shifted so today's target does not enter today's features.

### Why use a seasonal-naive baseline?
Complex models should beat a simple, credible baseline. Here the baseline forecasts using the sales value seven days earlier, which captures a weekly pattern. Both methods are tested on the same dates and horizon.

### Why these metrics?
- MAE is interpretable in units sold.
- RMSE penalizes larger errors more heavily.
- WAPE summarizes absolute error relative to total sales volume.
- sMAPE expresses a symmetric percentage error and handles both-zero cases as zero here.

No single metric captures every business cost. Evaluate per item/store and consider under- vs over-forecasting when translating results into decisions.

### Why MLflow?
To track parameters, metrics, model artifacts, and runs reproducibly. A registry adds controlled model versioning and promotion when backed by a supported tracking database and deployment process.

### Why Airflow?
To schedule and coordinate validation, training, monitoring, and forecast generation, with dependencies, retries, and operational visibility. Airflow itself does not make a pipeline idempotent; the tasks and output paths must be designed for safe reruns.

### Batch vs real-time inference?
This project creates batch predictions for many store/item/date rows. An API is also included to demonstrate online serving for a single series. Retail replenishment may favor scheduled batch forecasts if consumers need a complete forecast table rather than an immediate request/response.

### What does drift monitoring do?
The PSI report compares feature distributions between an earlier reference period and a later period. It helps identify changes that deserve investigation. Drift does not prove accuracy declined; model errors should be checked when actual outcomes become available.

### What are important limitations?
The competition data only has date/store/item/sales. It does not have inventory, product price, promotions, weather, holidays, costs, or stock availability. It is also a relatively small, tidy benchmark rather than a real enterprise retail platform. Results may not transfer directly to Lowe's production setting.

## Operational failure scenarios

- **Incoming file missing a column:** fail validation, alert the pipeline owner, do not train on malformed data.
- **Duplicate observations:** fail the data contract and investigate rather than silently aggregate them.
- **Airflow task fails:** inspect logs, retry transient failures, make task outputs idempotent, and avoid promoting partial artifacts.
- **Model artifact missing:** the API returns a service-unavailable response rather than fabricating a forecast.
- **New model performs worse:** retain the known-good model and require a comparison/approval gate before promotion.
- **PSI rises:** inspect upstream data changes and evaluate forecast errors; do not automatically retrain without validation.
- **Latency/cost rises:** profile preprocessing and inference, reduce repeated loading, batch requests when appropriate, and measure the trade-offs before changing architecture.

## Your own results to fill in after running

- Model validation MAE / RMSE / WAPE / sMAPE: read `reports/validation_metrics.json`.
- Baseline validation metrics: same file, `seasonal_naive_metrics`.
- Whether the model beats the baseline: compare the same metric and validation period; be explicit if it does not.
- Validation window: `validation_start` to `validation_end` in the metrics JSON.
- Test rows and forecast range: inspect `reports/dataset_profile.json` and `data/processed/submission.csv`.
- Actual implementation status: distinguish code present, code tested locally, service started, and cloud deployed. These are different levels of completion.
