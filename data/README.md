# Dataset instructions

This project uses the Store Item Demand Forecasting Challenge on Kaggle:
https://www.kaggle.com/competitions/demand-forecasting-kernels-only

Expected local files:
- `raw/train.csv` (`date`, `store`, `item`, `sales`)
- `raw/test.csv` (`id`, `date`, `store`, `item`)
- `raw/sample_submission.csv` (`id`, `sales`)

The competition data is not committed to Git. If it is missing, log in to Kaggle, accept the competition rules, download the files, and place them here. Raw files are ignored in `.gitignore` to avoid accidental redistribution. Never commit Kaggle API credentials.
