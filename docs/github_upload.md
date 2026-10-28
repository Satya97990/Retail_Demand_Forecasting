# Publish the code repository to GitHub

The local project bundle contains the downloaded dataset and a trained model so the example can run immediately. `.gitignore` deliberately excludes `data/raw/*.csv`, generated processed CSVs, the binary model artifact, MLflow run data, and local caches. Keep those exclusions in place to avoid publishing competition files or generated binary artifacts unintentionally.

From the `Retail_Demand_Forecasting` folder, create an empty repository on GitHub, then run:

```bash
git init
git add .
git status --short
git commit -m "Add retail demand forecasting MLOps project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own repository URL. Before committing, check `git status` and confirm the raw CSV files, `models/model.joblib`, `data/processed/*.csv`, `.env` files, and credentials are not staged. Do not use `git add -f` on ignored data/model files.

Anyone cloning the repository will need to download the dataset from Kaggle directly, accept the competition rules, and place the files in `data/raw/`. The tracked figures and validation reports summarize the project run but do not include the raw competition data.
