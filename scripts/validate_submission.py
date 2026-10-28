"""Small artifact checker for the generated Kaggle-format submission."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="data/processed/submission.csv")
    args = parser.parse_args()
    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"Missing submission file: {path}")
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "sales"]:
        raise SystemExit(f"Expected columns ['id', 'sales'], found {frame.columns.tolist()}")
    if frame["id"].isna().any() or frame["id"].duplicated().any():
        raise SystemExit("Submission IDs must be present and unique.")
    if frame["sales"].isna().any() or (frame["sales"] < 0).any():
        raise SystemExit("Predictions must be present and non-negative.")
    print(f"OK: {len(frame):,} rows; predictions range {frame.sales.min():.3f}–{frame.sales.max():.3f}")


if __name__ == "__main__":
    main()
