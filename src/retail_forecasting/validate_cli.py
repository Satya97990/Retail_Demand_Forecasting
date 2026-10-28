"""Command-line dataset validation."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from retail_forecasting.config import DEFAULT_TEST_PATH, DEFAULT_TRAIN_PATH, REPORTS_DIR
from retail_forecasting.data import load_test, load_train, validate_dataset_coverage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--test-path", default=str(DEFAULT_TEST_PATH))
    parser.add_argument("--report-path", default=str(REPORTS_DIR / "dataset_profile.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train, test = load_train(args.train_path), load_test(args.test_path)
    manifest = validate_dataset_coverage(train, test)
    output = Path(args.report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
