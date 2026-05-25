"""
monitor.py — Data drift detection and prediction log analysis.

Usage:
    python monitor.py check-drift
    python monitor.py analyze-logs
    python monitor.py report
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PREDICTION_LOG = os.getenv("PREDICTION_LOG", "predictions.jsonl")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))

TRAINING_DATA_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

FEATURE_COLS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


def load_training_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    zero_as_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df[zero_as_nan] = df[zero_as_nan].replace(0, np.nan)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    return df


def load_prediction_log(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        logger.warning("Prediction log not found at %s", path)
        return pd.DataFrame()

    records = []
    with open(path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not records:
        return pd.DataFrame()

    inputs = pd.json_normalize(records)
    result_cols = [c for c in inputs.columns if c.startswith("result.")]
    if result_cols:
        results = inputs[result_cols].copy()
        results.columns = [c.replace("result.", "") for c in result_cols]
        return results

    return inputs


def check_drift(training_df: pd.DataFrame, production_df: pd.DataFrame) -> list:
    drifted = []
    for col in FEATURE_COLS:
        if col not in production_df.columns:
            continue
        train_vals = training_df[col].dropna()
        prod_vals = production_df[col].dropna()
        if len(prod_vals) < 5:
            continue
        stat, p_value = ks_2samp(train_vals, prod_vals)
        status = "DRIFT" if p_value < DRIFT_THRESHOLD else "OK"
        info = {
            "feature": col,
            "ks_statistic": round(stat, 4),
            "p_value": round(p_value, 6),
            "status": status,
            "train_mean": round(float(train_vals.mean()), 2),
            "prod_mean": round(float(prod_vals.mean()), 2),
            "train_std": round(float(train_vals.std()), 2),
            "prod_std": round(float(prod_vals.std()), 2),
        }
        drifted.append(info)
        logger.info(
            "  %-25s  KS=%.4f  p=%.6f  %s  (train: %.2f, prod: %.2f)",
            col, stat, p_value, status,
            train_vals.mean(), prod_vals.mean(),
        )
    return drifted


def cmd_check_drift():
    logger.info("=== Data Drift Check ===")
    logger.info("Loading training data...")
    train_df = load_training_data(TRAINING_DATA_URL)
    logger.info("Training samples: %d", len(train_df))

    logger.info("Loading production data from logs...")
    prod_df = load_prediction_log(PREDICTION_LOG)
    if prod_df.empty:
        logger.warning("No production data found.")
        return

    logger.info("Production samples: %d", len(prod_df))
    logger.info("Drift threshold (p-value): %.4f", DRIFT_THRESHOLD)

    results = check_drift(train_df, prod_df)
    drifted = [r for r in results if r["status"] == "DRIFT"]

    if drifted:
        logger.warning(
            "DRIFT DETECTED in %d/%d features:", len(drifted), len(results)
        )
        for r in drifted:
            logger.warning("  %s (p=%.6f)", r["feature"], r["p_value"])
        sys.exit(1)
    else:
        logger.info("No drift detected. Model is stable.")
        sys.exit(0)


def cmd_analyze_logs():
    logger.info("=== Prediction Log Analysis ===")
    df = load_prediction_log(PREDICTION_LOG)
    if df.empty:
        logger.warning("No predictions logged yet.")
        return

    logger.info("Total predictions: %d", len(df))
    if "latency_ms" in df.columns:
        logger.info(
            "Latency: %.2fms avg, %.2fms max, %.2fms min",
            df["latency_ms"].mean(),
            df["latency_ms"].max(),
            df["latency_ms"].min(),
        )


def cmd_report():
    logger.info("=== Full Monitoring Report ===")
    cmd_analyze_logs()
    print()
    try:
        cmd_check_drift()
    except SystemExit:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model monitoring & drift detection")
    parser.add_argument("command", choices=["check-drift", "analyze-logs", "report"])
    args = parser.parse_args()

    if args.command == "check-drift":
        cmd_check_drift()
    elif args.command == "analyze-logs":
        cmd_analyze_logs()
    elif args.command == "report":
        cmd_report()
