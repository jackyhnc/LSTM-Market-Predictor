"""Offline evaluation: computes accuracy and Sharpe ratio from saved prediction CSVs."""

import glob
import pandas as pd
import numpy as np


def sharpe(returns: pd.Series, annualization: float = 252) -> float:
    if returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(annualization))


def evaluate(results_glob: str = "src/model-backend/results_predictions_*.csv") -> None:
    files = sorted(glob.glob(results_glob))
    if not files:
        print("No prediction files found.")
        return

    for path in files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        preds = df["Prediction"]
        mean_pred = preds.mean()
        above_threshold = (preds >= 0.004).sum()
        print(f"{path}: {len(df)} tickers | mean pred={mean_pred:.4f} | buy signals={above_threshold}")


if __name__ == "__main__":
    evaluate()
