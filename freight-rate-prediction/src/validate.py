"""Validation: forward-chained (time-based) backtest + unseen-city stress test.

Why time-based? The real task trains on Jan-Oct 2025 and predicts Nov-Dec 2025 (the future).
A random split would let the model see loads from the same days it is tested on, which
overstates accuracy. Each fold below trains ONLY on the past and predicts the next 2 months,
which mimics the real task exactly (a 2-month horizon).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import (
    build_daily_market_index,
    build_daily_quote_signal,
    build_features,
    fit_weight_imputation,
)
from model import TwoStageFreightRateModel

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# (test_start, test_end): train = every load BEFORE test_start
FORWARD_FOLDS = [
    ("2025-05-01", "2025-06-30"),
    ("2025-07-01", "2025-08-31"),
    ("2025-09-01", "2025-10-31"),
]
SPIKE_RATE_PER_MILE = 5.0   # ~0.6% of loads have 5-14 $/mile vs a typical ~2.2: unpredictable label spikes


def error_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    absolute_error = np.abs(actual - predicted)
    return {
        "MAE ($)": absolute_error.mean(),
        "RMSE ($)": np.sqrt(((actual - predicted) ** 2).mean()),
        "MAPE (%)": 100 * (absolute_error / actual).mean(),
        "Median APE (%)": 100 * np.median(absolute_error / actual),
    }


# ---------------- baselines -----------------------------------------------------
def naive_baseline_predict(train, test):
    """Median rate-per-mile by equipment x distance bucket. No time, no market, no geography."""
    bins = [0, 150, 300, 600, 1200, 2400, 10_000]
    train_group = pd.DataFrame(
        {"eq": train["equipment"], "bucket": pd.cut(train["distance"], bins), "rpm": train["posted_rate"] / train["distance"]}
    )
    median_rpm = train_group.groupby(["eq", "bucket"], observed=True)["rpm"].median()
    keys = pd.MultiIndex.from_arrays([test["equipment"], pd.cut(test["distance"], bins)])
    return median_rpm.reindex(keys).to_numpy() * test["distance"].to_numpy()


def single_stage_boosting_predict(train_features, test_features, train, test):
    """Same features and loss as the final model, but a single boosting stage (no linear trend)."""
    model = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=500, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=60, categorical_features="from_dtype", random_state=0,
    ).fit(train_features, np.log(train["posted_rate"] / train["distance"]))
    return np.exp(model.predict(test_features)) * test["distance"].to_numpy()


def two_stage_predict(train_features, test_features, train, test):
    model = TwoStageFreightRateModel(random_state=0).fit(
        train_features, train["posted_rate"], train["distance"], train["date"]
    )
    return model.predict(test_features, test["distance"], test["date"])


def evaluate_split(train, test, daily_market, daily_quote):
    """Fit everything on `train`, score on `test`. Returns a dict of {model_name: predictions}."""
    weight_fill = fit_weight_imputation(train)            # learned from the past only
    train_features = build_features(train, daily_market, daily_quote, weight_fill)
    test_features = build_features(test, daily_market, daily_quote, weight_fill)
    return {
        "Naive (equipment x distance bucket)": naive_baseline_predict(train, test),
        "Single-stage boosting": single_stage_boosting_predict(train_features, test_features, train, test),
        "Two-stage (final)": two_stage_predict(train_features, test_features, train, test),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    development = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])

    # Label-free daily market signals (built from feature columns only, so no target leakage).
    daily_market = build_daily_market_index(development, validation)
    daily_quote = build_daily_quote_signal(development, validation)

    # ---------------- 1) forward-chained backtest ---------------------------------
    records = []
    for test_start, test_end in FORWARD_FOLDS:
        train = development[development["date"] < test_start]
        test = development[(development["date"] >= test_start) & (development["date"] <= test_end)]
        actual = test["posted_rate"].to_numpy()
        is_clean = (actual / test["distance"].to_numpy()) < SPIKE_RATE_PER_MILE
        for model_name, predicted in evaluate_split(train, test, daily_market, daily_quote).items():
            records.append({"fold": f"{test_start[:7]}..{test_end[:7]}", "model": model_name, "subset": "all loads",
                            **error_metrics(actual, predicted)})
            records.append({"fold": f"{test_start[:7]}..{test_end[:7]}", "model": model_name, "subset": "excluding rate spikes",
                            **error_metrics(actual[is_clean], predicted[is_clean])})
    by_fold = pd.DataFrame(records)
    by_fold.to_csv(OUTPUT_DIR / "backtest_by_fold.csv", index=False)
    summary = by_fold.drop(columns="fold").groupby(["subset", "model"], sort=False).mean().round(2)
    summary.to_csv(OUTPUT_DIR / "backtest_summary.csv")
    print("=== Forward-chained backtest (mean of 3 folds) ===")
    print(summary.to_string())

    # ---------------- 2) unseen-city stress test ----------------------------------
    # validation.csv has 8 cities never seen in training. Simulate that: hide 8 random cities
    # from training entirely, then test (on the latest fold) only on loads touching those cities.
    rng = np.random.default_rng(42)
    all_cities = sorted(set(development["pickup"]) | set(development["delivery"]))
    hidden_cities = set(rng.choice(all_cities, size=8, replace=False))
    touches_hidden = development["pickup"].isin(hidden_cities) | development["delivery"].isin(hidden_cities)
    last_start = FORWARD_FOLDS[-1][0]
    train = development[(development["date"] < last_start) & ~touches_hidden]
    test = development[(development["date"] >= last_start) & touches_hidden]
    actual = test["posted_rate"].to_numpy()
    rows = {name: error_metrics(actual, pred) for name, pred in evaluate_split(train, test, daily_market, daily_quote).items()}
    unseen = pd.DataFrame(rows).T.round(2)
    unseen.to_csv(OUTPUT_DIR / "unseen_city_stress_test.csv")
    print(f"\n=== Unseen-city stress test (hidden cities: {sorted(hidden_cities)}; {len(test)} test loads) ===")
    print(unseen.to_string())


if __name__ == "__main__":
    main()
