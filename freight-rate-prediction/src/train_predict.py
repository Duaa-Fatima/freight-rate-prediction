"""Train on ALL labeled data (Jan-Oct 2025) and write the two required prediction files.

Outputs
  validation_predictions.csv         -> exactly: load_id,predicted_rate  (12,000 rows, template order)
  outputs/december_predictions.csv   -> the completed December chart inputs (feeds score.py)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features import (
    build_daily_market_index,
    build_daily_quote_signal,
    build_features,
    fit_weight_imputation,
)
from model import TwoStageFreightRateModel

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


def build_december_rows(december_inputs: pd.DataFrame, development: pd.DataFrame) -> pd.DataFrame:
    """Expand the fixed-lane December file into full feature rows.

    The file only has pickup, delivery, distance, equipment, weight, date. The model also needs
    coordinates (looked up from the city names in the labeled data) and the market signals
    (market_index / quote_signal), which are filled by build_features from the daily tables
    built on validation.csv for those exact dates. Only the date changes between rows.
    """
    city_coordinates = pd.concat(
        [
            development[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1),
            development[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1),
        ]
    ).drop_duplicates("city").set_index("city")

    rows = december_inputs.copy()
    rows["pickup_lat"] = rows["pickup"].map(city_coordinates["lat"])
    rows["pickup_lon"] = rows["pickup"].map(city_coordinates["lon"])
    rows["delivery_lat"] = rows["delivery"].map(city_coordinates["lat"])
    rows["delivery_lon"] = rows["delivery"].map(city_coordinates["lon"])
    assert rows[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].notna().all().all(), "unknown December city"
    return rows


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    development = pd.read_csv(DATA_DIR / "train_test.csv", parse_dates=["date"])
    validation = pd.read_csv(DATA_DIR / "validation.csv", parse_dates=["date"])
    template = pd.read_csv(DATA_DIR / "validation_predictions_template.csv")
    december_inputs = pd.read_csv(DATA_DIR / "december_chart_inputs.csv")

    # Label-free daily market signals (feature columns only; see features.py).
    daily_market = build_daily_market_index(development, validation)
    daily_quote = build_daily_quote_signal(development, validation)
    weight_fill = fit_weight_imputation(development)

    # ---- fit on every labeled load ------------------------------------------------
    train_features = build_features(development, daily_market, daily_quote, weight_fill)
    model = TwoStageFreightRateModel(random_state=0).fit(
        train_features, development["posted_rate"], development["distance"], development["date"]
    )

    # ---- validation.csv -> validation_predictions.csv ----------------------------
    validation_features = build_features(validation, daily_market, daily_quote, weight_fill)
    validation_rates = model.predict(validation_features, validation["distance"], validation["date"])
    predictions = pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": np.round(validation_rates, 2)})
    # Re-order to the template so that row order matches exactly what the scorer expects.
    predictions = template[["load_id"]].merge(predictions, on="load_id", how="left")
    assert predictions["predicted_rate"].notna().all() and len(predictions) == 12_000
    assert (predictions["predicted_rate"] > 0).all()
    predictions.to_csv("validation_predictions.csv", index=False)

    # ---- December chart ----------------------------------------------------------
    december_rows = build_december_rows(december_inputs, development)
    december_features = build_features(december_rows, daily_market, daily_quote, weight_fill)
    december_rows["predicted_rate"] = np.round(
        model.predict(december_features, december_rows["distance"].astype(float), december_rows["date"]), 2
    )
    december_output = december_inputs.copy()
    december_output["predicted_rate"] = december_rows["predicted_rate"].values
    december_output.to_csv(OUTPUT_DIR / "december_predictions.csv", index=False)

    # ---- sanity checks printed for the report ------------------------------------
    train_rpm = (development["posted_rate"] / development["distance"]).median()
    validation_rpm = (predictions.merge(validation[["load_id", "distance"]], on="load_id").eval("predicted_rate / distance")).median()
    print(f"Median $/mile  train actual: {train_rpm:.3f} | validation predicted: {validation_rpm:.3f}")
    print(f"Validation predicted_rate  min={predictions.predicted_rate.min():.2f}  median={predictions.predicted_rate.median():.2f}  max={predictions.predicted_rate.max():.2f}")
    print("December predictions: min={:.2f} max={:.2f} mean={:.2f}".format(
        december_output.predicted_rate.min(), december_output.predicted_rate.max(), december_output.predicted_rate.mean()))
    print("Wrote validation_predictions.csv and outputs/december_predictions.csv")


if __name__ == "__main__":
    main()
