"""Feature engineering for the freight rate model.

Every step below has a "why" comment. The same code is used for the backtest,
the final model, and the December chart, so there is no train/serve mismatch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_WEIGHT_LB = 47_500.0          # the data is capped at this value (1,200+ rows sit exactly on it)
EARTH_RADIUS_MILES = 3958.8

EQUIPMENT_TYPES = ["Dry Van", "Reefer", "Flatbed"]


def haversine_miles(lat1, lon1, lat2, lon2):
    """Straight-line distance between two points on Earth, in miles.

    Why: the given `distance` is a noisy road distance (it varies for the same lane)
    and is floored at 70 miles. A clean geometric distance gives the model a stable
    second view of how far the load really goes.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def build_daily_market_index(*feature_frames: pd.DataFrame) -> pd.Series:
    """Median `market_index` for every calendar date, using ONLY feature columns (no labels).

    Why: market_index is one market-wide number per day plus small per-row noise
    (std ~0.025). The daily median removes that noise and also fills the ~1% of rows where
    it is missing. It uses no `posted_rate`, so using validation.csv here is not leakage.
    It is also how we get a market_index for the December chart dates.
    """
    all_rows = pd.concat(
        [frame[["date", "market_index"]] for frame in feature_frames], ignore_index=True
    )
    all_rows["date"] = pd.to_datetime(all_rows["date"])
    return all_rows.groupby("date")["market_index"].median()


def build_daily_quote_signal(*feature_frames: pd.DataFrame) -> pd.Series:
    """Median `quote_signal` for every (date, equipment) pair, using ONLY feature columns.

    Why: quote_signal is a mix of a market-level component (date x equipment explains ~32%
    of its variance, far above the ~2% expected by chance) and per-load noise. The group median
    keeps the market part and drops the noise. Backtest MAE: 116 vs 118 with the raw value.
    Bonus: it can be computed for the December chart, whose input file has no quote_signal.
    """
    all_rows = pd.concat(
        [frame[["date", "equipment", "quote_signal"]] for frame in feature_frames], ignore_index=True
    )
    all_rows["date"] = pd.to_datetime(all_rows["date"])
    return all_rows.groupby(["date", "equipment"])["quote_signal"].median()


def fit_weight_imputation(train_frame: pd.DataFrame) -> dict:
    """Median absolute weight per equipment type, learned from training rows only.

    Why: weight is missing in ~0.6% of rows. Equipment type is the best simple predictor of weight.
    """
    absolute_weight = train_frame["weight"].abs()
    return absolute_weight.groupby(train_frame["equipment"]).median().to_dict()


def build_features(
    frame: pd.DataFrame,
    daily_market_index: pd.Series,
    daily_quote_signal: pd.Series,
    median_weight_by_equipment: dict,
) -> pd.DataFrame:
    """Turn raw load rows into the model's feature matrix."""
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    out = pd.DataFrame(index=df.index)

    # ---- Distance --------------------------------------------------------------
    # log(distance) because rate scales roughly proportionally with miles (corr 0.97 in logs).
    out["log_distance"] = np.log(df["distance"])
    straight_line = haversine_miles(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    )
    out["log_straight_line_miles"] = np.log(straight_line.clip(lower=1.0))
    # Detour ratio: how much longer the road is than the crow flies (mountains, no direct highway...).
    out["road_to_straight_ratio"] = (df["distance"] / straight_line.clip(lower=1.0)).clip(upper=3.0)
    # Short hauls are priced very differently ($/mile is higher). Flag the 70-mile floor.
    out["is_distance_floor"] = (df["distance"] <= 70.0).astype(int)

    # ---- Geography -------------------------------------------------------------
    # Raw coordinates instead of city names: 8 validation cities never appear in training,
    # and a model that only knows city IDs cannot price them. Coordinates generalise.
    out["pickup_lat"] = df["pickup_lat"]
    out["pickup_lon"] = df["pickup_lon"]
    out["delivery_lat"] = df["delivery_lat"]
    out["delivery_lon"] = df["delivery_lon"]
    # Direction of travel: freight rates are asymmetric (headhaul vs backhaul).
    out["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
    out["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]

    # ---- Weight ----------------------------------------------------------------
    # Negative weights are sign errors: a truck cannot carry -36,559 lb. Take the absolute value.
    out["weight_was_negative"] = (df["weight"] < 0).astype(int)
    weight_missing = df["weight"].isna()
    out["weight_was_missing"] = weight_missing.astype(int)
    fallback_weight = df["equipment"].map(median_weight_by_equipment)
    clean_weight = df["weight"].abs().where(~weight_missing, fallback_weight)
    out["weight"] = clean_weight
    # The cap is a censoring artefact: "47,500" really means "47,500 or more".
    out["weight_at_cap"] = (clean_weight >= MAX_WEIGHT_LB).astype(int)

    # ---- Equipment -------------------------------------------------------------
    # Ordered fixed category list so that train / validation / December get identical codes.
    out["equipment"] = pd.Categorical(df["equipment"], categories=EQUIPMENT_TYPES)

    # ---- Date / market ---------------------------------------------------------
    # Day of week captures the weekly cycle seen in the market index.
    out["day_of_week"] = df["date"].dt.dayofweek
    # Clean market index: daily median (see build_daily_market_index).
    out["market_index_daily"] = df["date"].map(daily_market_index)
    # Not using month / day_of_year on purpose: trees cannot extrapolate to unseen months
    # (Nov-Dec are not in training), whereas market_index carries the seasonal signal.

    # ---- Quote signal ----------------------------------------------------------
    # Denoised market quote for this date and equipment type (see build_daily_quote_signal).
    quote_key = pd.MultiIndex.from_arrays([df["date"].values, df["equipment"].values])
    out["quote_signal_daily"] = daily_quote_signal.reindex(quote_key).values

    return out
