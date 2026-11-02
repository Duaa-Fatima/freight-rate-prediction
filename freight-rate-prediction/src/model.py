"""Two-stage freight rate model.

Stage 1 (linear, robust)  : captures the smooth parts that must EXTRAPOLATE into the future:
                            time trend, market index level, distance scaling, equipment premium.
Stage 2 (gradient boosting): learns the non-linear leftovers on the residual of stage 1:
                            lanes / geography, weight effects, short-haul premium, interactions.

Why two stages? Tree models cannot extrapolate: beyond the last training date they return the
last level they saw. The data has a slow upward trend (~+3% over the year), so a pure tree model
under-predicts future months by ~3%. A linear stage 1 fixes that (backtest MAE 125 -> 118).

The target is log(rate per mile): rates scale with distance, and log makes errors relative (%),
which suits a quantity that ranges from $57 to $25,000.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

TIME_ORIGIN = pd.Timestamp("2025-01-01")   # day 0 of the time trend


class TwoStageFreightRateModel:
    def __init__(self, random_state: int = 0):
        self.random_state = random_state

    # ----- stage 1 design matrix -------------------------------------------------
    @staticmethod
    def _linear_design_matrix(features: pd.DataFrame, dates: pd.Series) -> np.ndarray:
        days_since_origin = (pd.to_datetime(dates) - TIME_ORIGIN).dt.days.to_numpy() / 100.0
        equipment = features["equipment"].astype(str)
        return np.column_stack(
            [
                features["market_index_daily"].to_numpy(),
                features["log_distance"].to_numpy(),
                days_since_origin,                               # the trend that trees cannot extrapolate
                (equipment == "Reefer").to_numpy().astype(float),
                (equipment == "Flatbed").to_numpy().astype(float),  # Dry Van is the baseline category
            ]
        )

    def fit(self, features: pd.DataFrame, rates: pd.Series, distances: pd.Series, dates: pd.Series):
        log_rate_per_mile = np.log(rates.to_numpy() / distances.to_numpy())

        # Stage 1: Huber loss is robust to the ~0.6% label spikes (rates 3-6x normal) that would
        # otherwise drag an ordinary least-squares line.
        design = self._linear_design_matrix(features, dates)
        self.scaler_ = StandardScaler().fit(design)
        self.linear_stage_ = HuberRegressor(epsilon=1.35, alpha=0.0, max_iter=500).fit(
            self.scaler_.transform(design), log_rate_per_mile
        )
        stage1_prediction = self.linear_stage_.predict(self.scaler_.transform(design))

        # Stage 2: boosting on what stage 1 could not explain.
        # absolute_error loss = median regression, again robust to spikes (RMSE-style losses chase them).
        # Small leaves + large min_samples_leaf = strong regularisation; the signal is smooth.
        self.boosting_stage_ = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=500,
            learning_rate=0.05,
            max_leaf_nodes=15,
            min_samples_leaf=60,
            categorical_features="from_dtype",   # equipment is a pandas Categorical
            early_stopping=True,                 # stops adding trees once an internal 10% hold-out stops improving
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=self.random_state,
        ).fit(features, log_rate_per_mile - stage1_prediction)
        return self

    def predict(self, features: pd.DataFrame, distances: pd.Series, dates: pd.Series) -> np.ndarray:
        design = self._linear_design_matrix(features, dates)
        log_rate_per_mile = self.linear_stage_.predict(self.scaler_.transform(design)) + self.boosting_stage_.predict(features)
        return np.exp(log_rate_per_mile) * distances.to_numpy()   # back to dollars: rate = rate_per_mile * miles
