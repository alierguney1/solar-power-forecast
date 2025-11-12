# -*- coding: utf-8 -*-
"""Model utilities for building, training, and evaluating solar power forecasts."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, cast

import numpy as np
import pandas as pd
import sklearn.metrics as sklm
from keras.layers import LSTM, Dense, InputLayer
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.models import Model, Sequential
from keras.optimizers import Adam

from sklearn.preprocessing import MinMaxScaler

from src.config import (
    DROPOUT_RATE,
    EPOCHS,
    LEARNING_RATE,
    LSTM_UNITS,
    MAX_CAPACITIES,
    PRED_DISTANCE,
    RECURRENT_DROPOUT_RATE,
    WINDOW_SIZE,
)


@dataclass
class WindowedDataset:
    """Container for windowed features, targets, and metadata."""

    X: np.ndarray
    y: np.ndarray
    baselines: np.ndarray
    feature_names: Sequence[str]
    scaler: Optional[MinMaxScaler] = None

    def with_scaled_features(self, scaler: MinMaxScaler) -> "WindowedDataset":
        """Return a new dataset with features transformed by the supplied scaler."""
        X_scaled = scaler.transform(self.X.reshape(-1, self.X.shape[-1])).reshape(self.X.shape)
        return WindowedDataset(
            X=X_scaled,
            y=self.y,
            baselines=self.baselines,
            feature_names=self.feature_names,
            scaler=scaler,
        )

    def limit(self, max_windows: Optional[int]) -> "WindowedDataset":
        """Down-sample the dataset to at most ``max_windows`` entries while preserving order."""

        if max_windows is None or self.X.shape[0] <= max_windows:
            return self

        step = max(1, self.X.shape[0] // max_windows)
        indices = np.arange(0, self.X.shape[0], step)[:max_windows]

        return WindowedDataset(
            X=self.X[indices].copy(),
            y=self.y[indices].copy(),
            baselines=self.baselines[indices].copy(),
            feature_names=self.feature_names,
            scaler=self.scaler,
        )


def create_dataset(
    df: pd.DataFrame,
    window_size: int,
    pred_distance: int,
    target_column: str = "power",
) -> WindowedDataset:
    """Create chronologically ordered windows and targets for forecasting.

    Returns
    -------
    WindowedDataset
        Features (X), multi-step targets (y), persistence baseline, and metadata.
    """

    df = df.copy().reset_index(drop=True)

    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' not present in dataframe")

    feature_names = list(df.columns)
    target_values = df[target_column].to_numpy()
    values = df.to_numpy()

    max_start = len(df) - window_size - pred_distance + 1
    if max_start <= 0:
        raise ValueError(
            "Insufficient rows to create windows: "
            f"len(df)={len(df)}, window_size={window_size}, pred_distance={pred_distance}"
        )

    X, y, baselines = [], [], []
    target_idx = feature_names.index(target_column)

    for start in range(max_start):
        end = start + window_size
        horizon_end = end + pred_distance

        window = values[start:end, :]
        targets = target_values[end:horizon_end]

        X.append(window)
        y.append(targets)

        last_observed = window[-1, target_idx]
        baselines.append(np.full(pred_distance, last_observed, dtype=float))

    return WindowedDataset(
        X=np.asarray(X, dtype=float),
        y=np.asarray(y, dtype=float),
        baselines=np.asarray(baselines, dtype=float),
        feature_names=feature_names,
    )


def split_windowed_dataset(
    dataset: WindowedDataset, train_ratio: float = 0.8
) -> tuple[WindowedDataset, WindowedDataset]:
    """Split a windowed dataset into chronological train/test partitions."""

    total = dataset.X.shape[0]
    if total < 2:
        raise ValueError("Need at least two samples to perform a train/test split")

    split_index = int(total * train_ratio)
    split_index = min(max(split_index, 1), total - 1)

    def _slice(start: int, end: int) -> WindowedDataset:
        return WindowedDataset(
            X=dataset.X[start:end].copy(),
            y=dataset.y[start:end].copy(),
            baselines=dataset.baselines[start:end].copy(),
            feature_names=dataset.feature_names,
        )

    return _slice(0, split_index), _slice(split_index, total)


def build_lstm_model(n_features: int, window_size: int = WINDOW_SIZE) -> Model:
    """Build and compile the LSTM model with the provided feature dimension."""

    model = Sequential(
        [
            InputLayer((window_size, n_features)),
            LSTM(LSTM_UNITS, dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE),
            Dense(2 * PRED_DISTANCE, activation="relu"),
            Dense(PRED_DISTANCE, activation="relu"),
        ]
    )
    model.compile(
        loss=MeanSquaredError(),
        optimizer=Adam(learning_rate=LEARNING_RATE),  # type: ignore[arg-type]
        metrics=[RootMeanSquaredError()],
    )
    return model


def fit_scaler(windowed: WindowedDataset) -> MinMaxScaler:
    """Fit a MinMax scaler on a windowed dataset (flattening time dimension)."""

    scaler = MinMaxScaler()
    scaler.fit(windowed.X.reshape(-1, windowed.X.shape[-1]))
    return scaler


def apply_scaler(windowed: WindowedDataset, scaler: MinMaxScaler) -> WindowedDataset:
    """Return a dataset with features transformed by the provided scaler."""

    return windowed.with_scaled_features(scaler)


def train_model(
    model: Model,
    datasets: Sequence[WindowedDataset],
    epochs: Optional[int] = None,
    verbose: int = 1,
) -> List[List[float]]:
    """Train the model sequentially on multiple datasets."""

    loss_history: List[List[float]] = []
    total = len(datasets)
    use_epochs = EPOCHS if epochs is None else epochs

    for idx, dataset in enumerate(datasets, start=1):
        print(f"[train] Dataset {idx}/{total} - epochs={use_epochs}", flush=True)
        history = model.fit(
            dataset.X,
            dataset.y,
            epochs=use_epochs,
            verbose=cast(Any, verbose),
        )
        loss_history.append(history.history.get("loss", []))

    return loss_history


def actual_power(y_pred: np.ndarray, y_actual: np.ndarray, station: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert normalized predictions and labels back to MW using station capacity."""

    if station not in MAX_CAPACITIES:
        raise KeyError(f"Station {station} missing from MAX_CAPACITIES")

    factor = MAX_CAPACITIES[station]
    return y_pred * factor, y_actual * factor


def rmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error for flattened series."""

    return math.sqrt(sklm.mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))


def mae_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error for flattened series."""

    return sklm.mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared with graceful fallback for constant targets."""

    try:
        return sklm.r2_score(y_true.reshape(-1), y_pred.reshape(-1))
    except ValueError:
        return float("nan")


def _horizon_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    return [
        math.sqrt(sklm.mean_squared_error(y_true[:, h], y_pred[:, h]))
        for h in range(y_true.shape[1])
    ]


def _horizon_mae(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    return [sklm.mean_absolute_error(y_true[:, h], y_pred[:, h]) for h in range(y_true.shape[1])]


def _horizon_r2(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    scores: List[float] = []
    for h in range(y_true.shape[1]):
        try:
            scores.append(sklm.r2_score(y_true[:, h], y_pred[:, h]))
        except ValueError:
            scores.append(float("nan"))
    return scores


def evaluate_model(
    model: Model,
    datasets: Dict[int, WindowedDataset],
    *,
    label: str = "Evaluation",
    extra_baselines: Optional[Dict[str, Dict[int, np.ndarray]]] = None,
    store_predictions: bool = True,
) -> Dict[int, Dict[str, Any]]:
    """Evaluate the model and report metrics for each dataset.

    Parameters
    ----------
    datasets:
        Mapping of station id to dataset (train or test split).
    label:
        Label shown in logs (e.g. "Train", "Test").
    extra_baselines:
        Optional mapping of baseline name -> predictions per station (normalized units).
    store_predictions:
        When ``True`` include full prediction arrays in the result (can be large).
    """

    print(f"\n=== {label} metrics ===")

    evaluation: Dict[int, Dict[str, Any]] = {}
    rmse_values, mae_values, r2_values = [], [], []
    baseline_aggregates: Dict[str, Dict[str, List[float]]] = {}

    def _record_baseline_metric(name: str, rmse: float, mae: float) -> None:
        bucket = baseline_aggregates.setdefault(name, {"rmse": [], "mae": []})
        bucket["rmse"].append(rmse)
        bucket["mae"].append(mae)

    for station, dataset in datasets.items():
        preds = model.predict(dataset.X, verbose=0)
        preds_actual, y_actual_actual = actual_power(preds, dataset.y, station)

        station_metrics: Dict[str, Any] = {
            "rmse_per_horizon": _horizon_rmse(y_actual_actual, preds_actual),
            "mae_per_horizon": _horizon_mae(y_actual_actual, preds_actual),
            "r2_per_horizon": _horizon_r2(y_actual_actual, preds_actual),
            "rmse_overall": rmse_score(y_actual_actual, preds_actual),
            "mae_overall": mae_score(y_actual_actual, preds_actual),
            "r2_overall": r2_score(y_actual_actual, preds_actual),
        }

        baselines: Dict[str, Dict[str, Any]] = {}

        persistence_actual, _ = actual_power(dataset.baselines, dataset.y, station)
        baselines["persistence"] = {
            "rmse_overall": rmse_score(y_actual_actual, persistence_actual),
            "mae_overall": mae_score(y_actual_actual, persistence_actual),
            "rmse_per_horizon": _horizon_rmse(y_actual_actual, persistence_actual),
            "mae_per_horizon": _horizon_mae(y_actual_actual, persistence_actual),
        }
        _record_baseline_metric("persistence", baselines["persistence"]["rmse_overall"], baselines["persistence"]["mae_overall"])

        if extra_baselines:
            for name, preds_map in extra_baselines.items():
                if station not in preds_map:
                    continue
                baseline_pred = np.asarray(preds_map[station], dtype=float)
                baseline_actual, _ = actual_power(baseline_pred, dataset.y, station)
                baselines[name] = {
                    "rmse_overall": rmse_score(y_actual_actual, baseline_actual),
                    "mae_overall": mae_score(y_actual_actual, baseline_actual),
                    "rmse_per_horizon": _horizon_rmse(y_actual_actual, baseline_actual),
                    "mae_per_horizon": _horizon_mae(y_actual_actual, baseline_actual),
                }
                _record_baseline_metric(name, baselines[name]["rmse_overall"], baselines[name]["mae_overall"])

        if store_predictions:
            station_metrics["predictions_actual"] = preds_actual
            station_metrics["targets_actual"] = y_actual_actual
            baseline_actuals: Dict[str, Optional[np.ndarray]] = {"persistence": persistence_actual}
            if extra_baselines:
                for key, preds_map in extra_baselines.items():
                    if station in preds_map:
                        baseline_actuals[key], _ = actual_power(
                            np.asarray(preds_map[station], dtype=float), dataset.y, station
                        )
                    else:
                        baseline_actuals[key] = None
            station_metrics["baseline_actuals"] = baseline_actuals

        station_metrics["baselines"] = baselines
        evaluation[station] = station_metrics

        rmse_values.append(station_metrics["rmse_overall"])
        mae_values.append(station_metrics["mae_overall"])
        r2_values.append(station_metrics["r2_overall"])

        best_baseline = min(baselines.items(), key=lambda item: item[1]["rmse_overall"])
        print(
            f"Station {station}: RMSE={station_metrics['rmse_overall']:.3f} | "
            f"Best baseline ({best_baseline[0]}): {best_baseline[1]['rmse_overall']:.3f}; "
            f"MAE={station_metrics['mae_overall']:.3f}; R2={station_metrics['r2_overall']:.3f}"
        )

    if evaluation:
        print("\nAggregate metrics across stations:")
        print(f"  Mean RMSE: {float(np.mean(rmse_values)):.3f}")
        print(f"  Mean MAE: {float(np.mean(mae_values)):.3f}")
        print(f"  Mean R²:  {float(np.nanmean(r2_values)):.3f}")
        for baseline_name, values in baseline_aggregates.items():
            print(
                f"  {baseline_name.title()} RMSE: {float(np.mean(values['rmse'])):.3f} | "
                f"MAE: {float(np.mean(values['mae'])):.3f}"
            )

    return evaluation
