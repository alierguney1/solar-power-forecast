# -*- coding: utf-8 -*-
"""Main script orchestrating the solar power forecasting workflow.

This script loads station CSVs, creates engineered features, builds
windowed datasets, fits per-station scalers (fit on train only), trains
a global LSTM, and evaluates against a persistence baseline.
"""

from typing import Dict

import numpy as np
import pandas as pd

from src.config import DATA_PATHS, WINDOW_SIZE, PRED_DISTANCE
from src.data_loader import load_data
from src.feature_engineering import create_features_for_station
from src.model import (
    apply_scaler,
    build_lstm_model,
    create_dataset,
    evaluate_model,
    fit_scaler,
    split_windowed_dataset,
    train_model,
    WindowedDataset,
)
from src.visualization import plot_feature_correlations, plot_predictions


def main() -> None:
    # Load and engineer features for every station
    station_frames: Dict[int, pd.DataFrame] = {}
    for idx, path in enumerate(DATA_PATHS, start=1):
        df = load_data(path, idx)
        df = create_features_for_station(df).dropna()
        station_frames[idx] = df

    plot_feature_correlations(list(station_frames.values()))

    # Prepare datasets per station (skip station 3 as in original notebook)
    train_sets: list[WindowedDataset] = []
    test_sets: Dict[int, WindowedDataset] = {}

    for station_id, frame in station_frames.items():
        if station_id == 3:
            continue

        try:
            dataset = create_dataset(frame, WINDOW_SIZE, PRED_DISTANCE)
        except ValueError as exc:
            print(f"Skipping station {station_id}: {exc}")
            continue

        train_dataset, test_dataset = split_windowed_dataset(dataset, train_ratio=0.8)
        scaler = fit_scaler(train_dataset)
        train_sets.append(apply_scaler(train_dataset, scaler))
        test_sets[station_id] = apply_scaler(test_dataset, scaler)

    if not train_sets:
        raise RuntimeError("No training datasets prepared. Check data availability and preprocessing.")

    # Build and train the model
    n_features = train_sets[0].X.shape[-1]
    lstm_model = build_lstm_model(n_features=n_features, window_size=WINDOW_SIZE)
    train_model(lstm_model, train_sets)

    # Evaluate against persistence baseline and log metrics
    evaluation = evaluate_model(lstm_model, test_sets)

    # Plot predictions for station 8 first horizon as a sample diagnostic
    if 8 in evaluation:
        station_eval = evaluation[8]
        preds = np.asarray(station_eval.get("predictions_actual"))
        targets = np.asarray(station_eval.get("targets_actual"))
        if preds.size and targets.size:
            preview_df = pd.DataFrame(
                {"Predictions": preds[:, 0], "Actuals": targets[:, 0]}
            )
            plot_predictions(preview_df, "Station 8 Forecast vs Actuals (Horizon 1)")
        else:
            print("[main] Station 8 evaluation lacks prediction/target data to plot.")


if __name__ == "__main__":
    main()
