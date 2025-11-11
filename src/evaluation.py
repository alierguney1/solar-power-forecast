from typing import Dict, List

import numpy as np
import pandas as pd

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


def leave_one_station_out(
    station_dfs: Dict[int, pd.DataFrame],
    window_size: int,
    pred_distance: int,
    scale_ref_station: int = 6,
) -> Dict[int, pd.DataFrame]:
    """
    Train on all stations except one, evaluate on the left-out station.

    Returns a dict mapping left-out station -> results dataframe (Predictions vs Actuals).
    """
    stations = sorted(station_dfs.keys())
    results_by_leftout: Dict[int, pd.DataFrame] = {}

    for left in stations:
        train_stations = [s for s in stations if s != left]

        # Build windowed datasets for each training station
        train_windowed: List[WindowedDataset] = []
        for s in train_stations:
            ds = create_dataset(station_dfs[s], window_size, pred_distance)
            train_ds, _ = split_windowed_dataset(ds, train_ratio=0.8)
            scaler = fit_scaler(train_ds if train_ds.X.shape[0] > 0 else ds)
            train_windowed.append(apply_scaler(train_ds, scaler))

        if not train_windowed:
            continue

        # Build and train model
        n_features = train_windowed[0].X.shape[-1]
        model = build_lstm_model(n_features=n_features, window_size=window_size)
        train_model(model, train_windowed)

        # Prepare test set for left-out station
        ds_left = create_dataset(station_dfs[left], window_size, pred_distance)
        _, test_ds = split_windowed_dataset(ds_left, train_ratio=0.8)
        scaler = fit_scaler(train_windowed[0])
        test_ds = apply_scaler(test_ds, scaler)

        res = evaluate_model(model, {left: test_ds})
        metrics = res.get(left)
        if not metrics:
            continue

        preds = metrics.get("predictions_actual")
        actuals = metrics.get("targets_actual")
        if preds is None or actuals is None:
            continue

        preds_arr = np.asarray(preds)
        actuals_arr = np.asarray(actuals)
        results_by_leftout[left] = pd.DataFrame(
            {
                "prediction_h1": preds_arr[:, 0],
                "actual_h1": actuals_arr[:, 0],
            }
        )

    return results_by_leftout
