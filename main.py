# -*- coding: utf-8 -*-
"""Main script orchestrating the solar power forecasting workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.baseline_models import (
    fit_linear_regression_baseline,
    predict_with_baseline,
)
from src.config import (
    DATA_PATHS,
    MAX_WINDOWS_PER_STATION,
    PRED_DISTANCE,
    WINDOW_SIZE,
)
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


def _aggregate_metrics(evaluation: Dict[int, Dict[str, object]]) -> Dict[str, object]:
    """Compute simple aggregates (mean RMSE/MAE/R2) and baseline summaries."""

    if not evaluation:
        return {}

    rmse = [float(metrics["rmse_overall"]) for metrics in evaluation.values()]
    mae = [float(metrics["mae_overall"]) for metrics in evaluation.values()]
    r2 = [float(metrics["r2_overall"]) for metrics in evaluation.values()]

    baseline_summary: Dict[str, Dict[str, list[float]]] = {}
    for metrics in evaluation.values():
        baselines = metrics.get("baselines", {})  # type: ignore[arg-type]
        for name, values in baselines.items():
            if not isinstance(values, dict):
                continue
            bucket = baseline_summary.setdefault(name, {"rmse": [], "mae": []})
            bucket["rmse"].append(float(values.get("rmse_overall", np.nan)))
            bucket["mae"].append(float(values.get("mae_overall", np.nan)))

    baselines_mean = {}
    for name, stats in baseline_summary.items():
        rmse_vals = np.asarray(stats["rmse"], dtype=float)
        mae_vals = np.asarray(stats["mae"], dtype=float)
        baselines_mean[name] = {
            "mean_rmse": float(np.nanmean(rmse_vals)) if rmse_vals.size else float("nan"),
            "mean_mae": float(np.nanmean(mae_vals)) if mae_vals.size else float("nan"),
        }

    return {
        "mean_rmse": float(np.nanmean(rmse)),
        "mean_mae": float(np.nanmean(mae)),
        "mean_r2": float(np.nanmean(r2)),
        "baselines": baselines_mean,
    }


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
    train_sets_map: Dict[int, WindowedDataset] = {}
    test_sets: Dict[int, WindowedDataset] = {}

    for station_id, frame in station_frames.items():
        if station_id == 3:
            continue

        try:
            dataset = create_dataset(frame, WINDOW_SIZE, PRED_DISTANCE)
        except ValueError as exc:
            print(f"Skipping station {station_id}: {exc}")
            continue

        dataset = dataset.limit(MAX_WINDOWS_PER_STATION)
        train_dataset, test_dataset = split_windowed_dataset(dataset, train_ratio=0.8)
        scaler = fit_scaler(train_dataset)
        train_scaled = apply_scaler(train_dataset, scaler)
        test_scaled = apply_scaler(test_dataset, scaler)

        train_sets.append(train_scaled)
        train_sets_map[station_id] = train_scaled
        test_sets[station_id] = test_scaled

    if not train_sets:
        raise RuntimeError("No training datasets prepared. Check data availability and preprocessing.")

    # Build and train the model
    n_features = train_sets[0].X.shape[-1]
    lstm_model = build_lstm_model(n_features=n_features, window_size=WINDOW_SIZE)
    loss_histories = train_model(lstm_model, train_sets)

    mean_loss_per_epoch: Tuple[float, ...] = tuple()
    if loss_histories:
        min_len = min(len(hist) for hist in loss_histories)
        if min_len > 0:
            trimmed = np.array([hist[:min_len] for hist in loss_histories], dtype=float)
            mean_loss_per_epoch = tuple(np.mean(trimmed, axis=0).tolist())
            print(
                "Average training loss per epoch:",
                ", ".join(f"{loss:.4f}" for loss in mean_loss_per_epoch),
            )

    # Evaluate against persistence baseline and log metrics
    lr_baseline = fit_linear_regression_baseline(train_sets)

    train_linear_preds = predict_with_baseline(lr_baseline, train_sets_map)
    test_linear_preds = predict_with_baseline(lr_baseline, test_sets)

    train_evaluation = evaluate_model(
        lstm_model,
        train_sets_map,
        label="Train",
        extra_baselines={"linear_regression": train_linear_preds},
        store_predictions=False,
    )
    test_evaluation = evaluate_model(
        lstm_model,
        test_sets,
        label="Test",
        extra_baselines={"linear_regression": test_linear_preds},
        store_predictions=True,
    )

    train_summary = _aggregate_metrics(train_evaluation)
    test_summary = _aggregate_metrics(test_evaluation)

    if train_summary and test_summary:
        rmse_gap = test_summary["mean_rmse"] - train_summary["mean_rmse"]  # type: ignore[index]
        mae_gap = test_summary["mean_mae"] - train_summary["mean_mae"]  # type: ignore[index]
        print(
            f"\nOverfit check: ΔRMSE={rmse_gap:.4f}, ΔMAE={mae_gap:.4f} "
            "(positive values indicate worse test performance)."
        )

    # Plot predictions for station 8 first horizon as a sample diagnostic
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    for station, metrics in test_evaluation.items():
        preds = np.asarray(metrics.get("predictions_actual"))
        targets = np.asarray(metrics.get("targets_actual"))
        if preds.size and targets.size:
            df_out = pd.DataFrame({"Predictions": preds[:, 0], "Actuals": targets[:, 0]})
            df_out.to_csv(outputs_dir / f"station_{station}_pred_vs_actual_h1.csv", index=False)

    if 8 in test_evaluation:
        station_eval = test_evaluation[8]
        preds = np.asarray(station_eval.get("predictions_actual"))
        targets = np.asarray(station_eval.get("targets_actual"))
        if preds.size and targets.size:
            preview_df = pd.DataFrame(
                {"Predictions": preds[:, 0], "Actuals": targets[:, 0]}
            )
            plot_predictions(preview_df, "Station 8 Forecast vs Actuals (Horizon 1)")
        else:
            print("[main] Station 8 evaluation lacks prediction/target data to plot.")

    summary_payload = {
        "train": train_summary,
        "test": test_summary,
        "mean_training_loss_per_epoch": list(mean_loss_per_epoch),
    }
    (outputs_dir / "metrics.json").write_text(json.dumps(summary_payload, indent=2))
    print(f"Saved summary metrics to {outputs_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
