# -*- coding: utf-8 -*-
"""Script for comprehensive model evaluation and comparison.

This script:
1. Trains the LSTM model with training history
2. Trains a simple linear regression baseline
3. Compares predictions vs actuals
4. Analyzes overfitting
5. Generates comparison visualizations
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
    fit_scaler,
    split_windowed_dataset,
    train_model,
    WindowedDataset,
)
from src.baseline_models import train_linear_regression_baseline
from src.model_comparison import (
    evaluate_with_training_history,
    compare_models,
    plot_model_comparison,
    plot_overfitting_analysis,
    save_detailed_predictions,
)
from src.visualization import plot_feature_correlations


def main() -> None:
    print("=" * 80)
    print("SOLAR POWER FORECASTING - COMPREHENSIVE MODEL EVALUATION")
    print("=" * 80)
    
    # Load and engineer features for every station
    print("\n[1/6] Loading and engineering features...")
    station_frames: Dict[int, pd.DataFrame] = {}
    for idx, path in enumerate(DATA_PATHS, start=1):
        df = load_data(path, idx)
        df = create_features_for_station(df).dropna()
        station_frames[idx] = df
        print(f"  Station {idx}: {len(df)} samples")

    plot_feature_correlations(list(station_frames.values()))

    # Prepare datasets per station (skip station 3 as in original notebook)
    print("\n[2/6] Preparing windowed datasets...")
    train_sets: list[WindowedDataset] = []
    test_sets: Dict[int, WindowedDataset] = {}

    for station_id, frame in station_frames.items():
        if station_id == 3:
            continue

        try:
            dataset = create_dataset(frame, WINDOW_SIZE, PRED_DISTANCE)
        except ValueError as exc:
            print(f"  Skipping station {station_id}: {exc}")
            continue

        train_dataset, test_dataset = split_windowed_dataset(dataset, train_ratio=0.8)
        scaler = fit_scaler(train_dataset)
        train_sets.append(apply_scaler(train_dataset, scaler))
        test_sets[station_id] = apply_scaler(test_dataset, scaler)
        print(f"  Station {station_id}: {len(train_dataset.X)} train, {len(test_dataset.X)} test samples")

    if not train_sets:
        raise RuntimeError("No training datasets prepared. Check data availability and preprocessing.")

    # Build and train the LSTM model
    print("\n[3/6] Training LSTM model...")
    n_features = train_sets[0].X.shape[-1]
    lstm_model = build_lstm_model(n_features=n_features, window_size=WINDOW_SIZE)
    loss_history = train_model(lstm_model, train_sets)
    
    # Evaluate LSTM with overfitting analysis
    print("\n[4/6] Analyzing LSTM model performance...")
    evaluation = evaluate_with_training_history(lstm_model, train_sets, test_sets, loss_history)
    
    print(f"\n  Average Training Loss: {evaluation['avg_train_loss']:.6f}")
    print(f"  Average Test Loss: {evaluation['avg_test_loss']:.6f}")
    print(f"  Test/Train Loss Ratio: {evaluation['overfitting_indicator']:.3f}")
    
    if evaluation['is_overfit']:
        print("  ⚠️  WARNING: Model shows signs of overfitting (test loss > 1.5x train loss)")
    else:
        print("  ✓  Model appears to generalize well")
    
    # Plot training loss history
    plot_overfitting_analysis(loss_history)
    
    # Train linear regression baseline
    print("\n[5/6] Training linear regression baseline...")
    baseline_predictions = train_linear_regression_baseline(train_sets, test_sets)
    print(f"  Linear regression trained on {sum(len(ds.X) for ds in train_sets)} samples")
    
    # Compare models
    print("\n[6/6] Comparing models and generating visualizations...")
    comparison_df = compare_models(lstm_model, baseline_predictions, test_sets)
    
    print("\nModel Comparison Summary:")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    
    # Calculate average metrics
    print("\n" + "=" * 80)
    print("AVERAGE METRICS ACROSS ALL STATIONS:")
    print("=" * 80)
    
    metrics = ['rmse', 'mae', 'r2']
    models = ['lstm', 'linear', 'persistence']
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        for model in models:
            col = f'{model}_{metric}'
            if col in comparison_df.columns:
                avg = comparison_df[col].mean()
                if metric == 'r2':
                    avg = comparison_df[col][comparison_df[col].notna()].mean()
                print(f"  {model.capitalize():15s}: {avg:.4f}")
    
    # Save detailed predictions
    save_detailed_predictions(lstm_model, baseline_predictions, test_sets)
    print("\n✓ Detailed predictions saved to outputs/detailed_predictions_station_*.csv")
    
    # Generate comparison plots for selected stations
    print("\nGenerating comparison visualizations...")
    for station_id in [1, 6, 8]:
        if station_id in test_sets:
            plot_model_comparison(
                lstm_model, 
                baseline_predictions, 
                test_sets, 
                station_id=station_id,
                horizon=0,
                num_points=200
            )
            print(f"  ✓ Comparison plot saved for station {station_id}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    print("\nOutputs saved to 'outputs/' directory:")
    print("  - model_comparison.csv: Metrics comparison table")
    print("  - detailed_predictions_station_*.csv: Detailed predictions for each station")
    print("  - comparison_station_*_h*.png: Visual comparisons")
    print("  - training_loss_history.png: Overfitting analysis")
    print("=" * 80)


if __name__ == "__main__":
    main()
