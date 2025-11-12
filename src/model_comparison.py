# -*- coding: utf-8 -*-
"""Enhanced evaluation and comparison script for model predictions."""
import os
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model

from src.model import (
    WindowedDataset,
    actual_power,
    rmse_score,
    mae_score,
    r2_score,
)


def evaluate_with_training_history(
    model: Model,
    train_datasets: list,
    test_datasets: Dict[int, WindowedDataset],
    loss_history: List[List[float]]
) -> Dict:
    """
    Evaluate model performance including overfitting analysis.
    
    Args:
        model: Trained Keras model
        train_datasets: List of training datasets
        test_datasets: Dictionary of test datasets by station
        loss_history: Training loss history
        
    Returns:
        Dictionary containing evaluation metrics and overfitting indicators
    """
    results = {
        'train_loss_history': loss_history,
        'stations': {}
    }
    
    # Calculate training set performance
    train_losses = []
    for dataset in train_datasets:
        preds = model.predict(dataset.X, verbose=0)
        train_loss = np.mean((preds - dataset.y) ** 2)
        train_losses.append(train_loss)
    
    results['avg_train_loss'] = float(np.mean(train_losses))
    
    # Calculate test set performance
    test_losses = []
    for station_id, test_dataset in test_datasets.items():
        preds = model.predict(test_dataset.X, verbose=0)
        test_loss = np.mean((preds - test_dataset.y) ** 2)
        test_losses.append(test_loss)
    
    results['avg_test_loss'] = float(np.mean(test_losses))
    
    # Check for overfitting
    loss_ratio = results['avg_test_loss'] / results['avg_train_loss']
    results['overfitting_indicator'] = loss_ratio
    results['is_overfit'] = loss_ratio > 1.5  # If test loss is 50% higher than train
    
    return results


def compare_models(
    lstm_model: Model,
    baseline_predictions: Dict[int, np.ndarray],
    test_datasets: Dict[int, WindowedDataset],
    output_dir: str = "outputs"
) -> pd.DataFrame:
    """
    Compare LSTM model with baseline predictions.
    
    Args:
        lstm_model: Trained LSTM model
        baseline_predictions: Dictionary of baseline predictions by station
        test_datasets: Dictionary of test datasets by station
        output_dir: Directory to save comparison results
        
    Returns:
        DataFrame with comparison metrics
    """
    comparison_data = []
    
    for station_id, test_dataset in test_datasets.items():
        # LSTM predictions
        lstm_preds = lstm_model.predict(test_dataset.X, verbose=0)
        lstm_preds_actual, y_actual = actual_power(lstm_preds, test_dataset.y, station_id)
        
        # Baseline predictions
        baseline_preds = baseline_predictions.get(station_id)
        if baseline_preds is not None:
            baseline_preds_actual, _ = actual_power(baseline_preds, test_dataset.y, station_id)
        else:
            baseline_preds_actual = None
        
        # Persistence baseline
        persistence_actual, _ = actual_power(test_dataset.baselines, test_dataset.y, station_id)
        
        # Calculate metrics
        lstm_rmse = rmse_score(y_actual, lstm_preds_actual)
        lstm_mae = mae_score(y_actual, lstm_preds_actual)
        lstm_r2 = r2_score(y_actual, lstm_preds_actual)
        
        persistence_rmse = rmse_score(y_actual, persistence_actual)
        persistence_mae = mae_score(y_actual, persistence_actual)
        
        row = {
            'station': station_id,
            'lstm_rmse': lstm_rmse,
            'lstm_mae': lstm_mae,
            'lstm_r2': lstm_r2,
            'persistence_rmse': persistence_rmse,
            'persistence_mae': persistence_mae,
        }
        
        if baseline_preds_actual is not None:
            linear_rmse = rmse_score(y_actual, baseline_preds_actual)
            linear_mae = mae_score(y_actual, baseline_preds_actual)
            linear_r2 = r2_score(y_actual, baseline_preds_actual)
            
            row.update({
                'linear_rmse': linear_rmse,
                'linear_mae': linear_mae,
                'linear_r2': linear_r2,
            })
        
        comparison_data.append(row)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    return comparison_df


def plot_model_comparison(
    lstm_model: Model,
    baseline_predictions: Dict[int, np.ndarray],
    test_datasets: Dict[int, WindowedDataset],
    station_id: int = 8,
    horizon: int = 0,
    num_points: int = 200,
    output_dir: str = "outputs"
):
    """
    Create visualization comparing LSTM, baseline, and actual values.
    
    Args:
        lstm_model: Trained LSTM model
        baseline_predictions: Dictionary of baseline predictions
        test_datasets: Dictionary of test datasets
        station_id: Station to visualize
        horizon: Prediction horizon to visualize
        num_points: Number of points to plot
        output_dir: Directory to save plots
    """
    if station_id not in test_datasets:
        print(f"Station {station_id} not in test datasets")
        return
    
    test_dataset = test_datasets[station_id]
    
    # Get predictions
    lstm_preds = lstm_model.predict(test_dataset.X, verbose=0)
    lstm_preds_actual, y_actual = actual_power(lstm_preds, test_dataset.y, station_id)
    
    baseline_preds = baseline_predictions.get(station_id)
    if baseline_preds is not None:
        baseline_preds_actual, _ = actual_power(baseline_preds, test_dataset.y, station_id)
    
    persistence_actual, _ = actual_power(test_dataset.baselines, test_dataset.y, station_id)
    
    # Create plot
    plt.figure(figsize=(15, 8))
    
    n = min(num_points, len(y_actual))
    x_range = range(n)
    
    plt.plot(x_range, y_actual[:n, horizon], 'k-', linewidth=2, label='Actual', alpha=0.8)
    plt.plot(x_range, lstm_preds_actual[:n, horizon], 'b-', linewidth=1.5, label='LSTM', alpha=0.7)
    
    if baseline_preds is not None:
        plt.plot(x_range, baseline_preds_actual[:n, horizon], 'r--', linewidth=1.5, label='Linear Regression', alpha=0.7)
    
    plt.plot(x_range, persistence_actual[:n, horizon], 'g:', linewidth=1.5, label='Persistence', alpha=0.6)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Power (MW)', fontsize=12)
    plt.title(f'Model Comparison - Station {station_id}, Horizon {horizon+1}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'comparison_station_{station_id}_h{horizon}.png'), dpi=150)
    plt.show()


def plot_overfitting_analysis(
    loss_history: List[List[float]],
    output_dir: str = "outputs"
):
    """
    Plot training loss history to visualize overfitting.
    
    Args:
        loss_history: List of loss histories for each dataset
        output_dir: Directory to save plots
    """
    plt.figure(figsize=(12, 6))
    
    for i, losses in enumerate(loss_history):
        if losses:
            plt.plot(losses, label=f'Dataset {i+1}', alpha=0.7)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss (MSE)', fontsize=12)
    plt.title('Training Loss History - Overfitting Analysis', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_loss_history.png'), dpi=150)
    plt.show()


def save_detailed_predictions(
    lstm_model: Model,
    baseline_predictions: Dict[int, np.ndarray],
    test_datasets: Dict[int, WindowedDataset],
    output_dir: str = "outputs"
):
    """
    Save detailed predictions for each station including all models.
    
    Args:
        lstm_model: Trained LSTM model
        baseline_predictions: Dictionary of baseline predictions
        test_datasets: Dictionary of test datasets
        output_dir: Directory to save predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for station_id, test_dataset in test_datasets.items():
        # LSTM predictions
        lstm_preds = lstm_model.predict(test_dataset.X, verbose=0)
        lstm_preds_actual, y_actual = actual_power(lstm_preds, test_dataset.y, station_id)
        
        # Baseline predictions
        baseline_preds = baseline_predictions.get(station_id)
        if baseline_preds is not None:
            baseline_preds_actual, _ = actual_power(baseline_preds, test_dataset.y, station_id)
        
        # Persistence baseline
        persistence_actual, _ = actual_power(test_dataset.baselines, test_dataset.y, station_id)
        
        # Create DataFrame with all predictions (first horizon only for simplicity)
        data = {
            'actual': y_actual[:, 0],
            'lstm_pred': lstm_preds_actual[:, 0],
            'persistence_pred': persistence_actual[:, 0],
        }
        
        if baseline_preds is not None:
            data['linear_pred'] = baseline_preds_actual[:, 0]
        
        df = pd.DataFrame(data)
        df.to_csv(
            os.path.join(output_dir, f'detailed_predictions_station_{station_id}.csv'),
            index=False
        )
