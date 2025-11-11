# -*- coding: utf-8 -*-
"""Baseline models for comparison with the LSTM model."""
from typing import Dict
import numpy as np
from sklearn.linear_model import LinearRegression
from src.model import WindowedDataset


def train_linear_regression_baseline(
    train_datasets: list,
    test_datasets: Dict[int, WindowedDataset]
) -> Dict[int, np.ndarray]:
    """
    Train a simple linear regression model for each station.
    Uses the last timestep features to predict the target horizon.
    
    Args:
        train_datasets: List of training WindowedDataset objects
        test_datasets: Dictionary mapping station_id to test WindowedDataset
    
    Returns:
        Dictionary mapping station_id to predictions array
    """
    predictions = {}
    
    # Combine all training data
    X_train_list = []
    y_train_list = []
    
    for dataset in train_datasets:
        # Use the last timestep features as input
        X_train_list.append(dataset.X[:, -1, :])  # Last timestep
        y_train_list.append(dataset.y)
    
    X_train = np.vstack(X_train_list)
    y_train = np.vstack(y_train_list)
    
    # Train a linear regression model for each horizon
    models = []
    n_horizons = y_train.shape[1]
    
    for h in range(n_horizons):
        model = LinearRegression()
        model.fit(X_train, y_train[:, h])
        models.append(model)
    
    # Make predictions for each test station
    for station_id, test_dataset in test_datasets.items():
        X_test = test_dataset.X[:, -1, :]  # Last timestep
        preds = np.zeros((X_test.shape[0], n_horizons))
        
        for h, model in enumerate(models):
            preds[:, h] = model.predict(X_test)
        
        predictions[station_id] = preds
    
    return predictions
