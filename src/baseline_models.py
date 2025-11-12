# -*- coding: utf-8 -*-
"""Baseline models for comparison with the LSTM model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression

from src.model import WindowedDataset


@dataclass
class LinearRegressionBaseline:
    """Multi-output linear regression baseline that predicts each horizon step."""

    models: Sequence[LinearRegression]

    def predict(self, dataset: WindowedDataset) -> np.ndarray:
        """Predict normalized power for each horizon on the given dataset."""

        X_last = dataset.X[:, -1, :]
        preds = np.column_stack([model.predict(X_last) for model in self.models])
        return preds


def fit_linear_regression_baseline(train_datasets: Sequence[WindowedDataset]) -> LinearRegressionBaseline:
    """Fit a horizon-wise linear regression using the last timestep features."""

    X_train = np.vstack([dataset.X[:, -1, :] for dataset in train_datasets])
    y_train = np.vstack([dataset.y for dataset in train_datasets])

    models = []
    for horizon in range(y_train.shape[1]):
        model = LinearRegression()
        model.fit(X_train, y_train[:, horizon])
        models.append(model)

    return LinearRegressionBaseline(models=models)


def predict_with_baseline(
    baseline: LinearRegressionBaseline,
    datasets: Dict[int, WindowedDataset],
) -> Dict[int, np.ndarray]:
    """Generate predictions for each dataset using the provided baseline."""

    return {station_id: baseline.predict(ds) for station_id, ds in datasets.items()}
