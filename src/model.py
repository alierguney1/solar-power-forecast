# -*- coding: utf-8 -*-
"""
This module contains functions for building, training, and evaluating the prediction models.
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.layers import InputLayer, LSTM, Dense
from keras.models import Sequential
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
import math

from src.config import WINDOW_SIZE, PRED_DISTANCE, N_FEATURES, LSTM_UNITS, DROPOUT_RATE, RECURRENT_DROPOUT_RATE, LEARNING_RATE, EPOCHS, MAX_CAPACITIES

def create_dataset(df, window_size, pred_distance):
    """
    Creates a dataset for time series forecasting.

    Args:
        df (pd.DataFrame): The input dataframe.
        window_size (int): The size of the input window.
        pred_distance (int): The prediction distance.

    Returns:
        tuple: A tuple containing input and output arrays (np.array, np.array).
    """
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    inputs = []
    outputs = []
    for i in range(len(df) - window_size):
        X_list = []
        y_list = []

        for j in range(window_size):
            X_list.append(df.loc[(i - j) + window_size - 1, :])
        inputs.append(np.array(X_list))

        for k in range(pred_distance):
            y_list.append(df['power'][i + k + 1])
        outputs.append(np.array(y_list))

    return np.array(inputs), np.array(outputs)

def build_lstm_model():
    """
    Builds the LSTM model.

    Returns:
        keras.Model: The compiled LSTM model.
    """
    model = Sequential()
    model.add(InputLayer((WINDOW_SIZE, N_FEATURES)))
    model.add(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE, recurrent_dropout=RECURRENT_DROPOUT_RATE))
    model.add(Dense(8))
    model.add(Dense(PRED_DISTANCE, 'relu'))
    model.summary()
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=LEARNING_RATE), metrics=[RootMeanSquaredError()])
    return model

def train_model(model, datasets):
    """
    Trains the LSTM model on a list of datasets.

    Args:
        model (keras.Model): The LSTM model to train.
        datasets (list): A list of tuples, where each tuple contains (X_train, y_train).

    Returns:
        list: A list of loss histories from training.
    """
    loss_list = []
    for X_train, y_train in datasets:
        history = model.fit(X_train, y_train, epochs=EPOCHS)
        loss_list.append(history.history["loss"])
    return loss_list

def preprocess_features(X, df_source):
    """
    Preprocesses the feature data by scaling it.

    Args:
        X (np.array): The input features.
        df_source (pd.DataFrame): The source dataframe for scaling parameters.

    Returns:
        np.array: The preprocessed features.
    """
    j = 0
    for i in df_source.columns:
        X[:, :, j] = ((X[:, :, j] - (df_source[i].min())) / (df_source[i].max() - df_source[i].min()))
        if j <= len(df_source.columns) - 2:
            j = j + 1
    return X

def actual_power(y_pred, y_actual, number):
    """
    Converts normalized power predictions back to actual power values.

    Args:
        y_pred (np.array): The predicted power values.
        y_actual (np.array): The actual power values.
        number (int): The solar farm number.

    Returns:
        tuple: A tuple of actual predicted power and actual power (np.array, np.array).
    """
    actual_pred = y_pred * MAX_CAPACITIES[number]
    actual_actual = y_actual * MAX_CAPACITIES[number]
    return actual_pred, actual_actual

def evaluate_model(model, test_sets):
    """
    Evaluates the model on the given test sets and prints the metrics.

    Args:
        model (keras.Model): The trained model.
        test_sets (dict): A dictionary where keys are farm numbers and values are (X_test, y_test).
    """
    all_results = {}
    mean_rmse = 0
    mean_r2 = 0
    
    for number, (X_test, y_test) in test_sets.items():
        test_predictions = model.predict(X_test)
        test_predictions_actual, y_test_actual = actual_power(test_predictions, y_test, number)
        
        rmse = rmse_score(y_test_actual, test_predictions_actual)
        r2 = r2_score(y_test_actual, test_predictions_actual)
        
        mean_rmse += rmse
        mean_r2 += r2
        
        print(f"RMSE Score of test set {number}: {rmse}")
        print(f"R-Squared Score of test set {number}: {r2}")
        
        results_df = pd.DataFrame(data={
            'Predictions': test_predictions_actual[:, 0].flatten().tolist(),
            'Actuals': y_test_actual[:, 0].flatten().tolist()
        })
        all_results[number] = results_df

    print(f"\nMean of RMSE: {mean_rmse / len(test_sets)}")
    print(f"Mean of R-squared: {mean_r2 / len(test_sets)}")
    
    return all_results

def rmse_score(y_true, y_predicted):
    """Calculates the Root Mean Squared Error."""
    return math.sqrt(sklm.mean_squared_error(y_true[:,0].flatten().tolist(), y_predicted[:,0].flatten().tolist()))

def r2_score(y_true, y_predicted):
    """Calculates the R-squared score."""
    return sklm.r2_score(y_true[:,0].flatten().tolist(), y_predicted[:,0].flatten().tolist())
