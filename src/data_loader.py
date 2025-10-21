# -*- coding: utf-8 -*-
"""
This module handles loading and initial preprocessing of the solar power data.
"""
import pandas as pd
import numpy as np
from scipy import stats
from src.config import MAX_CAPACITIES, NEW_COLUMN_NAMES

def load_data(filepath, data_number):
    """
    Loads and preprocesses data from a given CSV file.

    Args:
        filepath (str): The path to the CSV file.
        data_number (int): The number of the solar farm.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    df = pd.read_csv(filepath,
                     dtype={
                         'tsi': float,
                         'dni': float,
                         'ghi': float,
                         'temp': float,
                         'atm': float,
                         'rh': float,
                         'power': float
                     })
    print(f"\nLoading data from: {filepath}\n")

    df.columns = NEW_COLUMN_NAMES

    # Convert all columns except the first one to float
    cols = df.columns[1:]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Normalize power by the maximum capacity of the plant
    df["power"] = df["power"] / MAX_CAPACITIES[data_number]

    # Convert to datetime object and set as index
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.dropna(inplace=True)

    # Create Daily and Yearly sin-cos values for temporal context
    df["Seconds"] = df.index.map(pd.Timestamp.timestamp)
    seconds_in_day = 60 * 60 * 24
    seconds_in_year = seconds_in_day * 365.2425
    df['Day sin'] = (np.sin(df['Seconds'] * (2 * np.pi / seconds_in_day)) + 1) / 2
    df['Day cos'] = (np.cos(df['Seconds'] * (2 * np.pi / seconds_in_day)) + 1) / 2
    df['Year sin'] = (np.sin(df['Seconds'] * (2 * np.pi / seconds_in_year)) + 1) / 2
    df['Year cos'] = (np.cos(df['Seconds'] * (2 * np.pi / seconds_in_year)) + 1) / 2
    df = df.drop('Seconds', axis=1)

    # Remove Outlier Values (Z score > 3)
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    print(df.describe())
    print(df.info())
    return df
