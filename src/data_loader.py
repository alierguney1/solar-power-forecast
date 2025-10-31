# -*- coding: utf-8 -*-
"""
This module handles loading and initial preprocessing of the solar power data.
"""
import pandas as pd
import numpy as np
from scipy import stats
from src.config import MAX_CAPACITIES, NEW_COLUMN_NAMES, DATA_PATHS

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

def load_all_data():
    """
    Loads all datasets, attaches station metadata, and concatenates them.
    """
    # Station metadata (from provided table) - fill missing values with None
    station_meta = {
        1: {'capacity': 50, 'panel_model': None, 'num_panels': None},
        2: {'capacity': 130, 'panel_model': None, 'num_panels': None},
        3: {'capacity': 30, 'panel_model': 'CS6U-325P', 'num_panels': 27995},
        4: {'capacity': 130, 'panel_model': None, 'num_panels': None},
        5: {'capacity': 110, 'panel_model': 'JNMP60-255', 'num_panels': 36828},
        6: {'capacity': 35, 'panel_model': 'SUN2000-50KTL-C', 'num_panels': 703},
        7: {'capacity': 30, 'panel_model': None, 'num_panels': 60},
        8: {'capacity': 0.93, 'panel_model': 'HR-260P-18/Bbd', 'num_panels': 3567}
    }

    dfs = {}
    for idx, path in enumerate(DATA_PATHS, start=1):
        try:
            df_i = load_data(path, idx)
            df_i = df_i.copy()
            df_i['station'] = idx
            df_i['capacity'] = station_meta[idx]['capacity']
            dfs[idx] = df_i
            print(f"Loaded station {idx}: {len(df_i)} rows")
        except Exception as e:
            print(f"Failed to load {path} (station {idx}): {e}")

    return dfs
