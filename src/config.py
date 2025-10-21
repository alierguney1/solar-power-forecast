# -*- coding: utf-8 -*-
"""
This file contains all the configurations and constants for the solar power forecast project.
"""

# Maximum Capacity of the Solar Farms (in MW)
MAX_CAPACITIES = {
    1: 50,
    2: 130,
    3: 30,
    4: 130,
    5: 110,
    6: 35,
    7: 30,
    8: 30
}

# Column names mapping
COLUMN_MAPPING = {
    'Time(year-month-day h:m:s)': 'time',
    'Total solar irradiance (W/m2)': 'tsi',
    'Direct normal irradiance (W/m2)': 'dni',
    'Global horizontal irradiance (W/m2)': 'ghi',
    'Air temperature  (°C)': 'temp',
    'Atmosphere (hpa)': 'atm',
    'Relative humidity (%)': 'rh',
    'Power (MW)': 'power'
}

NEW_COLUMN_NAMES = [
    'time', 'tsi', 'dni', 'ghi', 'temp', 'atm', 'rh', 'power'
]

# Time series parameters
WINDOW_SIZE = 8
PRED_DISTANCE = 8
N_FEATURES = 11

# Model parameters
LSTM_UNITS = 16
DROPOUT_RATE = 0.001
RECURRENT_DROPOUT_RATE = 0.001
LEARNING_RATE = 0.0001
EPOCHS = 3

# File paths
DATA_PATHS = [f"data{i}.csv" for i in range(1, 9)]
