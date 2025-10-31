import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

def make_hourly_features(df):
    """
    Generates hourly features for a single-station DataFrame.
    """
    df = df.copy()
    # Basic time features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    # Lag features
    for lag in [1, 2, 3, 4, 6, 12, 24]:
        df[f'lag_{lag}'] = df['power'].shift(lag)
    # Rolling statistics
    df['roll_mean_3'] = df['power'].shift(1).rolling(3, min_periods=1).mean()
    df['roll_std_3'] = df['power'].shift(1).rolling(3, min_periods=1).std().fillna(0)
    df['roll_mean_24'] = df['power'].shift(1).rolling(24, min_periods=1).mean()
    return df.dropna()

def add_daily_stl_features(df):
    """
    Computes daily STL decomposition features and merges them back.
    """
    # Compute daily STL on daily-mean aggregated series
    daily = df['power'].resample('D').mean().dropna()
    if len(daily) < 2 * 7:  # Not enough data for STL
        return df
        
    per = 365 if len(daily) >= 365 else 7
    stl_res = STL(daily, period=per, robust=True).fit()
    
    daily_feats = pd.DataFrame({
        'stl_trend_daily': stl_res.trend,
        'stl_seasonal_daily': stl_res.seasonal
    })
    
    # Map daily features back to hourly by joining on date
    df = df.copy()
    df['date'] = df.index.normalize()
    daily_feats = daily_feats.reset_index()
    # Handle arbitrary index name robustly (e.g., 'time' vs default 'index')
    date_source_col = daily_feats.columns[0]
    daily_feats['date'] = pd.to_datetime(daily_feats[date_source_col]).dt.normalize()
    daily_feats = daily_feats.drop(columns=[date_source_col]).set_index('date')
    
    df = df.join(daily_feats[['stl_trend_daily', 'stl_seasonal_daily']], on='date')
    df['stl_trend_daily'] = df['stl_trend_daily'].ffill().bfill()
    df['stl_seasonal_daily'] = df['stl_seasonal_daily'].ffill().bfill()
    df = df.drop(columns=['date'])
    
    return df

def create_features_for_station(df):
    """
    Creates a complete feature set for a single station's data.
    """
    df = make_hourly_features(df)
    df = add_daily_stl_features(df)
    return df
