from src.data_loader import load_data
from src.feature_engineering import create_features_for_station


def test_feature_engineering_adds_columns():
    df = load_data('data1.csv', 1)
    feats = create_features_for_station(df)
    # Expect hourly cyclical/lag columns
    expected = ['hour', 'hour_sin', 'hour_cos', 'lag_1', 'lag_24']
    for col in expected:
        assert col in feats.columns
    # STL features should be present if enough data
    assert 'stl_trend_daily' in feats.columns
    assert 'stl_seasonal_daily' in feats.columns
