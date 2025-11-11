import pandas as pd
from src.data_loader import load_data


def test_load_data_shapes_and_columns():
    df = load_data('data1.csv', 1)
    assert not df.empty
    # Expect at least these base columns
    for col in ['tsi', 'dni', 'ghi', 'temp', 'atm', 'rh', 'power', 'Day sin', 'Day cos', 'Year sin', 'Year cos']:
        assert col in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df.index)
