import pandas as pd
from pathlib import Path

base = Path('/workspaces/solar-power-forecast')
files = [base/f'data{i}.csv' for i in range(1,9)]
cols = ['Time(year-month-day h:m:s)', 'Total solar irradiance (W/m2)', 'Direct normal irradiance (W/m2)', 'Global horizontal irradiance (W/m2)', 'Air temperature  (°C)', 'Atmosphere (hpa)', 'Relative humidity (%)', 'Power (MW)']

for i, f in enumerate(files, start=1):
    try:
        df = pd.read_csv(f)
        # Try to coerce power to numeric
        power = pd.to_numeric(df.get('Power (MW)'), errors='coerce')
        print(f'Station {i}: rows={len(df)}, power_min={power.min():.4f}, power_max={power.max():.4f}, power_mean={power.mean():.4f}')
    except Exception as e:
        print(f'Station {i}: failed to read {f}: {e}')
