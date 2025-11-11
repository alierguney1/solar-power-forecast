# Solar Power Forecast

LSTM-based forecasting for solar power generation across multiple plants/datasets.

## What’s inside
- `eda.ipynb`: Notebook dedicated to Exploratory Data Analysis (EDA) only. It loads all CSVs, shows per-station summaries, and provides interactive plots. No model training code lives here.
- `main.py`: Runs the end-to-end training and evaluation pipeline using the code under `src/`.
- `src/`
	- `config.py`: Constants and paths (window sizes, capacities, etc.).
	- `data_loader.py`: CSV loading, normalization (per-station capacity), time encodings, and helpers to load all datasets.
	- `feature_engineering.py`: Reusable feature-building utilities (hourly features, daily STL merges, etc.).
	- `model.py`: LSTM dataset windowing, model build/train/eval utilities.
	- `visualization.py`: Matplotlib/Seaborn helpers for charts and diagnostics.

## Quick start
1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Explore the data (EDA only)
- Open `eda.ipynb` in Jupyter and execute the cells.
- The notebook will:
	- Load all datasets (`data1.csv` … `data8.csv`)
	- Show per-station summary statistics
	- Provide interactive time-series plots per station

3) Train and evaluate the model
```bash
python main.py
```
This will load data, build the LSTM, train, and evaluate on held-out splits.

4) Use the CLI (artifacts saved to `outputs/`)
```bash
python -m src.cli summarize -o outputs
python -m src.cli train-basic -o outputs --random-state 42
```
The CLI saves `summary.csv`, per-station prediction CSVs, and a `metrics.json`.

## Notes
- Power is normalized by plant capacity during loading (see `src/config.py::MAX_CAPACITIES`).
- Feature engineering functions are reusable and live under `src/feature_engineering.py`; the notebook imports them only to preview features as part of EDA.
- For cross-station generalization, see `src/evaluation.py` (leave-one-station-out helper).

## Tests
Run unit tests with:
```bash
pytest -q
```
