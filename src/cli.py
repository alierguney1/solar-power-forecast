import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import DATA_PATHS, WINDOW_SIZE, PRED_DISTANCE
from src.data_loader import load_data
from src.feature_engineering import create_features_for_station
from src.model import (
    apply_scaler,
    build_lstm_model,
    create_dataset,
    evaluate_model,
    fit_scaler,
    split_windowed_dataset,
    train_model,
    WindowedDataset,
)


def cmd_summarize(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    dataframes = []
    for i, path in enumerate(DATA_PATHS, start=1):
        try:
            df = load_data(path, i)
            dataframes.append(df)
            summaries.append({
                'station': i,
                'rows': len(df),
                'power_min': float(df['power'].min()),
                'power_max': float(df['power'].max()),
                'power_mean': float(df['power'].mean()),
            })
        except Exception as e:
            summaries.append({'station': i, 'error': str(e)})
    pd.DataFrame(summaries).to_csv(out_dir / 'summary.csv', index=False)
    print(f"Saved summary to {out_dir/'summary.csv'}")


def cmd_train_basic(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[cli] Loading and engineering data...", flush=True)
    station_frames = []
    for i, p in enumerate(DATA_PATHS, start=1):
        df = load_data(p, i)
        df = create_features_for_station(df).dropna()
        station_frames.append((i, df))

    print("[cli] Creating windowed datasets and splits...", flush=True)
    train_sets: list[WindowedDataset] = []
    test_sets: dict[int, WindowedDataset] = {}
    for station_id, df in station_frames:
        if station_id == 3:
            continue
        dataset = create_dataset(df, WINDOW_SIZE, PRED_DISTANCE)
        train_ds, test_ds = split_windowed_dataset(dataset, train_ratio=0.8)
        scaler = fit_scaler(train_ds)
        train_sets.append(apply_scaler(train_ds, scaler))
        test_sets[station_id] = apply_scaler(test_ds, scaler)

    print("[cli] Building model...", flush=True)
    n_features = train_sets[0].X.shape[-1]
    model = build_lstm_model(n_features=n_features, window_size=WINDOW_SIZE)
    print(f"[cli] Training model on {len(train_sets)} stations (epochs={args.epochs})...", flush=True)
    train_model(model, train_sets, epochs=args.epochs, verbose=1)
    print("[cli] Evaluating...", flush=True)
    results: dict[int, dict] = evaluate_model(model, test_sets)

    # Save sample predictions and simple metrics summary
    summary = {k: {'rmse': float(v['rmse_overall']), 'mae': float(v['mae_overall'])} for k, v in results.items()}
    for k, val in results.items():
        preds_arr = val['predictions_actual']
        targets_arr = val['targets_actual']
        df_out = pd.DataFrame({'Predictions': preds_arr[:, 0], 'Actuals': targets_arr[:, 0]})
        df_out.to_csv(out_dir / f'predictions_station_{k}.csv', index=False)

    (out_dir / 'metrics.json').write_text(json.dumps({'stations': sorted(list(results.keys())), 'summary': summary}, indent=2))
    print(f"Saved predictions and metrics to {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(description='Solar Power Forecast CLI')
    sub = p.add_subparsers(dest='command', required=True)

    ps = sub.add_parser('summarize', help='Summarize datasets and save CSV')
    ps.add_argument('-o', '--output-dir', default='outputs', help='Output directory')
    ps.set_defaults(func=cmd_summarize)

    pt = sub.add_parser('train-basic', help='Train basic LSTM and save predictions/metrics')
    pt.add_argument('-o', '--output-dir', default='outputs', help='Output directory')
    pt.add_argument('--random-state', type=int, default=None, help='Random state for split')
    pt.add_argument('--epochs', type=int, default=None, help='Override training epochs (default from config)')
    pt.set_defaults(func=cmd_train_basic)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
