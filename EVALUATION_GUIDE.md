# Model Evaluation and Comparison

This document explains how to use the comprehensive model evaluation and comparison functionality.

## Overview

The `evaluate_models.py` script provides a complete analysis of the solar power forecasting model, including:

1. **Predictions vs Actuals Visualization**: Compare what the model predicts vs actual values
2. **Baseline Comparison**: Compare LSTM performance against simple linear regression and persistence baselines
3. **Overfitting Analysis**: Check if the model is overfitting by comparing training and test losses

## Running the Evaluation

Simply run:

```bash
python evaluate_models.py
```

This will:
- Load and prepare data for all stations
- Train the LSTM model and track training history
- Train a linear regression baseline model
- Generate comprehensive comparisons and visualizations

## Outputs

All outputs are saved to the `outputs/` directory:

### 1. Model Comparison Table (`model_comparison.csv`)

Shows metrics for each model across all stations:
- **LSTM Model**: Deep learning LSTM network
- **Linear Regression**: Simple baseline using last timestep features
- **Persistence**: Naive baseline that assumes next value = current value

Metrics included:
- RMSE (Root Mean Squared Error): Lower is better
- MAE (Mean Absolute Error): Lower is better
- R² (R-Squared): Higher is better (closer to 1.0)

### 2. Detailed Predictions (`detailed_predictions_station_*.csv`)

For each station, saves a CSV with:
- `actual`: True power values
- `lstm_pred`: LSTM model predictions
- `linear_pred`: Linear regression predictions
- `persistence_pred`: Persistence baseline predictions

### 3. Visual Comparisons (`comparison_station_*_h*.png`)

Line plots showing:
- Black line: Actual power values
- Blue line: LSTM predictions
- Red dashed line: Linear regression predictions
- Green dotted line: Persistence predictions

These plots make it easy to visually assess model performance.

### 4. Overfitting Analysis (`training_loss_history.png`)

Shows training loss over epochs for each dataset. This helps identify:
- **Decreasing loss**: Model is learning
- **Flat/increasing loss**: Model may have converged or is overfitting
- **Compare with test loss ratio**: If test loss >> train loss, model is overfitting

## Understanding the Results

### Overfitting Indicators

The script automatically checks for overfitting by comparing:
- Average training loss
- Average test loss

If test loss > 1.5x training loss, it warns about potential overfitting.

### Model Performance

Good performance indicators:
- **Low RMSE/MAE**: Predictions close to actual values
- **High R²**: Model explains variance well (close to 1.0)
- **Better than baselines**: LSTM should outperform linear regression and persistence

### Example Interpretation

If you see:
```
Average Training Loss: 0.000245
Average Test Loss: 0.000389
Test/Train Loss Ratio: 1.587
⚠️  WARNING: Model shows signs of overfitting
```

This suggests the model memorized training data but doesn't generalize as well to test data.

Solutions could include:
- Increase dropout rate
- Add more regularization
- Reduce model complexity
- Get more training data

## Customization

You can modify `evaluate_models.py` to:
- Change which stations to visualize (line 140-146)
- Adjust number of points in plots (line 145, `num_points` parameter)
- Change prediction horizon to visualize (line 145, `horizon` parameter)
- Modify train/test split ratio (line 68, `train_ratio` parameter)

## Integration with Main Pipeline

The original `main.py` still works as before. The new `evaluate_models.py` is specifically for detailed analysis and comparison. Use it when you want to:
- Understand model performance in detail
- Compare against baselines
- Check for overfitting
- Generate publication-ready visualizations
