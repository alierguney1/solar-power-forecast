# How to Use the Model Evaluation Features

This guide shows you exactly how to see model predictions, compare with actuals, compare against simple regression, and check for overfitting.

## Quick Start

Run the comprehensive evaluation script:

```bash
python evaluate_models.py
```

This single command will:
1. Train the LSTM model
2. Train a linear regression baseline
3. Generate all comparisons and visualizations
4. Check for overfitting
5. Save everything to the `outputs/` directory

## What You Get

### 1. Seeing Predictions vs Actuals

**Visual Comparisons** (`outputs/comparison_station_*.png`):
- Black line = Actual power values
- Blue line = LSTM predictions
- Red dashed line = Linear regression predictions
- Green dotted line = Persistence baseline predictions

These plots clearly show how well each model tracks the actual values.

**Detailed Data** (`outputs/detailed_predictions_station_*.csv`):
Each file contains:
```csv
actual,lstm_pred,persistence_pred,linear_pred
1.79,1.2462857,1.01,1.5180155101443704
1.91,2.401481,1.79,2.389939747551758
...
```

You can load these into Excel or any data analysis tool for further investigation.

### 2. Comparing Against Simple Regression

**Metrics Table** (`outputs/model_comparison.csv`):
```
Station | LSTM RMSE | Linear RMSE | LSTM R² | Linear R²
   1    |   6.88    |    5.52     |  0.640  |  0.768
   8    |   2.87    |    1.71     |  0.803  |  0.930
```

**Current Results**:
- ✅ LSTM beats Persistence baseline
- ❌ Linear Regression beats LSTM in most cases
- 🤔 This suggests the LSTM is overfitting

**Metrics Comparison Chart** (`outputs/model_metrics_comparison.png`):
Side-by-side bar charts showing RMSE, MAE, and R² for all models across all stations.

### 3. Checking for Overfitting

**Training Loss History** (`outputs/training_loss_history.png`):
Shows how loss decreases over epochs during training. Look for:
- ✅ Steady decrease = model is learning
- ⚠️ Plateaus or increases = may be overfitting

**Overfitting Indicator**:
The script automatically calculates:
```
Average Training Loss: 0.000229
Average Test Loss: 0.000365
Test/Train Loss Ratio: 1.594

⚠️  WARNING: Model shows signs of overfitting (test loss > 1.5x train loss)
```

**What This Means**:
- Ratio < 1.3: Model generalizes well ✅
- Ratio 1.3-1.5: Minor overfitting, probably okay ⚠️
- Ratio > 1.5: Significant overfitting, needs fixing ❌

## Interpreting Results

### Good Signs
- LSTM RMSE < Linear RMSE
- LSTM R² > Linear R²
- Test/Train ratio < 1.3
- Predictions visually track actuals well

### Warning Signs (Current State)
- ❌ Linear regression outperforms LSTM (current state)
- ❌ Test/Train ratio = 1.59 (current state)
- ⚠️ LSTM predictions more volatile than linear

### What to Try Next

**To Fix Overfitting**:
1. Increase dropout rate in `src/config.py`
2. Add early stopping
3. Reduce LSTM units
4. Get more training data

**To Improve Performance**:
1. Feature engineering (add more relevant features)
2. Hyperparameter tuning (grid search)
3. Ensemble LSTM + Linear models
4. Station-specific models

## Customizing the Evaluation

Edit `evaluate_models.py` to:

**Change stations to visualize** (line 140-146):
```python
for station_id in [1, 6, 8]:  # Change these numbers
    if station_id in test_sets:
        plot_model_comparison(...)
```

**Change number of points in plots** (line 145):
```python
plot_model_comparison(..., num_points=500)  # Default is 200
```

**Change prediction horizon** (line 145):
```python
plot_model_comparison(..., horizon=1)  # 0=first horizon, 1=second, etc.
```

## Example Workflow

```bash
# 1. Run evaluation
python evaluate_models.py

# 2. Check the summary
cat outputs/model_comparison.csv

# 3. Look at visualizations
ls outputs/*.png

# 4. Analyze detailed predictions
head -20 outputs/detailed_predictions_station_8.csv

# 5. Read the summary report
cat MODEL_EVALUATION_SUMMARY.md
```

## Files Reference

| File | Purpose |
|------|---------|
| `evaluate_models.py` | Main evaluation script |
| `src/baseline_models.py` | Linear regression baseline implementation |
| `src/model_comparison.py` | Comparison utilities |
| `EVALUATION_GUIDE.md` | Detailed documentation |
| `MODEL_EVALUATION_SUMMARY.md` | Current evaluation results and insights |

## Questions?

- **Q: Why is linear regression better?**  
  A: The LSTM is overfitting. It memorizes training data instead of learning generalizable patterns.

- **Q: How do I fix overfitting?**  
  A: Increase regularization (dropout, L2), reduce model complexity, or get more data.

- **Q: Should I use the linear model instead?**  
  A: For now, yes, it generalizes better. But LSTM should work better with proper tuning.

- **Q: Can I combine both models?**  
  A: Yes! Ensemble methods often work well. Average predictions or use linear features + LSTM predictions.
