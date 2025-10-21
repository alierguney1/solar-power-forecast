# -*- coding: utf-8 -*-
"""
Main script for the solar power forecast project.
"""
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import DATA_PATHS, WINDOW_SIZE, PRED_DISTANCE
from src.data_loader import load_data
from src.model import create_dataset, build_lstm_model, train_model, preprocess_features, evaluate_model
from src.visualization import plot_predictions, plot_feature_correlations

def main():
    """
    Main function to run the solar power forecast workflow.
    """
    # Load data for all farms
    dataframes = [load_data(path, i + 1) for i, path in enumerate(DATA_PATHS)]

    # We decided to not include df3 to train the model
    train_dfs = [df for i, df in enumerate(dataframes) if i + 1 != 3]
    
    # Plot feature correlations for all datasets
    plot_feature_correlations(dataframes)

    # Create datasets
    datasets = [create_dataset(df, WINDOW_SIZE, PRED_DISTANCE) for df in train_dfs]
    
    # Split data into training and testing sets
    train_sets = []
    test_sets = {}
    
    farm_indices = [1, 2, 4, 5, 6, 7, 8]
    for i, (X, y) in enumerate(datasets):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=False)
        
        # Preprocess features
        # Using the 6th dataframe for scaling parameters as in the original notebook
        X_train = preprocess_features(X_train, dataframes[5])
        X_test = preprocess_features(X_test, dataframes[5])
        
        train_sets.append((X_train, y_train))
        test_sets[farm_indices[i]] = (X_test, y_test)

    # Build and train the model
    lstm_model = build_lstm_model()
    train_model(lstm_model, train_sets)

    # Evaluate the model
    all_results = evaluate_model(lstm_model, test_sets)

    # Plot predictions for a sample test set
    if 8 in all_results:
        plot_predictions(all_results[8], "Test Predictions vs Actuals for Farm 8")

if __name__ == "__main__":
    main()
