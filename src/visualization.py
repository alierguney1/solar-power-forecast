# -*- coding: utf-8 -*-
"""
This module contains functions for visualizing the solar power data.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

def plot_first_week_power(df, farm_number):
    """
    Plots the power generation for the first week of a given solar farm.

    Args:
        df (pd.DataFrame): The dataframe containing the power data.
        farm_number (int): The number of the solar farm.
    """
    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.plot(df.index[:96*7], df["power"][:96*7])
    ax.set_xlabel('Time')
    ax.set_ylabel('Power')
    ax.set_title(f'First Week Power Generation for Farm {farm_number}')
    plt.show()

def plot_sin_cos_waves(df):
    """
    Plots the generated sin and cos waves for day and year cycles.

    Args:
        df (pd.DataFrame): The dataframe with sin/cos columns.
    """
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(df.index[:96*2], df["Day sin"][:96*2])
    plt.title('Day sin & Day cos')
    ax.plot(df.index[:96*2], df["Day cos"][:96*2])
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(df.index[:96*2*365], df["Year sin"][:96*2*365])
    plt.title('Year sin & Year cos')
    ax.plot(df.index[:96*2*365], df["Year cos"][:96*2*365])
    plt.show()

def plot_box_plots(df):
    """
    Generates box plots for each column in the dataframe.

    Args:
        df (pd.DataFrame): The dataframe to plot.
    """
    for col in df.columns:
        sns.boxplot(df[col])
        sns.set(rc={'figure.figsize':(3,5)})
        name = 'Box Plot of ' +  str(col)
        plt.title(name)
        plt.show()

def plot_line_chart(df_i, source_df, title):
    """
    Plots a line chart of all variables in a dataframe.

    Args:
        df_i (pd.DataFrame): The dataframe to plot.
        source_df (pd.DataFrame): The source dataframe for normalization (not used here).
        title (str): The title of the plot.
    """
    columns = df_i.columns
    fig, ax = plt.subplots(figsize=(5,3))
    for col in columns:
        ax.plot(df_i.index, df_i[col], label=col)
    
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.legend()
    plt.show()

def plot_standardized_line_chart(df_i, source_df, title):
    """
    Plots a standardized line chart of all variables.

    Args:
        df_i (pd.DataFrame): The dataframe to plot.
        source_df (pd.DataFrame): The source dataframe for normalization.
        title (str): The title of the plot.
    """
    columns = df_i.columns
    for col in columns:
        df_i[col] = df_i[col] / source_df[col].max()
    fig, ax = plt.subplots(figsize=(10,3))
    for col in columns:
        ax.plot(df_i.index, df_i[col], label=col)
    ax.set_xlabel('Time')
    ax.set_title(title)
    ax.legend(loc=3)
    plt.show()

def plot_density_hist(df, bins=10, hist=False):
    """
    Plots a density histogram for each column.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        bins (int, optional): Number of bins. Defaults to 10.
        hist (bool, optional): Whether to show histogram. Defaults to False.
    """
    for col in df.columns:
        sns.set_style("whitegrid")
        sns.distplot(df[col], bins=bins, rug=True, hist=hist)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Number of data')
        plt.show()

def plot_feature_correlations(datasets):
    """
    Plots feature correlation heatmaps for a list of datasets.

    Args:
        datasets (list): A list of pandas DataFrames.
    """
    fig, axs = plt.subplots(4, 2, figsize=(8, 16))
    for i, data in enumerate(datasets):
        row = i // 2
        col = i % 2
        sns.heatmap(data.corr(), cmap='Reds', annot=False, ax=axs[row, col])
        axs[row, col].set_title(f'Feature correlation {i+1}')
    plt.tight_layout()
    plt.show()

def plot_predictions(results, title, limit=200):
    """
    Plots model predictions against actual values.

    Args:
        results (pd.DataFrame): DataFrame with 'Predictions' and 'Actuals' columns.
        title (str): The title for the plot.
        limit (int, optional): The number of data points to plot. Defaults to 200.
    """
    plt.figure(figsize=(15, 7))
    plt.plot(results['Predictions'][:limit], label="LSTM Prediction")
    plt.plot(results['Actuals'][:limit], label='Actual Data')
    plt.title(title)
    plt.legend()
    plt.show()
