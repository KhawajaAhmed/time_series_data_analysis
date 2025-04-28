"""
Utility functions for data visualization in time series analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Optional, Dict, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def set_plotting_style():
    """Set the default style for matplotlib plots."""
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12


def plot_distribution(df: pd.DataFrame, column: str, 
                      title: Optional[str] = None, 
                      figsize: Tuple[int, int] = (12, 6),
                      kde: bool = True) -> None:
    """
    Plot the distribution of a column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column to plot
    title : Optional[str]
        Plot title
    figsize : Tuple[int, int]
        Figure size
    kde : bool
        Whether to include KDE plot
    """
    plt.figure(figsize=figsize)
    
    if pd.api.types.is_numeric_dtype(df[column]):
        sns.histplot(df[column], kde=kde)
        plt.axvline(df[column].mean(), color='r', linestyle='--', label=f'Mean: {df[column].mean():.2f}')
        plt.axvline(df[column].median(), color='g', linestyle='-.', label=f'Median: {df[column].median():.2f}')
        plt.legend()
    else:
        sns.countplot(y=df[column].value_counts().index, 
                      data=df, 
                      order=df[column].value_counts().index)
    
    plt.title(title or f'Distribution of {column}')
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, 
                             columns: Optional[List[str]] = None,
                             figsize: Tuple[int, int] = (12, 10),
                             cmap: str = 'coolwarm',
                             mask_upper: bool = True) -> None:
    """
    Plot correlation heatmap for selected columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : Optional[List[str]]
        Columns to include in the heatmap. If None, use all numeric columns.
    figsize : Tuple[int, int]
        Figure size
    cmap : str
        Colormap for the heatmap
    mask_upper : bool
        Whether to mask the upper triangle of the heatmap
    """
    if columns is None:
        # Select only numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns
        columns = numeric_columns
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create mask for upper triangle
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, mask=mask, 
                vmin=-1, vmax=1, center=0, linewidths=0.5,
                fmt='.2f', cbar_kws={"shrink": 0.8})
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()


def plot_boxplot(df: pd.DataFrame, 
                 columns: List[str],
                 figsize: Tuple[int, int] = (12, 6),
                 vert: bool = True) -> None:
    """
    Create boxplots for multiple columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to plot
    figsize : Tuple[int, int]
        Figure size
    vert : bool
        Whether to plot vertical (True) or horizontal (False) boxplots
    """
    plt.figure(figsize=figsize)
    
    if vert:
        sns.boxplot(data=df[columns])
        plt.xticks(rotation=45)
    else:
        sns.boxplot(data=df[columns], orient='h')
    
    plt.title('Boxplot of Selected Features')
    plt.tight_layout()
    plt.show()


def plot_multiple_time_series(df: pd.DataFrame, 
                              columns: List[str],
                              title: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 8),
                              date_column: Optional[str] = None) -> None:
    """
    Plot multiple time series on the same graph.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to plot
    title : Optional[str]
        Plot title
    figsize : Tuple[int, int]
        Figure size
    date_column : Optional[str]
        Date column to use as x-axis. If None, assumes the index is a datetime.
    """
    plt.figure(figsize=figsize)
    
    if date_column is not None and date_column in df.columns:
        for column in columns:
            plt.plot(df[date_column], df[column], label=column)
        plt.xlabel(date_column)
    else:
        for column in columns:
            plt.plot(df.index, df[column], label=column)
        plt.xlabel('Date')
    
    plt.legend()
    plt.title(title or 'Multiple Time Series Plot')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_acf_pacf(series: pd.Series, 
                  lags: int = 40,
                  figsize: Tuple[int, int] = (12, 8),
                  title: Optional[str] = None) -> None:
    """
    Plot ACF and PACF for a time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to analyze
    lags : int
        Number of lags to include
    figsize : Tuple[int, int]
        Figure size
    title : Optional[str]
        Plot title
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')
    
    plot_pacf(series, lags=lags, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    
    plt.suptitle(title or f'ACF and PACF for {series.name}')
    plt.tight_layout()
    plt.show()


def plot_interactive_time_series(df: pd.DataFrame, 
                                 columns: List[str],
                                 date_column: Optional[str] = None,
                                 title: str = 'Interactive Time Series Plot') -> go.Figure:
    """
    Create an interactive time series plot using Plotly.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to plot
    date_column : Optional[str]
        Date column to use as x-axis. If None, assumes the index is a datetime.
    title : str
        Plot title
        
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    x_values = df.index if date_column is None else df[date_column]
    
    for column in columns:
        fig.add_trace(go.Scatter(
            x=x_values,
            y=df[column],
            mode='lines',
            name=column
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Value',
        legend_title='Variables',
        hovermode='x unified'
    )
    
    return fig


def plot_seasonal_subseries(df: pd.DataFrame, 
                            column: str,
                            period: int,
                            figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Create a seasonal subseries plot.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    column : str
        Column to plot
    period : int
        Period for seasonal decomposition (e.g., 12 for monthly data with yearly seasonality)
    figsize : Tuple[int, int]
        Figure size
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    series = df[column]
    
    # Create cycle plot
    fig, axes = plt.subplots(period, 1, figsize=figsize, sharex=True)
    
    for i in range(period):
        season_data = series.iloc[i::period]
        axes[i].plot(season_data.index, season_data.values)
        axes[i].set_ylabel(f'Period {i+1}')
        
        # Add horizontal line for the mean
        axes[i].axhline(season_data.mean(), color='r', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'Seasonal Subseries Plot for {column} (Period={period})')
    plt.tight_layout()
    plt.show()


def plot_lag_scatter(series: pd.Series, 
                     lag: int = 1,
                     figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Create a lag scatter plot.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to analyze
    lag : int
        Lag to use for the scatter plot
    figsize : Tuple[int, int]
        Figure size
    """
    plt.figure(figsize=figsize)
    
    plt.scatter(series.iloc[:-lag], series.iloc[lag:])
    plt.xlabel(f'{series.name} (t)')
    plt.ylabel(f'{series.name} (t+{lag})')
    plt.title(f'Lag {lag} Scatter Plot for {series.name}')
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        series.iloc[:-lag].dropna(), series.iloc[lag:].dropna())
    
    x = np.linspace(series.min(), series.max(), 100)
    y = slope * x + intercept
    plt.plot(x, y, 'r--', label=f'R² = {r_value**2:.3f}')
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
