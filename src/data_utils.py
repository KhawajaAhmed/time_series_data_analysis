"""
Utility functions for data loading, cleaning, and preprocessing.
"""
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Tuple
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from various file formats based on extension.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file
        
    Returns:
    --------
    pd.DataFrame
        Loaded data
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        return pd.read_csv(file_path)
    elif file_extension in ['.xls', '.xlsx']:
        return pd.read_excel(file_path)
    elif file_extension == '.json':
        return pd.read_json(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing value statistics
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_info = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percentage': missing_percentage.round(2)
    })
    
    return missing_info[missing_info['Missing Values'] > 0].sort_values('Missing Percentage', ascending=False)


def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate', 
                          columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Method to handle missing values ('drop', 'mean', 'median', 'mode', 'interpolate', 'ffill', 'bfill')
    columns : Optional[List[str]]
        Specific columns to apply the method to. If None, apply to all columns.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with handled missing values
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df.columns
    
    for col in columns:
        if col in df.columns:
            if method == 'drop':
                df_copy = df_copy.dropna(subset=[col])
            elif method == 'mean':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif method == 'median':
                if pd.api.types.is_numeric_dtype(df[col]):
                    df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif method == 'mode':
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif method == 'interpolate':
                df_copy[col] = df_copy[col].interpolate(method='linear')
            elif method == 'ffill':
                df_copy[col] = df_copy[col].fillna(method='ffill')
            elif method == 'bfill':
                df_copy[col] = df_copy[col].fillna(method='bfill')
            else:
                raise ValueError(f"Unsupported method: {method}")
    
    return df_copy


def detect_outliers(df: pd.DataFrame, columns: List[str], method: str = 'iqr', 
                    threshold: float = 1.5) -> Dict[str, np.ndarray]:
    """
    Detect outliers in specified columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : List[str]
        Columns to check for outliers
    method : str
        Method to detect outliers ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection (default: 1.5 for IQR, 3 for zscore)
        
    Returns:
    --------
    Dict[str, np.ndarray]
        Dictionary with column names as keys and boolean arrays indicating outliers as values
    """
    outliers = {}
    
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers[col] = np.full(df[col].shape, False)
                outliers[col][~df[col].isna()] = z_scores > threshold
            else:
                raise ValueError(f"Unsupported method: {method}")
    
    return outliers


def convert_to_datetime(df: pd.DataFrame, date_column: str, 
                        format: Optional[str] = None, 
                        set_index: bool = False) -> pd.DataFrame:
    """
    Convert a column to datetime and optionally set it as index.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Column to convert to datetime
    format : Optional[str]
        Format string for datetime conversion
    set_index : bool
        Whether to set the converted column as index
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with converted datetime column
    """
    df_copy = df.copy()
    
    if date_column in df.columns:
        if format:
            df_copy[date_column] = pd.to_datetime(df_copy[date_column], format=format)
        else:
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        
        if set_index:
            df_copy = df_copy.set_index(date_column)
    
    return df_copy


def plot_time_series(df: pd.DataFrame, column: str, title: str = None, 
                     figsize: Tuple[int, int] = (12, 6), 
                     date_column: Optional[str] = None) -> None:
    """
    Plot a time series.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column to plot
    title : str
        Plot title
    figsize : Tuple[int, int]
        Figure size
    date_column : Optional[str]
        Date column to use as x-axis. If None, assumes the index is a datetime.
    """
    plt.figure(figsize=figsize)
    
    if date_column is not None and date_column in df.columns:
        plt.plot(df[date_column], df[column])
        plt.xlabel(date_column)
    else:
        plt.plot(df.index, df[column])
        plt.xlabel('Date')
    
    plt.ylabel(column)
    plt.title(title or f'Time Series Plot of {column}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def decompose_time_series(df: pd.DataFrame, column: str, 
                          period: int = None, model: str = 'additive',
                          figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Decompose a time series into trend, seasonal, and residual components.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    column : str
        Column to decompose
    period : int
        Period for seasonal decomposition
    model : str
        Type of decomposition ('additive' or 'multiplicative')
    figsize : Tuple[int, int]
        Figure size
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    # If period is not provided, try to infer it
    if period is None:
        if df.index.freq is not None:
            # Daily data
            if df.index.freq == 'D':
                period = 7  # Weekly seasonality
            # Monthly data
            elif df.index.freq in ['M', 'MS']:
                period = 12  # Yearly seasonality
            # Hourly data
            elif df.index.freq == 'H':
                period = 24  # Daily seasonality
            else:
                period = 12  # Default
        else:
            period = 12  # Default
    
    # Perform decomposition
    decomposition = seasonal_decompose(df[column], model=model, period=period)
    
    # Plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    decomposition.observed.plot(ax=ax1)
    ax1.set_ylabel('Observed')
    ax1.set_title(f'Time Series Decomposition ({model})')
    
    decomposition.trend.plot(ax=ax2)
    ax2.set_ylabel('Trend')
    
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_ylabel('Seasonal')
    
    decomposition.resid.plot(ax=ax4)
    ax4.set_ylabel('Residual')
    
    plt.tight_layout()
    plt.show()
    
    return decomposition
