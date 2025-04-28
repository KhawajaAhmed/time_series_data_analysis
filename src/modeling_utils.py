"""
Utility functions for time series modeling and forecasting.
"""
import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


def check_stationarity(series: pd.Series, 
                       window: int = 12, 
                       figsize: Tuple[int, int] = (12, 8),
                       verbose: bool = True) -> Dict[str, Any]:
    """
    Check stationarity of a time series using rolling statistics and ADF test.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to check
    window : int
        Window size for rolling statistics
    figsize : Tuple[int, int]
        Figure size for the plot
    verbose : bool
        Whether to print the results
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with test results
    """
    # Calculate rolling statistics
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    
    # Perform ADF test
    adf_result = adfuller(series.dropna())
    
    adf_output = {
        'Test Statistic': adf_result[0],
        'p-value': adf_result[1],
        'Critical Values': adf_result[4]
    }
    
    # Plot rolling statistics
    plt.figure(figsize=figsize)
    plt.subplot(211)
    plt.plot(series, label='Original')
    plt.plot(rolling_mean, label=f'Rolling Mean (window={window})')
    plt.plot(rolling_std, label=f'Rolling Std (window={window})')
    plt.legend(loc='best')
    plt.title(f'Rolling Statistics for {series.name}')
    
    plt.subplot(212)
    plt.plot(series.diff().dropna(), label='Differenced Series')
    plt.legend(loc='best')
    plt.title('First Difference')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    if verbose:
        print('Augmented Dickey-Fuller Test Results:')
        print(f'Test Statistic: {adf_result[0]:.4f}')
        print(f'p-value: {adf_result[1]:.4f}')
        print('Critical Values:')
        for key, value in adf_result[4].items():
            print(f'\t{key}: {value:.4f}')
        
        # Interpretation
        if adf_result[1] <= 0.05:
            print("\nResult: The series is stationary (reject H0)")
        else:
            print("\nResult: The series is non-stationary (fail to reject H0)")
    
    return {
        'adf_result': adf_output,
        'is_stationary': adf_result[1] <= 0.05,
        'rolling_mean': rolling_mean,
        'rolling_std': rolling_std
    }


def make_stationary(series: pd.Series, 
                    method: str = 'diff',
                    order: int = 1,
                    log_transform: bool = False) -> pd.Series:
    """
    Transform a time series to make it stationary.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to transform
    method : str
        Method to use ('diff', 'pct_change', 'detrend', or 'decompose')
    order : int
        Order of differencing (for 'diff' method)
    log_transform : bool
        Whether to apply log transformation before other transformations
        
    Returns:
    --------
    pd.Series
        Transformed series
    """
    result = series.copy()
    
    # Apply log transformation if requested
    if log_transform and (result > 0).all():
        result = np.log(result)
    
    # Apply the selected transformation method
    if method == 'diff':
        for _ in range(order):
            result = result.diff().dropna()
    elif method == 'pct_change':
        result = result.pct_change().dropna()
    elif method == 'detrend':
        from scipy import signal
        result = pd.Series(
            signal.detrend(result.values),
            index=result.index,
            name=result.name
        )
    elif method == 'decompose':
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(result, model='additive')
        result = decomposition.resid.dropna()
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return result


def create_features(df: pd.DataFrame, 
                    date_column: Optional[str] = None,
                    target_column: Optional[str] = None,
                    lag_features: Optional[List[int]] = None,
                    rolling_features: Optional[List[int]] = None,
                    rolling_stats: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create time series features from datetime index.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_column : Optional[str]
        Date column to use. If None, assumes the index is a datetime.
    target_column : Optional[str]
        Target column to create lag and rolling features for
    lag_features : Optional[List[int]]
        List of lag values to create features for
    rolling_features : Optional[List[int]]
        List of window sizes for rolling features
    rolling_stats : Optional[List[str]]
        List of statistics to compute for rolling windows ('mean', 'std', 'min', 'max')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional features
    """
    df_features = df.copy()
    
    # Use index if date_column is None
    if date_column is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex when date_column is None")
        date_series = df.index
    else:
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df_features[date_column] = pd.to_datetime(df_features[date_column])
        date_series = df_features[date_column]
    
    # Extract datetime features
    if date_column is None:
        df_features['year'] = date_series.year
        df_features['month'] = date_series.month
        df_features['day'] = date_series.day
        df_features['dayofweek'] = date_series.dayofweek
        df_features['quarter'] = date_series.quarter
        df_features['dayofyear'] = date_series.dayofyear
        df_features['weekofyear'] = date_series.isocalendar().week
        df_features['is_month_start'] = date_series.is_month_start.astype(int)
        df_features['is_month_end'] = date_series.is_month_end.astype(int)
    else:
        df_features['year'] = date_series.dt.year
        df_features['month'] = date_series.dt.month
        df_features['day'] = date_series.dt.day
        df_features['dayofweek'] = date_series.dt.dayofweek
        df_features['quarter'] = date_series.dt.quarter
        df_features['dayofyear'] = date_series.dt.dayofyear
        df_features['weekofyear'] = date_series.dt.isocalendar().week
        df_features['is_month_start'] = date_series.dt.is_month_start.astype(int)
        df_features['is_month_end'] = date_series.dt.is_month_end.astype(int)
    
    # Create cyclical features
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    df_features['dayofweek_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
    df_features['dayofweek_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)
    
    # Create lag features
    if target_column is not None and lag_features is not None:
        for lag in lag_features:
            df_features[f'{target_column}_lag_{lag}'] = df_features[target_column].shift(lag)
    
    # Create rolling features
    if target_column is not None and rolling_features is not None:
        if rolling_stats is None:
            rolling_stats = ['mean', 'std']
        
        for window in rolling_features:
            for stat in rolling_stats:
                if stat == 'mean':
                    df_features[f'{target_column}_rolling_{window}_mean'] = df_features[target_column].rolling(window=window).mean()
                elif stat == 'std':
                    df_features[f'{target_column}_rolling_{window}_std'] = df_features[target_column].rolling(window=window).std()
                elif stat == 'min':
                    df_features[f'{target_column}_rolling_{window}_min'] = df_features[target_column].rolling(window=window).min()
                elif stat == 'max':
                    df_features[f'{target_column}_rolling_{window}_max'] = df_features[target_column].rolling(window=window).max()
                elif stat == 'median':
                    df_features[f'{target_column}_rolling_{window}_median'] = df_features[target_column].rolling(window=window).median()
    
    return df_features


def train_test_split_time(df: pd.DataFrame, 
                          test_size: float = 0.2,
                          target_column: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a time series dataset into train and test sets based on time.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with datetime index
    test_size : float
        Proportion of the data to include in the test split
    target_column : Optional[str]
        Target column to return as separate series
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame] or Tuple[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series]]
        Train and test splits
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    
    # Calculate split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    
    if target_column is not None:
        X_train = train.drop(columns=[target_column])
        y_train = train[target_column]
        X_test = test.drop(columns=[target_column])
        y_test = test[target_column]
        return (X_train, y_train), (X_test, y_test)
    else:
        return train, test


def evaluate_forecast(actual: pd.Series, 
                      predicted: pd.Series,
                      metrics: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Evaluate forecast performance using multiple metrics.
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values
    predicted : pd.Series
        Predicted values
    metrics : Optional[List[str]]
        List of metrics to compute
        
    Returns:
    --------
    Dict[str, float]
        Dictionary with metric names and values
    """
    if metrics is None:
        metrics = ['mse', 'rmse', 'mae', 'mape', 'r2']
    
    results = {}
    
    for metric in metrics:
        if metric == 'mse':
            results['MSE'] = mean_squared_error(actual, predicted)
        elif metric == 'rmse':
            results['RMSE'] = np.sqrt(mean_squared_error(actual, predicted))
        elif metric == 'mae':
            results['MAE'] = mean_absolute_error(actual, predicted)
        elif metric == 'mape':
            # Handle zero values in actual
            mask = actual != 0
            if mask.all():
                results['MAPE'] = np.mean(np.abs((actual - predicted) / actual)) * 100
            else:
                results['MAPE'] = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        elif metric == 'r2':
            results['R²'] = r2_score(actual, predicted)
        elif metric == 'smape':
            # Symmetric Mean Absolute Percentage Error
            results['SMAPE'] = 200 * np.mean(np.abs(predicted - actual) / (np.abs(predicted) + np.abs(actual)))
    
    return results


def plot_forecast(actual: pd.Series, 
                  predicted: pd.Series,
                  train_actual: Optional[pd.Series] = None,
                  figsize: Tuple[int, int] = (12, 6),
                  title: str = 'Forecast vs Actual',
                  include_metrics: bool = True) -> None:
    """
    Plot forecast vs actual values.
    
    Parameters:
    -----------
    actual : pd.Series
        Actual values for test period
    predicted : pd.Series
        Predicted values for test period
    train_actual : Optional[pd.Series]
        Actual values for training period
    figsize : Tuple[int, int]
        Figure size
    title : str
        Plot title
    include_metrics : bool
        Whether to include metrics in the plot
    """
    plt.figure(figsize=figsize)
    
    if train_actual is not None:
        plt.plot(train_actual.index, train_actual, label='Train Actual', color='blue', alpha=0.5)
    
    plt.plot(actual.index, actual, label='Test Actual', color='blue')
    plt.plot(predicted.index, predicted, label='Forecast', color='red', linestyle='--')
    
    # Add confidence intervals if available
    if hasattr(predicted, 'conf_int'):
        conf_int = predicted.conf_int()
        plt.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='red', alpha=0.1)
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Value')
    
    if include_metrics:
        metrics = evaluate_forecast(actual, predicted)
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                     va='top')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def fit_arima(series: pd.Series, 
              order: Tuple[int, int, int] = (1, 1, 1),
              seasonal_order: Optional[Tuple[int, int, int, int]] = None,
              return_residuals: bool = False) -> Union[sm.tsa.ARIMA, Tuple[sm.tsa.ARIMA, pd.Series]]:
    """
    Fit an ARIMA model to a time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to model
    order : Tuple[int, int, int]
        ARIMA order (p, d, q)
    seasonal_order : Optional[Tuple[int, int, int, int]]
        Seasonal order (P, D, Q, s)
    return_residuals : bool
        Whether to return residuals
        
    Returns:
    --------
    Union[sm.tsa.ARIMA, Tuple[sm.tsa.ARIMA, pd.Series]]
        Fitted model and optionally residuals
    """
    # Fit ARIMA model
    if seasonal_order is not None:
        model = sm.tsa.SARIMAX(series, order=order, seasonal_order=seasonal_order)
    else:
        model = sm.tsa.ARIMA(series, order=order)
    
    fitted_model = model.fit()
    
    if return_residuals:
        residuals = fitted_model.resid
        return fitted_model, residuals
    else:
        return fitted_model


def auto_arima(series: pd.Series, 
               max_p: int = 5, 
               max_d: int = 2, 
               max_q: int = 5,
               seasonal: bool = True,
               m: int = 12,
               information_criterion: str = 'aic',
               verbose: bool = True) -> Any:
    """
    Automatically find the best ARIMA model using pmdarima.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to model
    max_p : int
        Maximum value of p
    max_d : int
        Maximum value of d
    max_q : int
        Maximum value of q
    seasonal : bool
        Whether to fit a seasonal ARIMA model
    m : int
        The seasonal period
    information_criterion : str
        Information criterion to use ('aic', 'bic', 'hqic', 'oob')
    verbose : bool
        Whether to print progress
        
    Returns:
    --------
    Any
        Fitted pmdarima model
    """
    import pmdarima as pm
    
    model = pm.auto_arima(
        series,
        start_p=0, max_p=max_p,
        start_q=0, max_q=max_q,
        d=None, max_d=max_d,
        seasonal=seasonal, m=m,
        information_criterion=information_criterion,
        trace=verbose,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    if verbose:
        print(model.summary())
    
    return model


def fit_prophet(series: pd.Series, 
                forecast_periods: int = 30,
                yearly_seasonality: Union[bool, int] = 'auto',
                weekly_seasonality: Union[bool, int] = 'auto',
                daily_seasonality: Union[bool, int] = 'auto',
                include_history: bool = True) -> Tuple[Any, pd.DataFrame]:
    """
    Fit a Prophet model to a time series and make forecast.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to model
    forecast_periods : int
        Number of periods to forecast
    yearly_seasonality : Union[bool, int]
        Whether to include yearly seasonality
    weekly_seasonality : Union[bool, int]
        Whether to include weekly seasonality
    daily_seasonality : Union[bool, int]
        Whether to include daily seasonality
    include_history : bool
        Whether to include historical data in the forecast
        
    Returns:
    --------
    Tuple[Any, pd.DataFrame]
        Fitted Prophet model and forecast dataframe
    """
    from prophet import Prophet
    
    # Prepare data for Prophet
    df = pd.DataFrame({'ds': series.index, 'y': series.values})
    
    # Create and fit model
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality
    )
    
    model.fit(df)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=forecast_periods, freq=series.index.inferred_freq)
    
    # Forecast
    forecast = model.predict(future)
    
    return model, forecast
