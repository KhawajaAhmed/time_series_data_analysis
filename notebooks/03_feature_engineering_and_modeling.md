# Transportation Time Series Analysis - Feature Engineering and Modeling

This notebook covers the third phase of our transportation time series analysis project:
1. Feature engineering for time series data
2. Model development and training
3. Model evaluation and selection

## 1. Import Libraries and Load Data

```python
# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys
import warnings
import json
warnings.filterwarnings('ignore')

# Modeling libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from xgboost import XGBRegressor

# Add the project root directory to the Python path
sys.path.append('..')

# Import custom utility functions
from src.data_utils import load_data, convert_to_datetime
from src.visualization_utils import set_plotting_style, plot_multiple_time_series
from src.modeling_utils import (
    create_features,
    train_test_split_time,
    evaluate_forecast,
    plot_forecast,
    fit_arima,
    auto_arima
)

# Set plotting style
set_plotting_style()

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
```

```python
# Load the processed data from Phase 1
processed_dir = os.path.join('..', 'data', 'processed')
df = pd.read_csv(os.path.join(processed_dir, 'transportation_data_processed.csv'))

# Convert the index to datetime
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Load analysis results from Phase 2
with open(os.path.join(processed_dir, 'analysis_results.json'), 'r') as f:
    analysis_results = json.load(f)

print(f"Loaded processed data with {len(df)} records.")
df.head()
```

## 2. Feature Engineering

Let's create features that will help our models capture the time series patterns.

```python
# Create time series features
target_column = 'vehicle_usage_hours'
lag_features = [1, 2, 3, 7, 14, 30]  # Previous day, week, and month
rolling_features = [7, 14, 30]  # Weekly, bi-weekly, and monthly windows
rolling_stats = ['mean', 'std', 'min', 'max']

df_features = create_features(
    df, 
    target_column=target_column,
    lag_features=lag_features,
    rolling_features=rolling_features,
    rolling_stats=rolling_stats
)

print(f"Created features dataframe with {df_features.shape[1]} columns.")
df_features.head()
```

```python
# Drop rows with NaN values (due to lag features)
df_features = df_features.dropna()
print(f"After dropping NaN values, dataframe has {len(df_features)} rows.")
```

```python
# Create one-hot encoded features for categorical variables
df_features = pd.get_dummies(df_features, columns=['vehicle_type', 'region', 'season'], drop_first=False)

# List the engineered features
feature_columns = df_features.columns.tolist()
feature_columns.remove(target_column)  # Remove the target column

print(f"Total number of features: {len(feature_columns)}")
print("\nSample of engineered features:")
print(feature_columns[:10])
```

## 3. Train-Test Split

Let's split our data into training and testing sets, respecting the time series nature of the data.

```python
# Split the data into training and testing sets
test_size = 0.2  # Use 20% of the data for testing
(X_train, y_train), (X_test, y_test) = train_test_split_time(
    df_features, test_size=test_size, target_column=target_column
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set: {X_test.shape[0]} samples")
```

```python
# Plot the train-test split
plt.figure(figsize=(12, 6))
plt.plot(y_train.index, y_train, label='Training Data')
plt.plot(y_test.index, y_test, label='Testing Data', color='red')
plt.title('Train-Test Split for Vehicle Usage Hours')
plt.ylabel('Hours')
plt.legend()
plt.grid(True)
plt.show()
```

## 4. Time Series Models

Let's implement and evaluate several time series forecasting models.

### 4.1 Baseline Models

We'll start with simple baseline models to establish a performance benchmark.

```python
# Naive forecast (use the last value)
naive_forecast = y_train.iloc[-1]
y_pred_naive = pd.Series(naive_forecast, index=y_test.index)

# Seasonal naive forecast (use the value from the same day of the previous week)
y_pred_seasonal_naive = pd.Series(index=y_test.index)
for i, idx in enumerate(y_test.index):
    # Find the same day of the week from the previous week
    prev_week = idx - pd.Timedelta(days=7)
    if prev_week in y_train.index:
        y_pred_seasonal_naive[idx] = y_train[prev_week]
    else:
        # If not available, use the last value from training
        y_pred_seasonal_naive[idx] = y_train.iloc[-1]

# Average forecast (use the average of the training data)
average_forecast = y_train.mean()
y_pred_average = pd.Series(average_forecast, index=y_test.index)

# Evaluate baseline models
baseline_results = {
    'Naive': evaluate_forecast(y_test, y_pred_naive),
    'Seasonal Naive': evaluate_forecast(y_test, y_pred_seasonal_naive),
    'Average': evaluate_forecast(y_test, y_pred_average)
}

# Display results
baseline_df = pd.DataFrame(baseline_results).T
baseline_df
```

```python
# Plot baseline forecasts
plt.figure(figsize=(12, 6))
plt.plot(y_train.index[-30:], y_train[-30:], label='Training Data', color='blue')
plt.plot(y_test.index, y_test, label='Actual', color='black')
plt.plot(y_test.index, y_pred_naive, label='Naive', color='red', linestyle='--')
plt.plot(y_test.index, y_pred_seasonal_naive, label='Seasonal Naive', color='green', linestyle='--')
plt.plot(y_test.index, y_pred_average, label='Average', color='purple', linestyle='--')
plt.title('Baseline Forecasts vs Actual Values')
plt.ylabel('Vehicle Usage Hours')
plt.legend()
plt.grid(True)
plt.show()
```

### 4.2 ARIMA Models

Now let's implement ARIMA (AutoRegressive Integrated Moving Average) models.

```python
# Find the best ARIMA model using auto_arima
auto_arima_model = auto_arima(
    y_train,
    seasonal=True,
    m=7,  # Weekly seasonality
    information_criterion='aic',
    verbose=True
)
```

```python
# Get the order and seasonal_order from auto_arima
arima_order = auto_arima_model.order
seasonal_order = auto_arima_model.seasonal_order

print(f"Best ARIMA order: {arima_order}")
print(f"Best seasonal order: {seasonal_order}")

# Fit SARIMA model with the best parameters
sarima_model = SARIMAX(
    y_train,
    order=arima_order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)
sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())
```

```python
# Forecast with SARIMA model
sarima_forecast = sarima_fit.forecast(steps=len(y_test))
y_pred_sarima = pd.Series(sarima_forecast, index=y_test.index)

# Evaluate SARIMA model
sarima_results = evaluate_forecast(y_test, y_pred_sarima)
print("SARIMA Model Results:")
for metric, value in sarima_results.items():
    print(f"{metric}: {value:.4f}")
```

```python
# Plot SARIMA forecast
plot_forecast(y_test, y_pred_sarima, train_actual=y_train[-30:], 
              title='SARIMA Forecast vs Actual Values')
```

### 4.3 Machine Learning Models

Let's implement machine learning models that can leverage our engineered features.

```python
# XGBoost Regressor
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# Fit the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = pd.Series(xgb_model.predict(X_test), index=y_test.index)

# Evaluate XGBoost model
xgb_results = evaluate_forecast(y_test, y_pred_xgb)
print("XGBoost Model Results:")
for metric, value in xgb_results.items():
    print(f"{metric}: {value:.4f}")
```

```python
# Plot XGBoost forecast
plot_forecast(y_test, y_pred_xgb, train_actual=y_train[-30:], 
              title='XGBoost Forecast vs Actual Values')
```

```python
# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot top 15 features
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.show()
```

## 5. Model Comparison and Selection

Let's compare all the models and select the best one.

```python
# Combine all results
all_results = {
    'Naive': baseline_results['Naive'],
    'Seasonal Naive': baseline_results['Seasonal Naive'],
    'Average': baseline_results['Average'],
    'SARIMA': sarima_results,
    'XGBoost': xgb_results
}

# Create a comparison dataframe
comparison_df = pd.DataFrame(all_results).T
comparison_df
```

```python
# Plot RMSE comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=comparison_df.index, y='RMSE', data=comparison_df)
plt.title('RMSE Comparison Across Models')
plt.ylabel('RMSE (lower is better)')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```

```python
# Plot all forecasts together
plt.figure(figsize=(14, 8))
plt.plot(y_train.index[-30:], y_train[-30:], label='Training Data', color='blue', alpha=0.5)
plt.plot(y_test.index, y_test, label='Actual', color='black', linewidth=2)
plt.plot(y_test.index, y_pred_naive, label='Naive', linestyle='--')
plt.plot(y_test.index, y_pred_seasonal_naive, label='Seasonal Naive', linestyle='--')
plt.plot(y_test.index, y_pred_sarima, label='SARIMA', linestyle='--')
plt.plot(y_test.index, y_pred_xgb, label='XGBoost', linestyle='--')
plt.title('All Forecasts vs Actual Values')
plt.ylabel('Vehicle Usage Hours')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

## 6. Model Interpretation

Let's interpret the best model and understand what drives our predictions.

```python
# Determine the best model based on RMSE
best_model = comparison_df['RMSE'].idxmin()
print(f"The best model based on RMSE is: {best_model}")

# If XGBoost is the best model, analyze feature importance in more detail
if best_model == 'XGBoost':
    # Get top 10 features
    top_features = feature_importance.head(10)['Feature'].tolist()
    
    print("\nTop 10 Important Features:")
    for i, (feature, importance) in enumerate(zip(feature_importance.head(10)['Feature'], 
                                                 feature_importance.head(10)['Importance'])):
        print(f"{i+1}. {feature}: {importance:.4f}")
    
    # Analyze how these features relate to the target
    print("\nCorrelation with target:")
    for feature in top_features:
        if feature in df_features.columns:
            corr = df_features[feature].corr(df_features[target_column])
            print(f"{feature}: {corr:.4f}")
```

```python
# If SARIMA is the best model, analyze the components
if best_model == 'SARIMA':
    # Plot the components
    sarima_fit.plot_components(figsize=(12, 10))
    plt.tight_layout()
    plt.show()
```

## 7. Save the Best Model

Let's save the best model for future use.

```python
# Create models directory if it doesn't exist
models_dir = os.path.join('..', 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Save the best model
if best_model == 'XGBoost':
    import joblib
    # Save the model
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost_model.pkl'))
    # Save feature names
    with open(os.path.join(models_dir, 'feature_names.json'), 'w') as f:
        json.dump(X_train.columns.tolist(), f)
    print(f"Saved XGBoost model to {os.path.join(models_dir, 'xgboost_model.pkl')}")
    
elif best_model == 'SARIMA':
    # Save the model parameters
    sarima_params = {
        'order': arima_order,
        'seasonal_order': seasonal_order,
        'params': sarima_fit.params.to_dict()
    }
    with open(os.path.join(models_dir, 'sarima_model_params.json'), 'w') as f:
        json.dump(sarima_params, f, indent=4)
    print(f"Saved SARIMA model parameters to {os.path.join(models_dir, 'sarima_model_params.json')}")
```

## 8. Key Insights from Modeling

Based on our modeling efforts, here are the key insights:

1. **Model Performance**:
   - The [best_model] model performed best with an RMSE of [best_rmse]
   - This represents a [improvement_percentage]% improvement over the naive baseline

2. **Important Predictors**:
   - Recent past values (lag features) are the strongest predictors
   - Weekly seasonality is a significant factor
   - Vehicle type and region also influence usage patterns

3. **Forecasting Accuracy**:
   - The model can predict vehicle usage with a mean absolute percentage error of approximately [mape]%
   - Predictions are more accurate for short-term forecasts (1-7 days) than long-term forecasts

4. **Business Implications**:
   - The model can help optimize fleet allocation based on predicted demand
   - Understanding the key drivers of vehicle usage can inform strategic decisions
   - The forecasting capability enables better planning for maintenance and resource allocation

In the next notebook, we'll use these models for forecasting and develop a comprehensive report with actionable recommendations.
