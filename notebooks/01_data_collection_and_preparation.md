# Transportation Time Series Analysis - Data Collection and Preparation

This notebook covers the first phase of our transportation time series analysis project:
1. Project initiation and problem definition
2. Data collection and loading
3. Initial data exploration
4. Data cleaning and preprocessing

## 1. Project Initiation

### Problem Statement
We are analyzing transportation data to optimize fleet utilization and reduce idle time for a transportation company. The company operates a fleet of vehicles and wants to understand patterns in usage to better allocate resources.

### Business Objectives
- Identify patterns in vehicle usage across different time periods (daily, weekly, monthly)
- Detect anomalies in vehicle utilization that may indicate inefficiencies
- Forecast future demand to optimize fleet size and deployment
- Reduce idle time by 15% within 6 months

### Success Criteria
- Accurate forecasting model with MAPE < 10%
- Identification of at least 3 actionable insights for fleet optimization
- Clear visualization of usage patterns for stakeholder presentation

## 2. Import Libraries

```python
# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import sys

# Add the project root directory to the Python path
sys.path.append('..')

# Import custom utility functions
from src.data_utils import (
    load_data,
    check_missing_values,
    handle_missing_values,
    detect_outliers,
    convert_to_datetime,
    plot_time_series
)

from src.visualization_utils import (
    set_plotting_style,
    plot_distribution,
    plot_correlation_heatmap,
    plot_boxplot
)

# Set plotting style
set_plotting_style()

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Set random seed for reproducibility
np.random.seed(42)
```

## 3. Data Collection

For this project, we'll generate synthetic transportation data that mimics real-world patterns. In a real project, you would load data from files, databases, or APIs.

```python
def generate_synthetic_transportation_data(start_date='2022-01-01', end_date='2023-12-31'):
    """
    Generate synthetic transportation data with realistic patterns.
    
    Parameters:
    -----------
    start_date : str
        Start date for the dataset
    end_date : str
        End date for the dataset
        
    Returns:
    --------
    pd.DataFrame
        Synthetic transportation dataset
    """
    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(date_range)
    
    # Base demand with weekly seasonality
    weekday_effect = np.array([1.2, 1.3, 1.4, 1.3, 1.5, 0.8, 0.7])  # Mon-Sun
    weekday_indices = np.array([d.weekday() for d in date_range])
    base_demand = np.array([weekday_effect[i] for i in weekday_indices])
    
    # Add monthly seasonality
    month_indices = np.array([d.month for d in date_range])
    monthly_effect = np.sin(np.pi * month_indices / 6) * 0.3 + 1.0  # Peak in summer
    
    # Add yearly trend (increasing)
    yearly_trend = np.linspace(0, 0.5, n_days)
    
    # Add holidays effect (major US holidays)
    holidays = [
        '2022-01-01', '2022-07-04', '2022-11-24', '2022-12-25',  # 2022 holidays
        '2023-01-01', '2023-07-04', '2023-11-23', '2023-12-25'   # 2023 holidays
    ]
    holiday_effect = np.zeros(n_days)
    for holiday in holidays:
        holiday_idx = np.where(date_range == holiday)[0]
        if len(holiday_idx) > 0:
            holiday_effect[holiday_idx[0]] = -0.5  # Reduced demand on holidays
    
    # Combine all effects
    demand = (base_demand * monthly_effect + yearly_trend + holiday_effect) * 100
    
    # Add random noise
    noise = np.random.normal(0, 10, n_days)
    demand = demand + noise
    
    # Add some outliers
    outlier_indices = np.random.choice(n_days, size=int(n_days * 0.02), replace=False)
    outlier_effect = np.random.choice([-50, 50], size=len(outlier_indices))
    demand[outlier_indices] += outlier_effect
    
    # Ensure no negative values
    demand = np.maximum(demand, 10)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': date_range,
        'vehicle_usage_hours': demand.round(2),
        'idle_time_hours': (24 - demand/10).round(2),
        'fuel_consumption_gallons': (demand * 0.5 + np.random.normal(0, 5, n_days)).round(2),
        'maintenance_cost': (demand * 2 + np.random.normal(0, 20, n_days)).round(2),
        'distance_miles': (demand * 15 + np.random.normal(0, 50, n_days)).round(2)
    })
    
    # Add vehicle types
    vehicle_types = ['Sedan', 'SUV', 'Van', 'Truck']
    df['vehicle_type'] = np.random.choice(vehicle_types, size=n_days)
    
    # Add regions
    regions = ['North', 'South', 'East', 'West', 'Central']
    df['region'] = np.random.choice(regions, size=n_days)
    
    # Add driver_id
    n_drivers = 20
    df['driver_id'] = np.random.randint(1, n_drivers + 1, size=n_days)
    
    # Add some missing values
    missing_indices = np.random.choice(n_days, size=int(n_days * 0.05), replace=False)
    df.loc[missing_indices, 'fuel_consumption_gallons'] = np.nan
    
    missing_indices = np.random.choice(n_days, size=int(n_days * 0.03), replace=False)
    df.loc[missing_indices, 'maintenance_cost'] = np.nan
    
    return df

# Generate synthetic data
transportation_data = generate_synthetic_transportation_data()

# Save the data to CSV
data_dir = os.path.join('..', 'data', 'raw')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    
transportation_data.to_csv(os.path.join(data_dir, 'transportation_data.csv'), index=False)

print(f"Generated synthetic transportation data with {len(transportation_data)} records.")
transportation_data.head()
```

## 4. Data Exploration

Let's explore the dataset to understand its structure, patterns, and potential issues.

```python
# Load the data
df = load_data(os.path.join(data_dir, 'transportation_data.csv'))

# Display basic information
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)

# Display summary statistics
print("\nSummary Statistics:")
df.describe().T
```

```python
# Check for missing values
missing_info = check_missing_values(df)
print("Missing Values:")
missing_info
```

```python
# Convert date column to datetime
df = convert_to_datetime(df, 'date', set_index=True)

# Plot time series for vehicle usage hours
plot_time_series(df, 'vehicle_usage_hours', title='Vehicle Usage Hours Over Time')
```

```python
# Plot distributions of key metrics
numeric_cols = ['vehicle_usage_hours', 'idle_time_hours', 'fuel_consumption_gallons', 
                'maintenance_cost', 'distance_miles']

for col in numeric_cols:
    plot_distribution(df, col)
```

```python
# Plot correlation heatmap
plot_correlation_heatmap(df, numeric_cols)
```

```python
# Analyze patterns by vehicle type
vehicle_type_stats = df.groupby('vehicle_type')[numeric_cols].mean()
vehicle_type_stats
```

```python
# Plot boxplots by vehicle type
plt.figure(figsize=(14, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x='vehicle_type', y=col, data=df.reset_index())
    plt.title(f'{col} by Vehicle Type')
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

```python
# Analyze weekly patterns
df['dayofweek'] = df.index.dayofweek
df['dayname'] = df.index.day_name()

weekly_patterns = df.groupby('dayname')[numeric_cols].mean()
# Reorder days
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
weekly_patterns = weekly_patterns.reindex(day_order)
weekly_patterns
```

```python
# Plot weekly patterns
plt.figure(figsize=(12, 6))
weekly_patterns['vehicle_usage_hours'].plot(kind='bar')
plt.title('Average Vehicle Usage Hours by Day of Week')
plt.ylabel('Hours')
plt.grid(axis='y')
plt.show()
```

```python
# Analyze monthly patterns
df['month'] = df.index.month
df['month_name'] = df.index.month_name()

monthly_patterns = df.groupby('month_name')[numeric_cols].mean()
# Reorder months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
               'July', 'August', 'September', 'October', 'November', 'December']
monthly_patterns = monthly_patterns.reindex(month_order)
monthly_patterns
```

```python
# Plot monthly patterns
plt.figure(figsize=(14, 6))
monthly_patterns['vehicle_usage_hours'].plot(kind='bar')
plt.title('Average Vehicle Usage Hours by Month')
plt.ylabel('Hours')
plt.grid(axis='y')
plt.show()
```

## 5. Data Cleaning and Preprocessing

```python
# Handle missing values
df_clean = handle_missing_values(df, method='interpolate')

# Check if missing values were handled
missing_after = check_missing_values(df_clean)
print("Missing Values After Handling:")
missing_after if not missing_after.empty else print("No missing values remaining")
```

```python
# Detect outliers
outliers = detect_outliers(df_clean, numeric_cols, method='zscore', threshold=3.0)

# Count outliers in each column
outlier_counts = {col: sum(outliers[col]) for col in outliers}
print("Outlier Counts:")
for col, count in outlier_counts.items():
    print(f"{col}: {count} outliers ({count/len(df_clean)*100:.2f}%)")
```

```python
# Visualize outliers for vehicle_usage_hours
plt.figure(figsize=(12, 6))
plt.scatter(df_clean.index, df_clean['vehicle_usage_hours'], 
            c=['red' if x else 'blue' for x in outliers['vehicle_usage_hours']], alpha=0.5)
plt.title('Vehicle Usage Hours with Outliers Highlighted')
plt.ylabel('Hours')
plt.grid(True)
plt.show()
```

```python
# Handle outliers by capping
def cap_outliers(df, columns, method='zscore', threshold=3.0):
    """
    Cap outliers at upper and lower bounds.
    """
    df_capped = df.copy()
    outliers = detect_outliers(df, columns, method=method, threshold=threshold)
    
    for col in columns:
        if col in outliers:
            if method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
            
            # Cap outliers
            df_capped.loc[df_capped[col] < lower_bound, col] = lower_bound
            df_capped.loc[df_capped[col] > upper_bound, col] = upper_bound
    
    return df_capped

# Apply outlier capping
df_clean = cap_outliers(df_clean, numeric_cols)

# Check if outliers were handled
outliers_after = detect_outliers(df_clean, numeric_cols, method='zscore', threshold=3.0)
outlier_counts_after = {col: sum(outliers_after[col]) for col in outliers_after}
print("Outlier Counts After Capping:")
for col, count in outlier_counts_after.items():
    print(f"{col}: {count} outliers ({count/len(df_clean)*100:.2f}%)")
```

```python
# Create additional features
df_clean['is_weekend'] = df_clean['dayofweek'].isin([5, 6]).astype(int)
df_clean['is_holiday'] = 0  # We would populate this with actual holiday data
df_clean['season'] = pd.cut(df_clean['month'], bins=[0, 3, 6, 9, 12], 
                          labels=['Winter', 'Spring', 'Summer', 'Fall'], 
                          include_lowest=True)

# Calculate efficiency metrics
df_clean['miles_per_gallon'] = df_clean['distance_miles'] / df_clean['fuel_consumption_gallons']
df_clean['cost_per_mile'] = df_clean['maintenance_cost'] / df_clean['distance_miles']
df_clean['utilization_rate'] = df_clean['vehicle_usage_hours'] / 24

# Display the processed data
df_clean.head()
```

```python
# Save the processed data
processed_dir = os.path.join('..', 'data', 'processed')
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)
    
df_clean.to_csv(os.path.join(processed_dir, 'transportation_data_processed.csv'))
print(f"Saved processed data to {os.path.join(processed_dir, 'transportation_data_processed.csv')}")
```

## 6. Initial Findings

Based on our initial exploration, here are some key findings:

1. **Temporal Patterns**:
   - Weekday usage is significantly higher than weekend usage
   - There's a seasonal pattern with higher usage during summer months
   - There's an increasing trend in vehicle usage over the analyzed period

2. **Vehicle Type Insights**:
   - Different vehicle types show distinct usage patterns
   - [Vehicle type with highest usage] has the highest utilization rate
   - [Vehicle type with lowest usage] has the lowest utilization rate

3. **Efficiency Metrics**:
   - Strong correlation between vehicle usage hours and fuel consumption
   - Maintenance costs increase with vehicle usage
   - Idle time is inversely related to vehicle usage

4. **Data Quality**:
   - Successfully handled missing values through interpolation
   - Identified and capped outliers to improve data quality
   - Created additional features to enhance analysis capabilities

In the next notebook, we'll perform more detailed exploratory data analysis, including time series decomposition and statistical tests.
