"""
Utility functions for generating reports and insights from time series analysis.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Optional, Dict, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime


def create_summary_stats(df: pd.DataFrame, 
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Create a summary statistics table for selected columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : Optional[List[str]]
        Columns to include in the summary. If None, use all numeric columns.
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics dataframe
    """
    if columns is None:
        # Select only numeric columns
        columns = df.select_dtypes(include=['number']).columns
    
    # Calculate summary statistics
    summary = df[columns].describe().T
    
    # Add additional statistics
    summary['missing'] = df[columns].isnull().sum()
    summary['missing_pct'] = (df[columns].isnull().sum() / len(df) * 100).round(2)
    summary['unique'] = df[columns].nunique()
    
    # Reorder columns
    summary = summary[['count', 'missing', 'missing_pct', 'unique', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    
    return summary


def generate_time_patterns_report(df: pd.DataFrame, 
                                  target_column: str,
                                  date_column: Optional[str] = None,
                                  figsize: Tuple[int, int] = (15, 12)) -> Dict[str, Any]:
    """
    Generate a report on time-based patterns in the data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_column : str
        Target column to analyze
    date_column : Optional[str]
        Date column to use. If None, assumes the index is a datetime.
    figsize : Tuple[int, int]
        Figure size for plots
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary with time pattern analysis results
    """
    # Ensure we have a datetime index or column
    if date_column is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex when date_column is None")
        df_copy = df.copy()
    else:
        df_copy = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
            df_copy[date_column] = pd.to_datetime(df_copy[date_column])
        df_copy = df_copy.set_index(date_column)
    
    # Create time-based aggregations
    daily_avg = df_copy[target_column].resample('D').mean()
    weekly_avg = df_copy[target_column].resample('W').mean()
    monthly_avg = df_copy[target_column].resample('M').mean()
    quarterly_avg = df_copy[target_column].resample('Q').mean()
    yearly_avg = df_copy[target_column].resample('Y').mean()
    
    # Day of week and month patterns
    df_copy['dayofweek'] = df_copy.index.dayofweek
    df_copy['month'] = df_copy.index.month
    
    day_of_week_avg = df_copy.groupby('dayofweek')[target_column].mean()
    month_avg = df_copy.groupby('month')[target_column].mean()
    
    # Create plots
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # Daily and weekly averages
    daily_avg.plot(ax=axes[0, 0], title=f'Daily Average {target_column}')
    weekly_avg.plot(ax=axes[0, 1], title=f'Weekly Average {target_column}')
    
    # Monthly and quarterly averages
    monthly_avg.plot(ax=axes[1, 0], title=f'Monthly Average {target_column}')
    quarterly_avg.plot(ax=axes[1, 1], title=f'Quarterly Average {target_column}')
    
    # Day of week and month patterns
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_avg.index = [day_names[i] for i in day_of_week_avg.index]
    day_of_week_avg.plot(kind='bar', ax=axes[2, 0], title=f'Day of Week Average {target_column}')
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_avg.index = [month_names[i-1] for i in month_avg.index]
    month_avg.plot(kind='bar', ax=axes[2, 1], title=f'Month Average {target_column}')
    
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'daily_avg': daily_avg,
        'weekly_avg': weekly_avg,
        'monthly_avg': monthly_avg,
        'quarterly_avg': quarterly_avg,
        'yearly_avg': yearly_avg,
        'day_of_week_avg': day_of_week_avg,
        'month_avg': month_avg
    }


def generate_anomaly_report(series: pd.Series, 
                            method: str = 'zscore',
                            threshold: float = 3.0,
                            figsize: Tuple[int, int] = (12, 6)) -> pd.DataFrame:
    """
    Detect and report anomalies in a time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series to analyze
    method : str
        Method to detect anomalies ('zscore', 'iqr', or 'rolling')
    threshold : float
        Threshold for anomaly detection
    figsize : Tuple[int, int]
        Figure size for the plot
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with detected anomalies
    """
    if method == 'zscore':
        # Z-score method
        from scipy import stats
        z_scores = np.abs(stats.zscore(series.dropna()))
        anomalies = series.dropna()[z_scores > threshold]
    
    elif method == 'iqr':
        # IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        anomalies = series[(series < lower_bound) | (series > upper_bound)]
    
    elif method == 'rolling':
        # Rolling mean and std method
        rolling_mean = series.rolling(window=24).mean()
        rolling_std = series.rolling(window=24).std()
        upper_bound = rolling_mean + threshold * rolling_std
        lower_bound = rolling_mean - threshold * rolling_std
        anomalies = series[(series > upper_bound) | (series < lower_bound)]
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Plot the series with anomalies
    plt.figure(figsize=figsize)
    plt.plot(series.index, series, label='Original')
    plt.scatter(anomalies.index, anomalies, color='red', label='Anomalies')
    
    if method == 'rolling':
        plt.plot(upper_bound.index, upper_bound, 'g--', label='Upper Bound')
        plt.plot(lower_bound.index, lower_bound, 'g--', label='Lower Bound')
    
    plt.title(f'Anomaly Detection using {method.capitalize()} Method')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Create anomaly report
    anomaly_df = pd.DataFrame({
        'timestamp': anomalies.index,
        'value': anomalies.values,
        'deviation_pct': ((anomalies - series.mean()) / series.mean() * 100).values.round(2)
    })
    
    return anomaly_df


def create_html_report(title: str, 
                       sections: List[Dict[str, Any]],
                       output_path: str) -> str:
    """
    Create an HTML report with multiple sections.
    
    Parameters:
    -----------
    title : str
        Report title
    sections : List[Dict[str, Any]]
        List of sections, each with 'title', 'content', and optional 'figures'
    output_path : str
        Path to save the HTML report
        
    Returns:
    --------
    str
        Path to the saved HTML report
    """
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 20px;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #3498db;
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                text-align: left;
                padding: 8px;
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .figure {{
                margin: 20px 0;
                text-align: center;
            }}
            .figure img {{
                max-width: 100%;
                height: auto;
            }}
            .figure-caption {{
                font-style: italic;
                color: #666;
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .footer {{
                margin-top: 50px;
                border-top: 1px solid #ddd;
                padding-top: 10px;
                font-size: 0.8em;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    # Add sections
    for section in sections:
        html_content += f"""
        <div class="section">
            <h2>{section['title']}</h2>
            <div>{section['content']}</div>
        """
        
        # Add figures if present
        if 'figures' in section and section['figures']:
            for i, fig in enumerate(section['figures']):
                fig_path = f"figure_{section['title'].replace(' ', '_').lower()}_{i}.png"
                fig_full_path = os.path.join(os.path.dirname(output_path), fig_path)
                fig.savefig(fig_full_path, bbox_inches='tight')
                
                html_content += f"""
                <div class="figure">
                    <img src="{fig_path}" alt="{section['title']} Figure {i+1}">
                    <div class="figure-caption">Figure {i+1}: {section.get('figure_captions', [''])[i] if i < len(section.get('figure_captions', [])) else ''}</div>
                </div>
                """
        
        html_content += "</div>"
    
    # Add footer
    html_content += f"""
        <div class="footer">
            <p>This report was automatically generated by the Time Series Analysis Project.</p>
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path


def dataframe_to_html(df: pd.DataFrame, max_rows: int = 100) -> str:
    """
    Convert a DataFrame to HTML table.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to convert
    max_rows : int
        Maximum number of rows to include
        
    Returns:
    --------
    str
        HTML representation of the DataFrame
    """
    if len(df) > max_rows:
        df_display = pd.concat([df.head(max_rows//2), df.tail(max_rows//2)])
        html = df_display.to_html(classes='dataframe')
        html += f"<p><i>Note: Showing {max_rows} rows out of {len(df)} total rows.</i></p>"
    else:
        html = df.to_html(classes='dataframe')
    
    return html


def generate_executive_summary(results: Dict[str, Any], 
                               metrics: Dict[str, float],
                               insights: List[str]) -> str:
    """
    Generate an executive summary of the analysis.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Dictionary with analysis results
    metrics : Dict[str, float]
        Dictionary with performance metrics
    insights : List[str]
        List of key insights
        
    Returns:
    --------
    str
        HTML formatted executive summary
    """
    summary = """
    <div class="executive-summary">
        <h3>Executive Summary</h3>
        <p>This analysis provides insights into the time series data, identifying patterns, trends, and anomalies.</p>
        
        <h4>Key Performance Metrics</h4>
        <ul>
    """
    
    # Add metrics
    for metric, value in metrics.items():
        summary += f"<li><strong>{metric}:</strong> {value:.4f}</li>"
    
    summary += """
        </ul>
        
        <h4>Key Insights</h4>
        <ul>
    """
    
    # Add insights
    for insight in insights:
        summary += f"<li>{insight}</li>"
    
    summary += """
        </ul>
    </div>
    """
    
    return summary
