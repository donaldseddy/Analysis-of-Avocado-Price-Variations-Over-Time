#bibliothèque pandas pour la manipulation des données
import pandas as pd
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import unicodedata
import re



def naive_forecast_model(train, test,label='visits'):
    """Generate naive forecasts by repeating the last observed value."""
    naive_forecast = train[label].iloc[-1]
    naive_forecasts = pd.Series(naive_forecast, index=test.index)
    return naive_forecasts

def seasonal_naive_forecast_model(train, test,label='visits'):
    """Generate seasonal naive forecasts by shifting the series by 7 days."""
    seasonal_naive_forecasts = train[label].shift(7).iloc[-30:]
    seasonal_naive_forecasts.index = test.index
    return seasonal_naive_forecasts

def moving_average_forecast_model(train, test,label='visits'):
    """Generate moving average forecasts with a window of 7 days."""
    moving_average_forecasts = train[label].rolling(window=7).mean().iloc[-30:]
    moving_average_forecasts.index = test.index
    return moving_average_forecasts

def evaluate_forecasts(true_values, forecast_values):
    """Calculate MAE, RMSE, and MAPE between true values and forecast values."""
    mae = mean_absolute_error(true_values, forecast_values)
    rmse = np.sqrt(mean_squared_error(true_values, forecast_values))
    mape = np.mean(np.abs((true_values - forecast_values) / true_values)) * 100
    return mae, rmse, mape



def normalize_columns(df):
    new_cols = []
    for col in df.columns:
        col_norm = unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('utf-8')
        col_norm = col_norm.lower()
        col_norm = re.sub(r'[^a-z0-9]+', '_', col_norm)
        col_norm = col_norm.strip('_')
        new_cols.append(col_norm)
    
    df.columns = new_cols
    return df

