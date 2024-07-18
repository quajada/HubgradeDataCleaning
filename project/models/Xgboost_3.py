import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import numpy as np
from datetime import datetime, timedelta, timezone
from data_extraction.dummy_data_extractor import extract_dummy_data

def xgboost_3(df, length_of_missing_data, data_logging_interval, dqStart):
    df.reset_index(inplace=True)
    df = df.dropna()

    # Keep only the first two columns
    df = df.iloc[:, :2]

    # Rename columns
    df.columns = ['ds', 'temp']

    # Remove ' Dubai' from the datetime strings
    df['ds'] = df['ds'].astype(str).str.replace(' Dubai', '', regex=False)

    # Convert the 'ds' column to datetime format
    df['ds'] = pd.to_datetime(df['ds'], format="%Y-%m-%dT%H:%M:%S%z")

    # Drop rows where datetime parsing failed
    df = df.dropna(subset=['ds'])

    # Clean temperature column and convert to numeric
    df['temp'] = df['temp'].str.replace('°C', '').astype(float)

    # Rename columns for convenience
    df.columns = ['ds', 'y']

    # Ensure 'ds' column is timezone-naive
    df['ds'] = df['ds'].dt.tz_localize(None)

    # Extract numerical features from datetime
    df['year'] = df['ds'].dt.year
    df['month'] = df['ds'].dt.month
    df['day'] = df['ds'].dt.day
    df['hour'] = df['ds'].dt.hour
    df['minute'] = df['ds'].dt.minute

    # Create future DataFrame starting from dq_start
    future_periods = int(length_of_missing_data / data_logging_interval) + 1
    dq_start = pd.Timestamp(dqStart).tz_convert('Asia/Dubai').tz_localize(None)
    future_temp = pd.DataFrame()
    future_temp['ds'] = [dq_start + timedelta(minutes=5 * i) for i in range(future_periods)]

    # Extract features for XGBoost
    future_temp['year'] = future_temp['ds'].dt.year
    future_temp['month'] = future_temp['ds'].dt.month
    future_temp['day'] = future_temp['ds'].dt.day
    future_temp['hour'] = future_temp['ds'].dt.hour
    future_temp['minute'] = future_temp['ds'].dt.minute

    # Initialize XGBoost model with parameters to reduce noise
    model_temp = xgb.XGBRegressor(
        n_estimators=90,   # Number of boosting rounds
        max_depth=1,        # Maximum depth of each tree
        learning_rate=0.1,  # Learning rate
        min_child_weight=1, # Minimum sum of instance weight needed in a child
        subsample=0.8,      # Subsample ratio of the training instances
        colsample_bytree=0.8,  # Subsample ratio of columns when constructing each tree
        objective='reg:squarederror'  # Objective function for regression task
    )

    # Fit the model
    model_temp.fit(df[['year', 'month', 'day', 'hour', 'minute']], df['y'])

    # Predict the future values
    future_temp['yhat'] = model_temp.predict(future_temp[['year', 'month', 'day', 'hour', 'minute']])

    # Filter predictions to start from dq_start
    predictions = future_temp[['ds', 'yhat']]

    # Set 'ds' as the index
    predictions.set_index('ds', inplace=True)

    return predictions