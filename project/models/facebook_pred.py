import pandas as pd
from prophet import Prophet
from data_extraction.dummy_data_extractor import extract_dummy_data

def facebook_pred(df, length_of_missing_data, data_logging_interval, dqStart):
    df.reset_index(inplace=True)
    df = df.dropna()

    # Keep only the first two columns
    df = df.iloc[:, :2]

    # Rename columns
    df.columns = ['ds', 'temp']

    # Remove ' Dubai' from the datetime strings
    df['ds'] = df['ds'].str.replace(' Dubai', '', regex=False)

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

    # Initialize Prophet model with tuned hyperparameters
    model_temp = Prophet(seasonality_mode='additive',  # Adjust based on data exploration
                         interval_width=0.95,          # Adjust prediction interval if needed
                         changepoint_prior_scale=0.01) # Tune based on data patterns

    # Fit the model
    model_temp.fit(df)

    # Number of predictions
    samples = int(length_of_missing_data / data_logging_interval) + 1

    # Create future DataFrame
    future_temp = model_temp.make_future_dataframe(periods=samples, freq='5T')

    # Predict the future values
    forecast_temp = model_temp.predict(future_temp)

    # Convert dq_start to timezone-naive
    dq_start = pd.Timestamp(dqStart).tz_convert('Asia/Dubai').tz_localize(None)

    # Ensure 'ds' column in forecast_temp is timezone-naive
    forecast_temp['ds'] = forecast_temp['ds'].dt.tz_localize(None)

    # Filter predictions to start from dq_start
    predictions = forecast_temp[forecast_temp['ds'] >= dq_start][['ds', 'yhat']]

    # Set 'ds' as the index
    predictions.set_index('ds', inplace=True)

    return predictions