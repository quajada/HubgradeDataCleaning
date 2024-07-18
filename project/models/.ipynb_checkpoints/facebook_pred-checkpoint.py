{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3fb7d4e3-93eb-4be4-be30-3a269b3c48e2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from prophet import Prophet\n",
    "from data_extraction.dummy_data_extractor import extract_dummy_data\n",
    "\n",
    "def facebook_pred(df, length_of_missing_data, data_logging_interval, dqStart):\n",
    "    df.reset_index(inplace=True)\n",
    "    df = df.dropna()\n",
    "\n",
    "    # Keep only the first two columns\n",
    "    df = df.iloc[:, :2]\n",
    "\n",
    "    # Rename columns\n",
    "    df.columns = ['ds', 'temp']\n",
    "\n",
    "    # Remove ' Dubai' from the datetime strings\n",
    "    df['ds'] = df['ds'].str.replace(' Dubai', '', regex=False)\n",
    "\n",
    "    # Convert the 'ds' column to datetime format\n",
    "    df['ds'] = pd.to_datetime(df['ds'], format=\"%Y-%m-%dT%H:%M:%S%z\")\n",
    "\n",
    "    # Drop rows where datetime parsing failed\n",
    "    df = df.dropna(subset=['ds'])\n",
    "\n",
    "    # Clean temperature column and convert to numeric\n",
    "    df['temp'] = df['temp'].str.replace('°C', '').astype(float)\n",
    "\n",
    "    # Rename columns for convenience\n",
    "    df.columns = ['ds', 'y']\n",
    "\n",
    "    # Ensure 'ds' column is timezone-naive\n",
    "    df['ds'] = df['ds'].dt.tz_localize(None)\n",
    "\n",
    "    # Initialize Prophet model with tuned hyperparameters\n",
    "    model_temp = Prophet(seasonality_mode='additive',  # Adjust based on data exploration\n",
    "                         interval_width=0.95,          # Adjust prediction interval if needed\n",
    "                         changepoint_prior_scale=0.01) # Tune based on data patterns\n",
    "\n",
    "    # Fit the model\n",
    "    model_temp.fit(df)\n",
    "\n",
    "    # Number of predictions\n",
    "    samples = int(length_of_missing_data / data_logging_interval) + 1\n",
    "\n",
    "    # Create future DataFrame\n",
    "    future_temp = model_temp.make_future_dataframe(periods=samples, freq='5T')\n",
    "\n",
    "    # Predict the future values\n",
    "    forecast_temp = model_temp.predict(future_temp)\n",
    "\n",
    "    # Convert dq_start to timezone-naive\n",
    "    dq_start = pd.Timestamp(dqStart).tz_convert('Asia/Dubai').tz_localize(None)\n",
    "\n",
    "    # Ensure 'ds' column in forecast_temp is timezone-naive\n",
    "    forecast_temp['ds'] = forecast_temp['ds'].dt.tz_localize(None)\n",
    "\n",
    "    # Filter predictions to start from dq_start\n",
    "    predictions = forecast_temp[forecast_temp['ds'] >= dq_start][['ds', 'yhat']]\n",
    "\n",
    "    # Set 'ds' as the index\n",
    "    predictions.set_index('ds', inplace=True)\n",
    "\n",
    "    return predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e7ef4b83-32c2-4f97-83ca-d502a8426409",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example usage:\n",
    "master_table = extract_dummy_data(\"dummy_data\")\n",
    "df = master_table.at[1, \"his\"].iloc[:, :2].copy()\n",
    "\n",
    "# Extract values from the second row of master_table\n",
    "length_of_missing_data = pd.Timedelta(master_table.at[1, \"dqDuration\"])\n",
    "data_logging_interval = pd.Timedelta(master_table.at[1, \"pointInterval\"])\n",
    "dqStart = master_table.at[1, \"dqStart\"]\n",
    "\n",
    "# Call the function\n",
    "predictions = xgboost_1(df, length_of_missing_data, data_logging_interval, dqStart)\n",
    "\n",
    "# Display the resulting dataframe\n",
    "print(predictions.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec7065d2-0690-4c8e-aa46-5672ab67ffb6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}