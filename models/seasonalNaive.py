import pandas as pd

def seasonalNaive(df, length_of_missing_data, data_logging_interval):
    """
    Inputs
    df: df used for training set (from SS)
    length_of_missing_data: interval length of missing data (from SS)
    data_logging_interval: data logging interval - called from the hisDQInterval tag on the point (from SS)

    Output
    forecasts_df: dataframe with predictions for the period missing data. Index names as ts, values column named as "v0
    """
    
    # step 1 convert the grid to a dataframe, and set first column as index
    #df = df.to_dataframe()
    #df.set_index(df.columns[0], inplace=True, drop=True)

    # rename the first column as "target"
    new_column_name = "target"
    df = df.rename(columns={df.columns[0]: new_column_name})

    # number of predictions
    horizon = int(length_of_missing_data/data_logging_interval)
    
    # season length
    season_length = int(pd.Timedelta(24, 'h') / data_logging_interval)      

    # frequency
    #freq = str(data_logging_interval.total_seconds()/3600)+"h"

    # The Model
    model = SeasonalNaive(season_length=season_length)
        
    # Model fitting
    model = model.fit(y=df["target"])
    
    # Predictions
    forecasts_df = model.predict(h=horizon)
    forecasts_df = pd.DataFrame(forecasts_df)

    forecasts_df = forecasts_df.rename(columns={forecasts_df.columns[0]: "predictions"})

    return forecasts_df#.reset_index()