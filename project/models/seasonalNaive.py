import pandas as pd

def seasonal_naive(df, length_of_missing_data, data_logging_interval, dqStart):
    """
    Inputs
    df: df used for training set (from SS)
    length_of_missing_data: interval length of missing data (from SS)
    data_logging_interval: data logging interval - called from the hisDQInterval tag on the point (from SS)

    Output
    forecasts_df: dataframe with predictions for the period missing data. Index names as ts, values column named as "v0
    """
    
    
    # step 1 convert the grid to a dataframe, and set first column as index     ### UNCOMMENT THIS ONLY IF RUNNING THE MODEL DIRECTLY ON SS. THIS IS DONE IN THE ENSEMBLE MODEL SO NO NEED TO HAVE THIS WHEN RUNNING THROUGH ENSEMBLE MODEL
    #df = df.to_dataframe()
    #df.set_index(df.columns[0], inplace=True, drop=True)

    # rename the first column as "target"
    new_column_name = "target"
    df = df.rename(columns={df.columns[0]: new_column_name})

    # keep only the history BEFORE the start of the data quality issue, since this is a statisitcal model not ML model
    df = df[df.index < dqStart]

    # format the df to statsforecast format
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: "y"})
    df['unique_id'] = "v0"    

    # number of predictions
    horizon = int(length_of_missing_data/data_logging_interval) #+ 1 # why -1? because if you do length_of_missing_data/data_logging_interval you will get prediction length that is exclusive of the start ts (start ts is the last ts with actual data before the gap), and inclusive of the end ts (end ts is the first ts with actual data after the gap). +1 to get predictions also for the start and end timestamp. Can remove them later

    # season length
    season_length = int(pd.Timedelta(24, 'h') / data_logging_interval)      

    # frequency
    freq = str(data_logging_interval.total_seconds()/3600)+"h"


    # LIST OF MODELS
    models = [
        SeasonalNaive(season_length=season_length) 
    ]

    # The Model
    sf = StatsForecast( 
        models=models,
        freq=freq, 
        # fallback_model = SeasonalNaive(season_length=season_length),
        n_jobs=-1,
    )

    # Model fitting
    forecasts_df = sf.forecast(df=df[["ds", "y", "unique_id"]], h=horizon, level=[90])  

    # removing the -hi- and -lo- columns
    for col in forecasts_df.columns:
        if re.search("-hi-", col) or re.search("-lo-", col):
            forecasts_df.drop(col, axis=1, inplace=True)
            
    forecasts_df = forecasts_df.rename(columns={"ds": "timestamp", "SeasonalNaive":"seasonalNaive"})

    forecasts_df.set_index("timestamp", inplace=True)

    return forecasts_df