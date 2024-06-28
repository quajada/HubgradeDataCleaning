import pandas as pd
from data_extraction.dummy_data_extractor import extract_dummy_data
from data_extraction.skyspark_data_extractor import extractData
from models.seasonalNaive import seasonalNaive
from models.dynamic_optimized_theta import dynamic_optimized_theta
from project.models.iterativeImputation import iterative_Imputation

from sklearn.metrics import mean_squared_error
from statsforecast import StatsForecast

import re
from statsforecast.models import (
    DynamicOptimizedTheta as DOT,
    SeasonalNaive,
)

#_______________________________
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
#_______________________________


master_table = extract_dummy_data("dummy_data")


def extractData(data):
    """
    Function that extracts data for python from the SS grid.

    Input:
    - data: hisGrid (<class 'hxpy.haystack.grid.Grid>)
    Output:
    - DataFrame with following columns 
        - pointID => point id of target variable
        - unit
        - dqType => type of data quality issue
        - dqStart => timestamp of start of data quality issue
        - dqDuration => duration of data quality issue
        - pointInterval => logging interval for the point
        - features => point ids of model features
        - his => history to be used as training data

    ** NOTE_: this function is written to mainly be compatable with python on SS. Running it locally will not work (since it is designed for 
    an input of <class 'hxpy.haystack.grid.Grid> type from SS) 
    
    """

    # convert the Grid object to df to be able to manipulate it (capitalizing on the hxPy facilitation using the .to_dataframe() function)
    ssData = data.to_dataframe()

    # initiate a new empty dataframe to construct the output
    pythonDF = pd.DataFrame()

    # loop over the ssData and extract the data from each row
    for i in range(len(ssData)):
        pythonDF.loc[i, 'pointID'] = ssData['id'].iloc[i]
        pythonDF.loc[i, 'unit'] = ssData["unit"].iloc[i]
        pythonDF.loc[i, 'dqType'] = ssData["dqType"].iloc[i]
        pythonDF.loc[i, 'dqStart'] = ssData['ts'].iloc[i]
        pythonDF.loc[i, 'dqDuration'] = pd.Timedelta(ssData['dur'].iloc[i], "min")
        pythonDF.loc[i, 'pointInterval'] =  pd.Timedelta(ssData["freq"].iloc[i], "min" )
        pythonDF.loc[i, 'features'] =  ssData['featId'].iloc[i]
        pythonDF.loc[i, 'his'] =  ssData['data'].iloc[i]#.to_dataframe()
        
    return pythonDF

def seasonal_naive(df, length_of_missing_data, data_logging_interval):
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

    # format the df to statsforecast format
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: "y"})
    df['unique_id'] = "v0"    

    # number of predictions
    horizon = int(length_of_missing_data/data_logging_interval) + 1 # why -1? because if you do length_of_missing_data/data_logging_interval you will get prediction length that is exclusive of the start ts (start ts is the last ts with actual data before the gap), and inclusive of the end ts (end ts is the first ts with actual data after the gap). +1 to get predictions also for the start and end timestamp. Can remove them later

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

def dynamic_optimized_theta(df, length_of_missing_data, data_logging_interval):
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

    # format the df to statsforecast format
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: 'ds', df.columns[1]: "y"})
    df['unique_id'] = "v0"    

    # number of predictions
    horizon = int(length_of_missing_data/data_logging_interval) + 1 # why -1? because if you do length_of_missing_data/data_logging_interval you will get prediction length that is exclusive of the start ts (start ts is the last ts with actual data before the gap), and inclusive of the end ts (end ts is the first ts with actual data after the gap). +1 to get predictions also for the start and end timestamp. Can remove them later

    # season length
    season_length = int(pd.Timedelta(24, 'h') / data_logging_interval)      

    # frequency
    freq = str(data_logging_interval.total_seconds()/3600)+"h"


    # LIST OF MODELS
    models = [
        DOT(season_length=season_length) 
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
            
    forecasts_df = forecasts_df.rename(columns={"ds": "timestamp", "DynamicOptimizedTheta":"dynamicOptimizedTheta"})

    forecasts_df.set_index("timestamp", inplace=True)

    return forecasts_df

def ensemble_model(python_master_table):
    """
    Function to run all models, and return the one with lowest RMSE.
    Models running through the ensemble model will have input DataFrame (AKA the "his" column on master_table) 
    with timestamp as index, target variable as first column, feature variables as the rest of the columns.

    Make sure the output predictions of all models are INCLUSIVE of both the "start ts" and "end ts" (AKA
    last ts with real data before gap, and first ts with real data after gap) 

    Make sure to follow camelCase for DataFrame column naming for compatibility with SS
    """

    # dictionary to save predictions for each point
    scores_df_dict = {
    "pointID": [],
    "predictions": [],
    "rmse": [],
    "modelName": []
    }

    # Create a DataFrame from the dictionary
    scores_df = pd.DataFrame(scores_df_dict)

    for i, row in python_master_table.iterrows():

        #-----------------
        # INPUTS TO MODELS
        #-----------------

        pointID = row["pointID"]
        df = row["his"]#.to_dataframe()                           #### IMPORTANT : UNCOMMENT THIS ON SS
        df.set_index(df.columns[0], inplace=True, drop=True)
        length_of_missing_data = row["dqDuration"]
        data_logging_interval = row["pointInterval"]


        #----------------------------
        # Dict of Data Quality Models                              ############# ADD NEW MODELS HERE 
        #----------------------------

        dq_models = {
            "Seasonal Naive" : seasonal_naive,
            "Dynamic Optimized Theta": dynamic_optimized_theta
        }

        for model_name, model in dq_models.items():
            
            #------------------------
            # ** Calculating RMSE **
            #------------------------

            # number of predictions needed
            horizon = int(length_of_missing_data/data_logging_interval) +1 # why +1? because if you do length_of_missing_data/data_logging_interval you will get prediction length that is exclusive of the start ts (start ts is the last ts with actual data before the gap), and inclusive of the end ts (end ts is the first ts with actual data after the gap). +1 to get predictions INCLUSIVE of BOTH start and end ts

            # training set size (relative to the horizon/prediction size)
            training_set_size = horizon * 10

            # training / testing set to evaluate the model (relative to horizon of prediction)
            train_data = df.iloc[-1*int(training_set_size):-1*int(horizon)]
            test_data = df.iloc[-1*int(horizon):]

            # the prediction. USED ONLY TO EVALUATE RMSE
            predictions_for_rmse = model(df = train_data, length_of_missing_data = length_of_missing_data, data_logging_interval = data_logging_interval)
            rmse_score = mean_squared_error(test_data[test_data.columns[0]].to_numpy(), predictions_for_rmse[predictions_for_rmse.columns[0]].to_numpy(), squared=False)

            #------------------
            # ** Predictions **
            #------------------

            # the predictions. USED FOR DATA CLEANING (uses all the data as training)
            predictions_for_data_quality = model(df, length_of_missing_data, data_logging_interval)

            # keep only timestamps for null periods (rows where there are null values on SS)
            start = row['dqStart']
            duration = row['dqDuration']
            interval = row['pointInterval']
            timestamps = pd.date_range(start=start, end=start+duration, freq=interval)[1:-1] # clipping the first and last timestamps, as they already exist with actual data on SS

            predictions_for_data_quality = predictions_for_data_quality[predictions_for_data_quality.index.isin(timestamps)]

            # reset index to make the ts a column instead of index. SS doesnt show the index of a DF
            predictions_for_data_quality = predictions_for_data_quality.reset_index()

            # append data to the scores DF
            row_to_append = {'pointID': pointID, 'predictions': predictions_for_data_quality, 
                            "rmse": rmse_score, "modelName": model_name, 
                            "identifier": 
                                str(row["pointID"])
                                +str(row["dqStart"])
                                +str(row["dqDuration"])
                                +str(row["dqType"])}
            
            scores_df = pd.concat([scores_df, pd.DataFrame([row_to_append])], ignore_index=True)

            # return predictions with least RMSE for each point/dq issue
            idx = scores_df.groupby('identifier')['rmse'].idxmin()
            scores_df = scores_df.loc[idx].reset_index(drop=True)
            
    return scores_df    

