import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# imports for DOT and seasonalNaive models
import re
from statsforecast.models import (
    # HoltWinters,
    # CrostonClassic as Croston, 
    # HistoricAverage,
    DynamicOptimizedTheta as DOT,
    SeasonalNaive,
    # AutoARIMA
)
from statsforecast import StatsForecast
# imports for polynomialRegression model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# imports for randomForestRegressor model
from sklearn.ensemble import RandomForestRegressor
# imports for kNeighborsRegressor model
from sklearn.neighbors import KNeighborsRegressor


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
        # pythonDF.loc[i, 'features'] =  ssData['featId'].iloc[i]
        pythonDF.loc[i, 'his'] =  ssData['data'].iloc[i]#.to_dataframe()
        
    return pythonDF

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

def dynamic_optimized_theta(df, length_of_missing_data, data_logging_interval, dqStart):
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

def polynomial_regression(df, length_of_missing_data, data_logging_interval, dqStart, featureNumber):

    """
    Inputs
    df: df used for training set (from SS)
    dqStart: start of the predictions

    Output
    forecasts_df: dataframe with predictions for the period missing data. Index names as ts
    """

    # Drop all NaN
    # df = df.dropna()

    # Splitting variables
    y = df[df.columns[0]]  # independent variable
    X = df[[df.columns[featureNumber+1]]]  # dependent variable

    # Filter data for training and testing
    X_train = X[X.index < dqStart]
    y_train = y[X.index < dqStart]
    X_test = X[X.index >= dqStart]
    #y_test = y[X.index >= dqStart]

    # Generate polynomial features
    poly = PolynomialFeatures(degree = 4)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train polynomial regression model on the whole dataset
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    # Create a new DataFrame with the timestamp as index and y_pred as values
    pred_df = pd.DataFrame(data=y_pred, index=X_test.index, columns=['y_pred'])

    return pred_df

def random_Forest_Regressor(df, length_of_missing_data, data_logging_interval, dqStart):
    """
    Input
    master_table: main table received from SS

    Output
    df: dataframe with predictions for all rows with missing columns. Index names as ts
    """
    X = df.iloc[:,1:-1]
    y = df.iloc[:,0:1]  

    X_train = X[X.index < dqStart]
    X_test = X[X.index >= dqStart]
    y_train = y[y.index < dqStart]

    # Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
    
    # Fit the regressor with x and y data
    regressor.fit(X_train, y_train)
    pred = regressor.predict(X_test)
    predictions = pd.DataFrame(data=pred, index=X_test.index, columns=['y_pred'])
    
    return predictions

def kNeighbors_Regressor_Uniform(df, length_of_missing_data, data_logging_interval, dqStart):
    """
    Input
    df:data table from SS with the ts as index
    dqStart: start datetime

    Output
    df: dataframe with predictions for all rows with missing data inclusive of the start date and end date. Index names as ts

    #Uniform: gives each data point equal weight
    """
    X = df.iloc[:,1:-1]
    y = df.iloc[:,0:1]  

    X_train = X[X.index < dqStart]
    X_test = X[X.index >= dqStart]
    y_train = y[y.index < dqStart]

    knn_regressor = KNeighborsRegressor(n_neighbors=3,weights="uniform")
    knn_regressor.fit(X_train, y_train)
    pred = knn_regressor.predict(X_test)
    predictions = pd.DataFrame(data=pred, index=X_test.index, columns=['y_pred'])

    return predictions

def ensemble_model(python_master_table, filter=True):
    """
    Function to run all models, and return the one with lowest RMSE.
    Models running through the ensemble model will have input DataFrame (AKA the "his" column on master_table) 
    with timestamp as index, target variable as first column, feature variables as the rest of the columns.

    Make sure the output predictions of all models are INCLUSIVE of both the "start ts" and "end ts" (AKA
    last ts with real data before gap, and first ts with real data after gap) 

    Make sure to follow camelCase for DataFrame column naming for compatibility with SS

    filter = True. Parameter to show all predictions for each dq issue, or only ones filtered with rmse and mape.
    """


    # dictionary to save predictions for each point
    scores_df_dict = {
    "pointID": [],
    "predictions": [],
    "rmse": [],
    "mape": [],
    "modelName": []
    }
    # Create a DataFrame from the dictionary
    scores_df = pd.DataFrame(scores_df_dict)


    for i, row in python_master_table.iterrows():

        #-----------------
        # INPUTS TO MODELS
        #-----------------
        # point id on SS
        pointID = row["pointID"]
        # data logging interval of the point 
        data_logging_interval = row["pointInterval"]
        freq = str(data_logging_interval.total_seconds()/3600)+"h"
        # the history df containing training data for the models
        df = row["his"]#.to_dataframe()                                      #### IMPORTANT : UNCOMMENT THIS ON SS  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        df.set_index(df.columns[0], inplace=True, drop=True)
        df = df.asfreq(freq)
        # dqStart and duration of dq issue (dqEnd = dqStart+length_of_missing_data)
        dqStart = row["dqStart"]
        length_of_missing_data = row["dqDuration"]


        # ----------------------
        # Training/Testing sets 
        # ----------------------
        data_before_gap = df.loc[:dqStart][:-1] # :-1 to not include the first ts with nulls, since df.loc[:dqStart] will include the first ts with NAN
        # doing a bfill and ffill to make sure there are no nulls in the training data. This step should be taken care of on SS by interpolating. This is done as a cautionary measure as nulls in the data will cause some models to fail
        data_before_gap.bfill(inplace=True)
        # data_before_gap.ffill(inplace=True)
        train_data = data_before_gap.loc[:dqStart-length_of_missing_data][:-1]
        test_data = data_before_gap.loc[dqStart-length_of_missing_data:dqStart]
        # test_data timestamps used to slice the predictions_for_rmse df to have exact dimension as testing set. This prevents raising error due to mismatching lengths when using the rmse or mape functions
        test_data_timestamps = pd.date_range(start=test_data.index[0], end=test_data.index[-1], freq=test_data.index.freq)


        # ----------------------------
        # Timestamps for null duration
        # ----------------------------
        start = row['dqStart']
        duration = row['dqDuration']
        interval = row['pointInterval']
        timestamps = pd.date_range(start=start, end=start+duration, freq=interval)[1:-1] # clipping the first and last timestamps, as they already exist with actual data on SS


        #----------------------------
        # Dict of Data Quality Models                              ############# ADD NEW MODELS HERE ################
        #----------------------------

        # UNIVARIATE Models
        dq_models_univariate = {
            "Seasonal Naive" : seasonal_naive,
            "Dynamic Optimized Theta": dynamic_optimized_theta
        }

        # MULTIVARIATE Models using one feature at a time
        dq_models_multivariate_1feature = {
            "Polynomial Regression" : polynomial_regression,
        }

        # MULTIVARIATE Models using all features to predict target
        dq_models_multivariate = {
            "Random Forest Regressor" : random_Forest_Regressor,
            "KNN Regressor Uniform ": kNeighbors_Regressor_Uniform,
            # "XGBoost 1": xgboost_1,
            # "XGboost 2": xgboost_2,
            # "XGboost 3": xgboost_3
        }


        ############################
        # LOOP FOR UNIVARIATE MODELS
        ############################
        for model_name, model in dq_models_univariate.items():

            #------------------------
            # ** Calculating RMSE **
            #------------------------
            # the prediction. USED ONLY TO EVALUATE RMSE
            predictions_for_rmse = model(train_data, length_of_missing_data, data_logging_interval, dqStart)
            predictions_for_rmse = predictions_for_rmse[predictions_for_rmse.index.isin(test_data_timestamps)]
            rmse_score = mean_squared_error(test_data[test_data.columns[0]].to_numpy(), predictions_for_rmse[predictions_for_rmse.columns[0]].to_numpy(), squared=False)
            mape_score = mean_absolute_percentage_error(test_data[test_data.columns[0]].to_numpy(), predictions_for_rmse[predictions_for_rmse.columns[0]].to_numpy())


            #------------------
            # ** Predictions **
            #------------------
            # the predictions. USED FOR DATA CLEANING (uses all the data as training)
            predictions_for_data_quality = model(df, length_of_missing_data, data_logging_interval, dqStart)

            # keep only timestamps for null periods (rows where there are null values on SS)
            predictions_for_data_quality = predictions_for_data_quality[predictions_for_data_quality.index.isin(timestamps)]

            # reset index to make the ts a column instead of index. SS doesnt show the index of a DF
            predictions_for_data_quality = predictions_for_data_quality.reset_index()

            # rename the ts and predictions column to "ts" and "predictions", to have similar naming for all ouutputs of models (makes it easier as well when using the dcInsert function on SS.)
            predictions_for_data_quality.columns = ["ts", "predictions"]

            # append data to the scores DF
            row_to_append = {'pointID': pointID, 'predictions': predictions_for_data_quality, 
                                "rmse": rmse_score, "mape": mape_score,
                                "modelName": model_name, 
                                "identifier": 
                                    str(row["pointID"])
                                    +str(row["dqStart"])
                                    +str(row["dqDuration"])
                                    +str(row["dqType"])}

            scores_df = pd.concat([scores_df, pd.DataFrame([row_to_append])], ignore_index=True)


        ##############################
        # Loop for MULTIVARIATE MODELS
        ##############################
        if len(df.columns)>1:  # only run multivariate if there are features available to use from the master table
            
            ##############
            # Multivariate models where all features are used to predict the target
            for model_name, model in dq_models_multivariate.items():
                #------------------------
                # ** Calculating RMSE **
                #------------------------
                # the prediction. USED ONLY TO EVALUATE RMSE
                predictions_for_rmse = model(data_before_gap, length_of_missing_data, data_logging_interval, dqStart-length_of_missing_data)
                predictions_for_rmse = predictions_for_rmse[predictions_for_rmse.index.isin(test_data_timestamps)]  
                rmse_score = mean_squared_error(test_data[test_data.columns[0]].to_numpy(), predictions_for_rmse[predictions_for_rmse.columns[0]].to_numpy(), squared=False)
                mape_score = mean_absolute_percentage_error(test_data[test_data.columns[0]].to_numpy(), predictions_for_rmse[predictions_for_rmse.columns[0]].to_numpy())

                #------------------
                # ** Predictions **
                #------------------
                # the predictions. USED FOR DATA CLEANING (uses all the data as training)
                predictions_for_data_quality = model(df, length_of_missing_data, data_logging_interval, dqStart)

                # keep only timestamps for null periods (rows where there are null values on SS)
                predictions_for_data_quality = predictions_for_data_quality[predictions_for_data_quality.index.isin(timestamps)]   # UNCOMMENT THIS ONCE THE interval of the data column in SS master table is corrected

                # reset index to make the ts a column instead of index. SS doesnt show the index of a DF
                predictions_for_data_quality = predictions_for_data_quality.reset_index()

                # rename the ts and predictions column to "ts" and "predictions", to have similar naming for all ouutputs of models (makes it easier as well when using the dcInsert function on SS.)
                predictions_for_data_quality.columns = ["ts", "predictions"]

                # append data to the scores DF
                row_to_append = {'pointID': pointID, 'predictions': predictions_for_data_quality, 
                                "rmse": rmse_score, "mape": mape_score,
                                "modelName": model_name, 
                                "identifier": 
                                    str(row["pointID"])
                                    +str(row["dqStart"])
                                    +str(row["dqDuration"])
                                    +str(row["dqType"])}

                scores_df = pd.concat([scores_df, pd.DataFrame([row_to_append])], ignore_index=True)

            ##############
            # Multivariate models where 1 feature is used at a time to predict the target
            for model_name, model in dq_models_multivariate_1feature.items():
                    
                    # Loop over different features to be used one at a time
                    for featureNumber, featureName in enumerate(df.columns.tolist()[1:]):

                        #------------------------
                        # ** Calculating RMSE **
                        #------------------------
                        # the prediction. USED ONLY TO EVALUATE RMSE
                        predictions_for_rmse = model(data_before_gap, length_of_missing_data, data_logging_interval, dqStart-length_of_missing_data, featureNumber)
                        predictions_for_rmse = predictions_for_rmse[predictions_for_rmse.index.isin(test_data_timestamps)]  
                        rmse_score = mean_squared_error(test_data[test_data.columns[0]].to_numpy(), predictions_for_rmse[predictions_for_rmse.columns[0]].to_numpy(), squared=False)
                        mape_score = mean_absolute_percentage_error(test_data[test_data.columns[0]].to_numpy(), predictions_for_rmse[predictions_for_rmse.columns[0]].to_numpy())

                        #------------------
                        # ** Predictions **
                        #------------------
                        # the predictions. USED FOR DATA CLEANING (uses all the data as training)
                        predictions_for_data_quality = model(df, length_of_missing_data, data_logging_interval, dqStart, featureNumber)

                        # keep only timestamps for null periods (rows where there are null values on SS)
                        predictions_for_data_quality = predictions_for_data_quality[predictions_for_data_quality.index.isin(timestamps)]

                        # reset index to make the ts a column instead of index. SS doesnt show the index of a DF
                        predictions_for_data_quality = predictions_for_data_quality.reset_index()

                        # rename the ts and predictions column to "ts" and "predictions", to have similar naming for all ouutputs of models (makes it easier as well when using the dcInsert function on SS.)
                        predictions_for_data_quality.columns = ["ts", "predictions"]

                        # append data to the scores DF
                        row_to_append = {'pointID': pointID, 'predictions': predictions_for_data_quality, 
                                        "rmse": rmse_score, "mape": mape_score,
                                        "modelName": model_name+" - Feature: " + str(featureName), 
                                        "identifier": 
                                            str(row["pointID"])
                                            +str(row["dqStart"])
                                            +str(row["dqDuration"])
                                            +str(row["dqType"])}

                        scores_df = pd.concat([scores_df, pd.DataFrame([row_to_append])], ignore_index=True)

    # filtering on RMSE and MAPE
    if filter:
        # keep only predictions with mean absolute percentage error <10%
        scores_df = scores_df[scores_df.mape < 0.1]

        # return predictions with least RMSE for each point/dq issue
        idx = scores_df.groupby('identifier')['rmse'].idxmin()
        scores_df = scores_df.loc[idx].reset_index(drop=True)

    return scores_df.drop(columns=["identifier"])

