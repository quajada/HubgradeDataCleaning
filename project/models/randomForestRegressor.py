from project.data_extraction.dummy_data_extractor import extract_dummy_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def random_Forest_Regressor(master_table):
    """
    Input
    master_table: main table received from SS

    Output
    df: dataframe with predictions for all rows with missing columns. Index names as ts
    """
    master_table = master_table.at[0,"his"]
    mt = master_table.set_index(["ts"])

    # Tag and filter rows with missing
    mt["status"] = mt.isna().any(axis=1)
    mt_predict = mt[mt["status"]==1]
    X_predict = mt_predict.iloc[:,0:1] 

    # Filtered master table
    mt_train = mt.dropna()
    mt_train

    # Load the dataset
    # X = mt.iloc[:,1:-1]  Enable for SS
    # y = mt.iloc[:,0:1]   Enable for SS

    y = mt_train.iloc[:,1:-1]    #Custom due to sample dataset
    X = mt_train.iloc[:,0:1]     #Custom due to sample dataset

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Check for and handle categorical variables
    # label_encoder = LabelEncoder()
    # x_categorical = mt_train.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
    # x_numerical = mt_train.select_dtypes(exclude=['object']).values
    # x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values

    # Fitting Random Forest Regression to the dataset
    regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)
    
    # Fit the regressor with x and y data
    regressor.fit(X_train, y_train)
    
    # Access the OOB Score
    # oob_score = regressor.oob_score_
    # print(f'Out-of-Bag Score: {oob_score}')
    
    # Making predictions on the same data or new data
    predictions_test = regressor.predict(X_test)
    
    # Evaluating the model
    mse = mean_squared_error(y_test, predictions_test)
    # print(f'Mean Squared Error: {mse}')
    
    r2 = r2_score(y_test, predictions_test)
    # print(f'R-squared: {r2}')

    predict = regressor.predict(X_predict)
    df = pd.DataFrame(data=predict, index=X_predict.index, columns=['y_pred'])
    
    return df