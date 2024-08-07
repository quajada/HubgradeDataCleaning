from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

def kNeighbors_Regressor_Uniform(df, length_of_missing_data, data_logging_interval, dqStart):
    """
    Input
    master_table: main table received from SS

    Output
    df: dataframe with predictions for all rows with missing columns. Index names as ts
    """
    df = df.at[0,"his"]
    mt = df.set_index(["ts"])

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
    
    # # Apply KNN regression
    knn_regressor = KNeighborsRegressor(n_neighbors=3,weights="uniform")
    knn_regressor.fit(X_train, y_train)
    predictions = knn_regressor.predict(X_test)
    predictions
    # Evaluate the model
    print('Score:', knn_regressor.score(X_test, y_test))

    predict = knn_regressor.predict(X_predict)
    df = pd.DataFrame(data=predict, index=X_predict.index, columns=['y_pred'])
    
    return df