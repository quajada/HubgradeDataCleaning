from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

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

    X_train = X[X.index <= dqStart]
    X_test = X[X.index >= dqStart]
    y_train = y[y.index <= dqStart]

    knn_regressor = KNeighborsRegressor(n_neighbors=3,weights="uniform")
    knn_regressor.fit(X_train, y_train)
    pred = knn_regressor.predict(X_test)
    predictions = pd.DataFrame(data=pred, index=X_test.index, columns=['y_pred'])

    return predictions