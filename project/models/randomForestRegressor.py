from sklearn.ensemble import RandomForestRegressor
import pandas as pd

def random_Forest_Regressor(df, length_of_missing_data, data_logging_interval, dqStart):
    """
    Input
    master_table: main table received from SS

    Output
    df: dataframe with predictions for all rows with missing columns. Index names as ts
    """
    X = df[[df.columns[0]]]
    y = df[df.columns[1:]]

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