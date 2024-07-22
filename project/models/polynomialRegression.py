import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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