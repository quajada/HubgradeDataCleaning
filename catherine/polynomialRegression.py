import pandas as pd

def polynomial_regression(df, dqStart):

    """
    Inputs
    df: df used for training set (from SS)
    dqStart: start of the predictions

    Output
    forecasts_df: dataframe with predictions for the period missing data. Index names as ts
    """

    # step 1 convert the grid to a dataframe, set first column as index and drop all NaN
    df = df.to_dataframe()
    df.set_index(df.columns[0], inplace=True, drop=True)
    df = df.dropna()

    # Splitting variables
    y = df[df.columns[0]]  # independent variable
    X = df[[df.columns[1]]]  # dependent variable

    # Filter data for training and testing
    X_train = X[X.index < dqStart]
    X_test = X[X.index >= dqStart]
    y_train = y[X.index < dqStart]
    #y_test = y[X.index >= dqStart]

    # Generate polynomial features
    poly = PolynomialFeatures(degree = 4)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train polynomial regression model on the whole dataset
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    return y_pred