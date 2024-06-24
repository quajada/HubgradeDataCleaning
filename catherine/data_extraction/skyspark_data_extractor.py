import pandas as pd

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