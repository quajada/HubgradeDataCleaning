import pandas as pd

def extract_dummy_data(path):
    df = pd.read_csv(path+"/masterTable1_.csv")
    df_data1 = pd.read_csv(path+"/masterTable1_data_1.csv", index_col=[0], date_format="%Y-%m-%dT%H:%M:%S%z Dubai").reset_index()  
    # df_data2 = pd.read_csv(path+"/masterTable1_data_2.csv", index_col=[0], date_format="%Y-%m-%dT%H:%M:%S%z Dubai").reset_index()
    df_data = pd.DataFrame({"data":[df_data1] })  
    df.loc[:, "data"] = df_data

    pythonDF = pd.DataFrame()
    # loop over the ssData and extract the data from each row
    for i in range(len(df)):
        pythonDF.loc[i, 'pointID'] = df['id'].iloc[i]
        pythonDF.loc[i, 'unit'] = df["unit"].iloc[i]
        pythonDF.loc[i, 'dqType'] = df["dqType"].iloc[i]
        pythonDF.loc[i, 'dqStart'] = pd.to_datetime(df['ts'].iloc[i], format="%Y-%m-%dT%H:%M:%S%z Dubai")
        pythonDF.loc[i, 'dqDuration'] = pd.Timedelta(df['dur'].iloc[i])
        pythonDF.loc[i, 'pointInterval'] =  pd.Timedelta(df["freq"].iloc[i])
        # pythonDF.loc[i, 'features'] =  df['featId'].iloc[i]

    pythonDF.loc[:, 'his'] =  df['data']

    ## Cleaning any units or special characters from all features except the ts column. This is done AUTOMATICALLY on SS when using the .to_dataframe() function
    ## Converting to float, this is also done automatically on SS from the .to_dataframe() function
    for i in range(len(pythonDF['his'])):
        for col in pythonDF["his"].iloc[i].columns[1:]:
            pythonDF["his"].iloc[i][col] = pythonDF["his"].iloc[i][col].astype(str).str.replace(r'[^\d.]', '', regex=True)   ## \d => all digits   ///  .  => dots    /////  [^  ]  => keep charachters that are mentioned in the brackets
            pythonDF["his"].iloc[i][col] = pythonDF["his"].iloc[i][col].replace('', 'NaN').astype(float)    
            
    return pythonDF
