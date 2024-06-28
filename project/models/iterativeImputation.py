import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def iterative_Imputation(master_table):
    """
    Input
    master_table: main table received from SS

    Output
    mt_predict: dataframe with predictions for all rows with missing columns. Index names as ts
    """
    master_table = master_table.at[0,"his"]
    mt = master_table.set_index(["ts"])
    mt["status"] = mt.isna().any(axis=1)
    imputer = IterativeImputer()
    imputed = imputer.fit_transform(mt)
    mt_imputed = pd.DataFrame(imputed, index=mt.index, columns=mt.columns)
    mt_predict = mt_imputed[mt_imputed["status"]==1].drop(["status"],axis=1)
    
    return mt_predict