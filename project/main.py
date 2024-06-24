import pandas as pd
from data_extraction.dummy_data_extractor import extract_dummy_data
from data_extraction.skyspark_data_extractor import extract_data
from models.seasonalNaive import seasonalNaive
#_______________________________
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
#_______________________________

master_table = extract_dummy_df("C:\Users\nramirez\OneDrive - Enova Facilities Management\Documents\enovaDataCleaning\HubgradeDataCleaning\dummy_data")

for row in master_table.iter():
    run models
    extract RMSE
    
find best models (ensemble model)
run prediction
return to skyspark