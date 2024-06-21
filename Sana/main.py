import pandas as pd
from data_extraction.dummy_data_extractor import extract_dummy_data
from data_extraction.skyspark_data_extractor import extract_data
from models.seasonalNaive import seasonalNaive


master_table = extract_dummy_df("C:\Users\nramirez\OneDrive - Enova Facilities Management\Documents\enovaDataCleaning\HubgradeDataCleaning\dummy_data")

for row in master_table.iter():
    run models
    extract RMSE
    
find best models (ensemble model)
run prediction
return to skyspark