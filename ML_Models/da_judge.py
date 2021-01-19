#%%
import pandas as pd
import pickle
from scipy import stats as st

with open('pickle_jar/MLdf.pickle', 'rb') as kosher:
    test_df = pickle.load(kosher)
with open('pickle_jar/knr_model.pickle', 'rb') as gherkin:
    knr_model = pickle.load(gherkin)
with open('pickle_jar/rf_model.pickle', 'rb') as dill:
    rf_model = pickle.load(dill)
input_df = test_df.drop('Price', axis = 1)
#%%

knr_row = knr_model.predict(input_df)

rf_row = rf_model.predict(input_df)

test_df['rf_prediction'] = rf_row
test_df['knr_prediction'] = knr_row
test_df['combined_prediction'] = (test_df['rf_prediction'] + test_df['knr_prediction'])/2

test_df['rf_test'] = 1
test_df['knr_test'] = 1
test_df['combined_test'] = 1

for index, row in test_df.iterrows():
    if (abs(row['rf_prediction']-row['Price'])/row['Price']) > 0.5:
        test_df.loc[index, 'rf_test'] = 0 
    if (abs(row['knr_prediction']-row['Price'])/row['Price']) > 0.5:
        test_df.loc[index, 'knr_test'] = 0 
    if (abs(row['combined_prediction']-row['Price'])/row['Price']) > 0.5:
        test_df.loc[index, 'combined_test'] = 0 

#%%
test_df.to_csv('../CSVs/test.csv')
# %%
