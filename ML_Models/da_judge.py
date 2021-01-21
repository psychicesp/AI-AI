#%%
import pandas as pd
import pickle
from scipy import stats as st

with open('pickle_jar/split_data.pickle', 'rb') as hamger_slices:
    X_train, X_hp_train, X_test, y_train,y_hp_train, y_test = pickle.load(hamger_slices)
with open('pickle_jar/knr_model.pickle', 'rb') as gherkin:
    knr_model = pickle.load(gherkin)
with open('pickle_jar/rf_model.pickle', 'rb') as dill:
    rf_model = pickle.load(dill)
price_row = pd.DataFrame(y_test)
input_df = X_test
#%%

knr_row = knr_model.predict(input_df)

rf_row = rf_model.predict(input_df)

input_df['rf_prediction'] = rf_row
input_df['knr_prediction'] = knr_row
input_df['combined_prediction'] = (input_df['rf_prediction'] + input_df['knr_prediction'])/2

input_df['rf_test'] = 1
input_df['knr_test'] = 1
input_df['combined_test'] = 1

input_df = input_df.merge(price_row, how = 'outer',on ='MLS')

for index, row in input_df.iterrows():
    if (abs(row['rf_prediction']-row['Price'])/row['Price']) > 0.10:
        input_df.loc[index, 'rf_test'] = 0 
    if (abs(row['knr_prediction']-row['Price'])/row['Price']) > 0.10:
        input_df.loc[index, 'knr_test'] = 0 
    if (abs(row['combined_prediction']-row['Price'])/row['Price']) > 0.10:
        input_df.loc[index, 'combined_test'] = 0 

#%%
input_df.to_csv('../CSVs/test.csv')
# %%
def da_judge(row_name):
    rows = len(input_df[f"{row_name}_test"].tolist())
    good_rows = input_df[f"{row_name}_test"].sum()
    print(f"The {row_name} model was within 5% on {round(((good_rows/rows)*100),2)}% of rows")
# %%
