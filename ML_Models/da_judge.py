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
with open('pickle_jar/line_model.pickle', 'rb') as dill:
    line_model = pickle.load(dill)
price_row = pd.DataFrame(y_test)
input_df = X_test
#%%

knr_row = knr_model.predict(input_df)

rf_row = rf_model.predict(input_df)

line_row = line_model.predict(input_df)

input_df['rf_prediction'] = rf_row
input_df['knr_prediction'] = knr_row
input_df['line_prediction'] = knr_row
input_df['combined_prediction'] = (input_df['rf_prediction'] + input_df['knr_prediction'])/2

input_df['rf_test'] = 1
input_df['knr_test'] = 1
input_df['line_test'] = 1
input_df['combined_test'] = 1

input_df['rf_off'] = 1
input_df['knr_off'] = 1
input_df['line_off'] = 1
input_df['combined_off'] = 1

input_df = input_df.merge(price_row, how = 'outer',on ='MLS')

def rf_guesser(row):
    return (abs(row['rf_prediction']-row['Price'])/row['Price'])
def knr_guesser(row):
    return (abs(row['knr_prediction']-row['Price'])/row['Price'])
def combined_guesser(row):
    return (abs(row['combined_prediction']-row['Price'])/row['Price'])
def line_guesser(row):
    return (abs(row['line_prediction']-row['Price'])/row['Price'])

input_df['rf_off'] = input_df.apply(rf_guesser, axis = 1)
input_df['knr_off'] = input_df.apply(knr_guesser, axis = 1)
input_df['combined_off'] = input_df.apply(combined_guesser, axis = 1)
input_df['line_off'] = input_df.apply(combined_guesser, axis = 1)

def rf_checker(row):
    if row['rf_off'] < 0.05:
        return 1
    else:
        return 0

def knr_checker(row):
    if row['knr_off'] < 0.05:
        return 1
    else:
        return 0
def combined_checker(row):
    if row['combined_off'] < 0.05:
        return 1
    else:
        return 0
def line_checker(row):
    if row['line_off'] < 0.05:
        return 1
    else:
        return 0

input_df['rf_test'] = input_df.apply(rf_checker, axis = 1)
input_df['knr_test'] = input_df.apply(knr_checker, axis = 1)
input_df['combined_test'] = input_df.apply(combined_checker, axis = 1)
input_df['line_test'] = input_df.apply(line_checker, axis = 1)

#%%
input_df.to_csv('../CSVs/test.csv')
# %%
def da_judge(row_name):
    rows = len(input_df[f"{row_name}_test"].tolist())
    good_rows = input_df[f"{row_name}_test"].sum()
    print(f"The {row_name} model was within 5% on {round(((good_rows/rows)*100),2)}% of rows")
# %%
