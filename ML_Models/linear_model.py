#%%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
file_location = "../CSVs/combined.csv"
df = pd.read_csv(file_location).set_index("MLS")
df = df[['Price', 'Bedrooms','Age','Square_Footage','Acres', 'Combined_Baths','Population Density','Median Home Value','Water_Land_Percent', 'Median Household Income', 'Years_since']]
df.dropna(inplace = True)
with open('pickle_jar/MLdf.pickle', 'wb') as kosher:
    pickle.dump(df,kosher)
df
#%%
X = df.drop('Price', axis = 1)
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = 0.80)

#A second split for a Hyperparameter tuning set for a final Fit/Tune/Test split of 60/20/20
X_train, X_hp_train, y_train, y_hp_train = train_test_split(X_train, y_train, random_state=0, train_size = 0.75)
split_data = (X_train, X_hp_train, X_test, y_train,y_hp_train, y_test)
with open('pickle_jar/split_data.pickle', 'wb') as dill:
    pickle.dump(split_data,dill)

#%%
line_model = LinearRegression()
line_model = line_model.fit(X_train,y_train)

print(f"Training Data Score: {line_model.score(X_train, y_train)}")
print(f"Testing Data Score: {line_model.score(X_test, y_test)}")
# %%
