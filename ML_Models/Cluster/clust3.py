#%%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
import numpy as np

with open('cluster_pickles/df3.pickle','rb') as dill:
    df = pickle.load(dill)

X = df.drop('Price', axis = 1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=8, train_size = 0.80)
X_train, X_hp_train, y_train, y_hp_train = train_test_split(X_train, y_train, random_state=5, train_size = 0.75)
def scoring_function(y,x):
    return np.sum(np.abs((x-y)/y) <0.05)/len(x)
my_scorer = make_scorer(scoring_function, greater_is_better = True)
#%%
print('----------Linear Model----------')
line3 = LinearRegression()
line3 = line3.fit(X,y)

print(f"Training Data Score: {line3.score(X, y)}")
print(f"Custom Score: {scoring_function(y_test, line3.predict(X_test))}")
print('--------------------------------')
columns = X_train.columns
weights = line3.coef_

for i, column in enumerate(columns):
    print(f"{column}: {weights[i]}")
#%%
knr3 = KNeighborsRegressor(
    n_jobs = -3,
    algorithm = 'auto',
    leaf_size = 10,
    n_neighbors = 20,
    p = 1,
    weights = 'distance')
knr3 = knr3.fit(X_train,y_train)
print('----------K Neighbors Regressor----------')
print(f"Training Data Score: {knr3.score(X_train, y_train)}")
print(f"Testing Data Score: {knr3.score(X_test, y_test)}")
print(f"Custom Score: {scoring_function(y_test, knr3.predict(X_test))}")
#%%
param_grid = {
    'n_neighbors':[3,4,5,6,7,8,9,10,15,20],
    'leaf_size':[10,20,30,40,50,60,70,80,90,100],
    'p': [1,2],
    'weights':['uniform','distance'],
    'algorithm':['auto', 'ball_tree', 'kd_tree'],
    'n_jobs':[1]
}
grid = GridSearchCV(knr3, param_grid, verbose=4, cv = 5, n_jobs = -4, scoring = my_scorer)
grid.fit(X_hp_train, y_hp_train)
print(grid.best_params_)
print(grid.best_score_)
#%%
rf3 = RandomForestRegressor(
            random_state = 91, 
            n_jobs = -3, 
            max_features = 'sqrt', 
            min_samples_leaf= 3,
            min_samples_split=7, 
            n_estimators = 20, 
            max_depth = 12,
            bootstrap = False)
rf3 = rf3.fit(X_train,y_train)

print('----------Random Forest Regressor----------')
print(f"Training Data Score: {rf3.score(X_train, y_train)}")
print(f"Testing Data Score: {rf3.score(X_test, y_test)}")
print(f"Custom Score: {scoring_function(y_test, rf3.predict(X_test))}")
# %%
param_grid = {
    'max_depth':[12,25,50,75,100,125,150],
    'n_jobs':[1],
    'min_samples_leaf': [2,3,4],
    'min_samples_split': [6,7,8],
    'n_estimators': [66,100,110,150,200,250],
    'max_features':['auto','sqrt'],
    'bootstrap':[True, False]
}
grid = GridSearchCV(rf3, param_grid, verbose=3, cv = 5, n_jobs = -3, scoring = my_scorer)
grid.fit(X_hp_train, y_hp_train)
print(grid.best_params_)
print(grid.best_score_)
#%%
neural3 = MLPRegressor(
    alpha = 0.00005,
    learning_rate="adaptive",
    learning_rate_init=0.01,
    tol=0.000005
)
neural3 = neural3.fit(X_train,y_train)

print('----------Neural Net----------')
print(f"Training Data Score: {neural3.score(X_train, y_train)}")
print(f"Testing Data Score: {neural3.score(X_test, y_test)}")
print(f"Custom Score: {scoring_function(y_test, neural3.predict(X_test))}")
# %%
param_grid = {
    'alpha':[0.00005, 0.00001, 0.000005, 0.000001],
    'learning_rate': ['constant','adaptive'],
    'learning_rate_init': [0.5, 0.1, 0.05, 0.01, 0.005],
    'tol': [0.00005, 0.00001, 0.000005, 0.000001]
}
grid = GridSearchCV(neural3, param_grid, verbose=3, cv = 5, n_jobs = -3, scoring = my_scorer)
grid.fit(X_hp_train, y_hp_train)
print(grid.best_params_)
print(grid.best_score_)

# %%
with open('cluster_pickles/model3.pickle', 'wb') as dill:
    pickle.dump(rf3,dill)
# %%
