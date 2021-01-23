#%%
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
#%%
with open('pickle_jar/split_data.pickle', 'rb') as dill:
    X_train, X_hp_train, X_test, y_train,y_hp_train, y_test = pickle.load(dill)

def scoring_function(y,x):
    return np.sum(np.abs((x-y)/y) <0.05)/len(x)

my_scorer = make_scorer(scoring_function, greater_is_better = True)

#%%
knr_model = KNeighborsRegressor(
    n_jobs = -3,
    algorithm = 'auto',
    leaf_size = 20,
    n_neighbors = 10,
    p = 1,
    weights = 'distance')
knr_model = knr_model.fit(X_train,y_train)

print(f"Training Data Score: {knr_model.score(X_train, y_train)}")
print(f"Testing Data Score: {knr_model.score(X_test, y_test)}")
print(scoring_function(y_test, knr_model.predict(X_test)))

# %%
param_grid = {
    'n_neighbors':[3,4,5,6,7,8,9,10,15,20],
    'leaf_size':[10,20,30,40,50,60,70,80,90,100],
    'p': [1,2],
    'weights':['distance'],
    'algorithm':['auto', 'ball_tree', 'kd_tree'],
    'n_jobs':[1]
}
grid = GridSearchCV(knr_model, param_grid, verbose=4, cv = 5, n_jobs = -4, scoring = my_scorer)
grid.fit(X_hp_train, y_hp_train)
print(grid.best_params_)
print(grid.best_score_)
#%%
with open('pickle_jar/knr_model.pickle', 'wb') as bread_butter:
    pickle.dump(knr_model,bread_butter)

