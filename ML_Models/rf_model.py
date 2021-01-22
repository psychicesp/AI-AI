#%%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np
#%%
#A second split for a Hyperparameter tuning set for a final Fit/Tune/Test split of 60/20/20
with open('pickle_jar/split_data.pickle', 'rb') as dill:
    X_train, X_hp_train, X_test, y_train,y_hp_train, y_test = pickle.load(dill)

def scoring_function(y,x):
    return np.sum(np.abs((x-y)/y) <0.05)/len(x)
my_scorer = make_scorer(scoring_function, greater_is_better = True)
#%%
rf_model = RandomForestRegressor(random_state = 91, n_jobs = -3, max_features = 'sqrt', min_samples_leaf=8, min_samples_split=2, n_estimators = 160, max_depth = 25)
rf_model = rf_model.fit(X_train,y_train)
#%%
print(f"Training Data Score: {rf_model.score(X_train, y_train)}")
print(f"Testing Data Score: {rf_model.score(X_test, y_test)}")
print(scoring_function(y_test, rf_model.predict(X_test)))
# %%
param_grid = {
    'max_depth':[25],
    'n_jobs':[1],
    'min_samples_leaf': [15,6,7,8,9,10,11,12],
    'min_samples_split': [2,3,4,5,6,7,8,9,10],
    'n_estimators': [160],
    'max_features':['sqrt', 'log2'],
    'random_state':[91]
}
grid = GridSearchCV(rf_model, param_grid, verbose=3, cv = 5, n_jobs = -3, scoring = my_scorer)
grid.fit(X_hp_train, y_hp_train)
print(grid.best_params_)
print(grid.best_score_)
#%%
with open('pickle_jar/rf_model.pickle', 'wb') as bread_butter:
    pickle.dump(rf_model,bread_butter)
# %%
