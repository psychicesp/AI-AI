#%%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
#%%
#A second split for a Hyperparameter tuning set for a final Fit/Tune/Test split of 60/20/20
with open('pickle_jar/split_data.pickle', 'rb') as dill:
    X_train, X_hp_train, X_test, y_train,y_hp_train, y_test = pickle.load(dill)

#%%
rf_model = RandomForestRegressor(random_state = 91, n_jobs = -3, max_features = 'sqrt', min_samples_leaf=1, min_samples_split=4, n_estimators = 50, max_depth = 20)
rf_model = rf_model.fit(X_train,y_train)

print(f"Training Data Score: {rf_model.score(X_train, y_train)}")
print(f"Testing Data Score: {rf_model.score(X_test, y_test)}")
# %%
param_grid = {
    'max_depth':[20,30,50,75],
    'n_jobs':[1],
    'min_samples_leaf': [1,2,3],
    'min_samples_split': [3,4,5],
    'n_estimators': [250,375,400,425],
    'max_features':['auto','sqrt'],
    'random_state':[91]
}
grid = GridSearchCV(rf_model, param_grid, verbose=3, cv = 5, n_jobs = -3)
grid.fit(X_hp_train, y_hp_train)
print(grid.best_params_)
print(grid.best_score_)
#%%
with open('pickle_jar/rf_model.pickle', 'wb') as bread_butter:
    pickle.dump(rf_model,bread_butter)