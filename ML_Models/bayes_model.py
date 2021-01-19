#%%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.model_selection import GridSearchCV
#%%
with open('pickle_jar/split_data.pickle', 'rb') as dill:
    X_train, X_hp_train, X_test, y_train,y_hp_train, y_test = pickle.load(dill)

#%%
bayes_model = BayesianRidge()
bayes_model = bayes_model.fit(X_train,y_train)

print(f"Training Data Score: {bayes_model.score(X_train, y_train)}")
print(f"Testing Data Score: {bayes_model.score(X_test, y_test)}")
# %%
param_grid = {
    'n_iter':[100,200,300,400],
    'tol':[0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005],
    'alpha_1': [0.00005,0.00001,0.000005,0.000001,0.0000005,0.0000001],
    'alpha_2': [0.00005,0.00001,0.000005,0.000001,0.0000005,0.0000001],
    'lambda_1': [0.00005,0.00001,0.000005,0.000001,0.0000005,0.0000001],
    'lambda_2':[0.00005,0.00001,0.000005,0.000001,0.0000005,0.0000001],
    'compute_score':[True, False],
    'fit_intercept':[True, False],
    'normalize':[True, False]
}
grid = GridSearchCV(bayes_model, param_grid, verbose=3, cv = 5, n_jobs = -3)
grid.fit(X_hp_train, y_hp_train)
#%%
print(grid.best_params_)
print(grid.best_score_)
#%%
with open('pickle_jar/bayes_model.pickle', 'wb') as bread_butter:
    pickle.dump(bayes_model,bread_butter)
#%%
# elastic_model = ElasticNet()
# elastic_model = elastic_model.fit(X_train,y_train)

# print(f"Training Data Score: {elastic_model.score(X_train, y_train)}")
# print(f"Testing Data Score: {elastic_model.score(X_test, y_test)}")
# # %%
# param_grid = {
#     'max_iter':[100,200,300,400, 500, 1000, 2000],
#     'tol':[0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005],
#     'l1': [0.5,0.4,0.3,0.2,0.1],
#     'alpha': [1.0, 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1],
#     'selection':['cyclic','random'],
#     'fit_intercept':[True, False],
#     'normalize':[True, False]
# }
# grid = GridSearchCV(elastic_model, param_grid, verbose=3, cv = 5, n_jobs = -3)
# grid.fit(X_hp_train, y_hp_train)
# #%%
# print(grid.best_params_)
# print(grid.best_score_)
# #%%
# with open('pickle_jar/elastic_model.pickle', 'wb') as spicy:
#     pickle.dump(elastic_model,spicy)

