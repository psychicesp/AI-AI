#%%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.cluster import KMeans
import numpy as np

#%%
#A second split for a Hyperparameter tuning set for a final Fit/Tune/Test split of 60/20/20
with open('pickle_jar/split_data.pickle', 'rb') as dill:
    X_train, X_hp_train, X_test, y_train,y_hp_train, y_test = pickle.load(dill)

def scoring_function(y,x):
    return np.sum(np.abs((x-y)/y) <0.05)/len(x)
my_scorer = make_scorer(scoring_function, greater_is_better = True)

cluster = KMeans()
cluster.fit(X_train)

for index, row in X_train.iterrows():
    row = np.array(row)
    row = np.reshape(row,(1,-1))
    X_train.loc[index, 'cluster'] = cluster.predict(row)

X0_train = X_train[X_train['cluster']==0]
X1_train = X_train[X_train['cluster']==1]
X2_train = X_train[X_train['cluster']==2]
X3_train = X_train[X_train['cluster']==3]
X4_train = X_train[X_train['cluster']==4]
X5_train = X_train[X_train['cluster']==5]
X6_train = X_train[X_train['cluster']==6]
X7_train = X_train[X_train['cluster']==7]

X0_train = X0_train.drop('cluster',axis = 1)
X1_train = X1_train.drop('cluster',axis = 1)
X2_train = X2_train.drop('cluster',axis = 1)
X3_train = X3_train.drop('cluster',axis = 1)
X4_train = X4_train.drop('cluster',axis = 1)
X5_train = X5_train.drop('cluster',axis = 1)
X6_train = X6_train.drop('cluster',axis = 1)
X7_train = X7_train.drop('cluster',axis = 1)

for index, row in X_test.iterrows():
    row = np.array(row)
    row = np.reshape(row,(1,-1))
    X_test.loc[index, 'cluster'] = cluster.predict(row)

X0_test = X_test[X_test['cluster']==0]
X1_test = X_test[X_test['cluster']==1]
X2_test = X_test[X_test['cluster']==2]
X3_test = X_test[X_test['cluster']==3]
X4_test = X_test[X_test['cluster']==4]
X5_test = X_test[X_test['cluster']==5]
X6_test = X_test[X_test['cluster']==6]
X7_test = X_test[X_test['cluster']==7]

X0_test = X0_test.drop('cluster',axis = 1)
X1_test = X1_test.drop('cluster',axis = 1)
X2_test = X2_test.drop('cluster',axis = 1)
X3_test = X3_test.drop('cluster',axis = 1)
X4_test = X4_test.drop('cluster',axis = 1)
X5_test = X5_test.drop('cluster',axis = 1)
X6_test = X6_test.drop('cluster',axis = 1)
X7_test = X7_test.drop('cluster',axis = 1)
#%%
y_train = y_train.reset_index()
y_test = y_test.reset_index()
#%%
X0_train = X0_train.reset_index()
X0_train = X0_train.merge(y_train, how = 'left', on ='MLS')

X0_train = X0_train.set_index('MLS')
y0_train = X0_train['Price']
X0_train.drop('Price', axis = 1, inplace = True)


X0_test = X0_test.reset_index()
X0_test = X0_test.merge(y_test, how = 'left', on ='MLS')

X0_test = X0_test.set_index('MLS')
y0_test = X0_test['Price']
X0_test.drop('Price', axis = 1, inplace = True)

rf0 = RandomForestRegressor(random_state = 91, n_jobs = -3, max_features = 'sqrt', min_samples_leaf=8, min_samples_split=2, n_estimators = 160, max_depth = 25)
rf0.fit(X0_train, y0_train)

print(f"Training Data Score: {rf0.score(X0_train, y0_train)}")
print(f"Testing Data Score: {rf0.score(X0_test, y0_test)}")
print(scoring_function(y0_test, rf0.predict(X0_test)))

#%%

X1_train = X1_train.reset_index()
X1_train = X1_train.merge(y_train, how = 'left', on ='MLS')

X1_train = X1_train.set_index('MLS')
y1_train = X1_train['Price']
X1_train.drop('Price', axis = 1, inplace = True)


X1_test = X1_test.reset_index()
X1_test = X1_test.merge(y_test, how = 'left', on ='MLS')

X1_test = X1_test.set_index('MLS')
y1_test = X1_test['Price']
X1_test.drop('Price', axis = 1, inplace = True)

rf1 = RandomForestRegressor(random_state = 91, n_jobs = -3, max_features = 'sqrt', min_samples_leaf=8, min_samples_split=2, n_estimators = 161, max_depth = 25)
rf1.fit(X1_train, y1_train)
print(f"Training Data Score: {rf1.score(X1_train, y1_train)}")
print(f"Testing Data Score: {rf1.score(X1_test, y1_test)}")
print(scoring_function(y1_test, rf1.predict(X1_test)))

#%%

X2_train = X2_train.reset_index()
X2_train = X2_train.merge(y_train, how = 'left', on ='MLS')

X2_train = X2_train.set_index('MLS')
y2_train = X2_train['Price']
X2_train.drop('Price', axis = 1, inplace = True)


X2_test = X2_test.reset_index()
X2_test = X2_test.merge(y_test, how = 'left', on ='MLS')

X2_test = X2_test.set_index('MLS')
y2_test = X2_test['Price']
X2_test.drop('Price', axis = 1, inplace = True)

rf2 = RandomForestRegressor(random_state = 92, n_jobs = -3, max_features = 'sqrt', min_samples_leaf=8, min_samples_split=2, n_estimators = 262, max_depth = 25)
rf2.fit(X2_train, y2_train)
print(f"Training Data Score: {rf2.score(X2_train, y2_train)}")
print(f"Testing Data Score: {rf2.score(X2_test, y2_test)}")
print(scoring_function(y2_test, rf2.predict(X2_test)))
#%%
X3_train = X3_train.reset_index()
X3_train = X3_train.merge(y_train, how = 'left', on ='MLS')

X3_train = X3_train.set_index('MLS')
y3_train = X3_train['Price']
X3_train.drop('Price', axis = 1, inplace = True)


X3_test = X3_test.reset_index()
X3_test = X3_test.merge(y_test, how = 'left', on ='MLS')

X3_test = X3_test.set_index('MLS')
y3_test = X3_test['Price']
X3_test.drop('Price', axis = 1, inplace = True)

rf3 = RandomForestRegressor(random_state = 93, n_jobs = -3, max_features = 'sqrt', min_samples_leaf=8, min_samples_split=3, n_estimators = 363, max_depth = 35)
rf3.fit(X3_train, y3_train)
print(f"Training Data Score: {rf3.score(X3_train, y3_train)}")
print(f"Testing Data Score: {rf3.score(X3_test, y3_test)}")
print(scoring_function(y3_test, rf3.predict(X3_test)))

#%%
X4_train = X4_train.reset_index()
X4_train = X4_train.merge(y_train, how = 'left', on ='MLS')

X4_train = X4_train.set_index('MLS')
y4_train = X4_train['Price']
X4_train.drop('Price', axis = 1, inplace = True)


X4_test = X4_test.reset_index()
X4_test = X4_test.merge(y_test, how = 'left', on ='MLS')

X4_test = X4_test.set_index('MLS')
y4_test = X4_test['Price']
X4_test.drop('Price', axis = 1, inplace = True)

rf4 = RandomForestRegressor(random_state = 94, n_jobs = -4, max_features = 'sqrt', min_samples_leaf=8, min_samples_split=4, n_estimators = 464, max_depth = 45)
rf4.fit(X4_train, y4_train)
print(f"Training Data Score: {rf4.score(X4_train, y4_train)}")
print(f"Testing Data Score: {rf4.score(X4_test, y4_test)}")
print(scoring_function(y4_test, rf4.predict(X4_test)))
#%%
X5_train = X5_train.reset_index()
X5_train = X5_train.merge(y_train, how = 'left', on ='MLS')

X5_train = X5_train.set_index('MLS')
y5_train = X5_train['Price']
X5_train.drop('Price', axis = 1, inplace = True)


X5_test = X5_test.reset_index()
X5_test = X5_test.merge(y_test, how = 'left', on ='MLS')

X5_test = X5_test.set_index('MLS')
y5_test = X5_test['Price']
X5_test.drop('Price', axis = 1, inplace = True)

rf5 = RandomForestRegressor(random_state = 95, n_jobs = -5, max_features = 'sqrt', min_samples_leaf=8, min_samples_split=5, n_estimators = 565, max_depth = 55)
rf5.fit(X5_train, y5_train)
print(f"Training Data Score: {rf5.score(X5_train, y5_train)}")
print(f"Testing Data Score: {rf5.score(X5_test, y5_test)}")
print(scoring_function(y5_test, rf5.predict(X5_test)))
#%%
X6_train = X6_train.reset_index()
X6_train = X6_train.merge(y_train, how = 'left', on ='MLS')

X6_train = X6_train.set_index('MLS')
y6_train = X6_train['Price']
X6_train.drop('Price', axis = 1, inplace = True)


X6_test = X6_test.reset_index()
X6_test = X6_test.merge(y_test, how = 'left', on ='MLS')

X6_test = X6_test.set_index('MLS')
y6_test = X6_test['Price']
X6_test.drop('Price', axis = 1, inplace = True)

rf6 = RandomForestRegressor(random_state = 96, n_jobs = -6, max_features = 'sqrt', min_samples_leaf=8, min_samples_split=6, n_estimators = 666, max_depth = 66)
rf6.fit(X6_train, y6_train)
print(f"Training Data Score: {rf6.score(X6_train, y6_train)}")
print(f"Testing Data Score: {rf6.score(X6_test, y6_test)}")
print(scoring_function(y6_test, rf6.predict(X6_test)))
#%%
X7_train = X7_train.reset_index()
X7_train = X7_train.merge(y_train, how = 'left', on ='MLS')

X7_train = X7_train.set_index('MLS')
y7_train = X7_train['Price']
X7_train.drop('Price', axis = 1, inplace = True)


X7_test = X7_test.reset_index()
X7_test = X7_test.merge(y_test, how = 'left', on ='MLS')

X7_test = X7_test.set_index('MLS')
y7_test = X7_test['Price']
X7_test.drop('Price', axis = 1, inplace = True)

rf7 = RandomForestRegressor(random_state = 97, n_jobs = -7, max_features = 'sqrt', min_samples_leaf=8, min_samples_split=7, n_estimators = 777, max_depth = 77)
rf7.fit(X7_train, y7_train)
print(f"Training Data Score: {rf7.score(X7_train, y7_train)}")
print(f"Testing Data Score: {rf7.score(X7_test, y7_test)}")
print(scoring_function(y7_test, rf7.predict(X7_test)))
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
