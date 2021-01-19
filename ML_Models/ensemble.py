#%%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV
#%%
with open('pickle_jar/split_data.pickle', 'rb') as dill:
    X_train, X_hp_train, X_test, y_train,y_hp_train, y_test = pickle.load(dill)
with open('pickle_jar/knr_model.pickle', 'rb') as gherkin:
    knr_model = pickle.load(gherkin)
with open('pickle_jar/rf_model.pickle', 'rb') as garlic:
    rf_model = pickle.load(garlic)

#%%
ensemble_model = VotingRegressor(estimators = [('KNR', knr_model),('forest', rf_model)], n_jobs = 1)
ensemble_model = ensemble_model.fit(X_train,y_train)


print(f"Training Data Score: {ensemble_model.score(X_train, y_train)}")
print(f"Testing Data Score: {ensemble_model.score(X_test, y_test)}")

#%%
with open('pickle_jar/ensemble_model.pickle', 'wb') as hamger_slices:
    pickle.dump(ensemble_model,hamger_slices)


# %%
