#%%
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

matplotlib.text.Text(rotation = 'vertical')
miss_zip = pd.read_csv('MissingZips.csv')
miss_zip.head()
#%%
group_zip = miss_zip.groupby('RetCity').agg({
    'MLS':'count'
}).reset_index()
group_zip.head(25)
# %%
plt.bar(group_zip['RetCity'], group_zip['MLS'])
# %%
