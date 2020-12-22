#%%
import pandas as pd

#%%
dfs = []

for x in range(1,93):
    df = pd.read_csv(f"Single Line ({x}).csv")
    dfs.append(df)

dfs[0].head()
# %%
