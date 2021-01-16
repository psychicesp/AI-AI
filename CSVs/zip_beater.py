#%%
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats as st
# %%

complete_df = pd.read_csv('real_estate_data_complete_ESPcopy.csv')
complete_df = complete_df[complete_df['RetZipCode'].notna()]
complete_df['Zip_Code'] = complete_df['RetZipCode'].apply(lambda x: float(x))
zip_df = pd.read_csv('zipcode_data.csv')
zip_df['Zip_Code'] = zip_df['Zip_Code'].apply(lambda x: float(x))
zip_df['Water_Land_Percent'] = (zip_df['Water Area'] / zip_df['Land Area'])*100
combined_df = complete_df.merge(zip_df, on = 'Zip_Code', how = 'outer')
combined_df['Combined_Baths'] = combined_df['Full_Baths']+(combined_df['Half_Baths']/2)
combined_df.head()
# %%
def num_cleaner(x):
    try:
        x = x.replace('$','')
        x = x.replace(',','')
        x = float(x)
        return x
    except:
        pass
combined_df['Median Household Income'] = combined_df['Median Household Income'].apply(num_cleaner)
combined_df['Median Home Value'] = combined_df['Median Home Value'].apply(num_cleaner)
combined_df['Population Density'] = combined_df['Population Density'].apply(num_cleaner)
combined_df = combined_df[['MLS', 'Price', 'Bedrooms','Age','Square_Footage','Acres', 'Combined_Baths','RetZipCode','Population Density','Median Home Value','Water_Land_Percent', 'Median Household Income']]
combined_df = combined_df.set_index('MLS')
combined_df.dropna()
for index, row in combined_df.iterrows():
    if (row['Age'] >= 4000) or (row['Acres']>5) or (row['Square_Footage']>20000) or (row['Square_Footage']<= 200) or (row['Combined_Baths']<1):
        try:
            combined_df.drop(index = index, inplace = True, axis = 1)
        except:
            pass
    if row['Age'] > 1000:
        try:
            combined_df.loc[index, 'Age'] = 2020-combined_df.loc[index, 'Age']
        except:
            pass
    if row['Age'] > 200:
        try:
            combined_df.drop(index = index, inplace = True, axis = 1)
        except:
            pass
combined_df.to_csv('combined.csv')
# %%
def Scatter_w_Trend(df,x,y, y_limit=None):
    try:
        new_df = df[[x,y]].dropna()
        new_df[x] = new_df[x].astype(float)
        new_df[y] = new_df[y].astype(float)
        new_df.head()
        x_list = new_df[x].tolist()
        y_list = new_df[y].tolist()
        plt.scatter(x_list, y_list, alpha = 0.15, color = 'navy')
        plt.ylim(y_limit)
        m, b, rSquare, pValue, stderr = st.linregress(x_list,y_list)
        line = "y = " + str(round(m,2)) + "x + " + str(round(b,2))
        print(f"The P Value for {x} is {pValue}")
        if pValue < 0.0001:
            pValue = "<0.0001"
        else:
            pValue = round(pValue,4)
        plt.annotate(line, xy = (10,10), fontsize = 12, color="red")
        #plot_location[1] = 0.80*plot_location[1]
        plt.annotate(f"p = {pValue}",xy = (20,20), fontsize = 18, color = 'red' )
        fit_line_y = new_df[x] * m + b
        plt.plot(x_list, fit_line_y, color = 'red')
    except:
        pass
#%%
X_values = combined_df.columns.to_list()
X_values.remove('Price')
X_values.remove('RetZipCode')
# %%
for value in X_values:
    try:
        Scatter_w_Trend(combined_df, value, 'Price', )
        plt.title(f"Price vs {value}")
        plt.xlabel(value)
        plt.ylabel("Price")
        plt.show()
    except:
        print(f"Error with {value}")
# %%
