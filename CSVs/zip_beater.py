#%%
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from scipy import stats as st
from datetime import datetime as dt
from datetime import timedelta
date_format = "%m/%d/%Y"
# %%

complete_df = pd.read_csv('real_estate_data_complete_ESPcopy.csv')
complete_df['Zip_Code'] = complete_df['RetZipCode'].apply(lambda x: float(x))
zip_df = pd.read_csv('zipcode_data.csv')
zip_df['Zip_Code'] = zip_df['Zip_Code'].apply(lambda x: float(x))
zip_df['Water_Land_Percent'] = (zip_df['Water Area'] / zip_df['Land Area'])*100
combined_df = complete_df.merge(zip_df, on = 'Zip_Code', how = 'outer')
combined_df.set_index('MLS')
combined_df = combined_df[combined_df['P_Type']=='RESI']
combined_df = combined_df[combined_df['St']=='S']
combined_df['Combined_Baths'] = combined_df['Full_Baths']+(combined_df['Half_Baths']/2)
missing_zips = pd.read_csv('MissingZips.csv').set_index('MLS')
missing_zips.dropna(subset = ['RetZipCode'], inplace = True)
for index, row in missing_zips.iterrows():
    combined_df.loc[index, 'RetZipCode'] = row['RetZipCode']
# %%
def num_cleaner(x):
    try:
        x = x.replace('$','')
        x = x.replace(',','')
        x = float(x)
        return x
    except:
        pass
combined_df = combined_df[combined_df['Status_Date'].notna()]
combined_df['Year_sold'] = combined_df['Status_Date'].apply(lambda x: int(dt.strptime(str(x), date_format).year))
combined_df['Month_sold'] = combined_df['Status_Date'].apply(lambda x: int(dt.strptime(str(x), date_format).month))
combined_df['Days_since'] = combined_df['Status_Date'].apply(lambda x: (dt.strptime(str(x), date_format)-dt.strptime("1/1/2010", date_format)).days)
combined_df['Status_Date'] = combined_df['Status_Date'].apply(lambda x: dt.strptime(str(x), date_format))
combined_df['Years_since'] = combined_df['Year_sold']-2009
combined_df['Median Household Income'] = combined_df['Median Household Income'].apply(num_cleaner)
combined_df['Median Home Value'] = combined_df['Median Home Value'].apply(num_cleaner)
combined_df['CDOM'] = combined_df['CDOM'].apply(num_cleaner)
combined_df['Population Density'] = combined_df['Population Density'].apply(num_cleaner)
combined_df = combined_df[['MLS', 'Price', 'Bedrooms','Age','Square_Footage','Acres', 'Combined_Baths','RetZipCode','Population Density','Median Home Value','Water_Land_Percent', 'Median Household Income', 'Status_Date', 'Days_since','Year_sold','Month_sold', 'Years_since', 'CDOM']]
combined_df = combined_df.set_index('MLS')
combined_df.dropna()
for index, row in combined_df.iterrows():
    if (row['Age'] >= 4000) or (row['Acres']>5) or (row['Square_Footage']>20000) or (row['Square_Footage']<= 200) or (row['Combined_Baths']<1)or (row['Bedrooms']<1)or(row['Combined_Baths']>=12) or (row['Bedrooms']>=15)or (row['Price']<25000)or (row['Price']>750000):
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
combined_df.dropna(inplace=True)
combined_df.reset_index()
combined_df = combined_df.groupby('MLS').first()
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
X_values.remove('Status_Date')
X_values.remove('Year_sold')
X_values.remove('Month_sold')
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
zip_controlled_df = combined_df[['Price', 'Population Density','Median Home Value','Water_Land_Percent', 'Median Household Income']]
zip_controlled_df['Price_Transformed'] = zip_controlled_df['Price']/zip_controlled_df['Median Household Income']
X_values = zip_controlled_df.columns.to_list()
X_values.remove('Price')
X_values.remove('Price_Transformed')
# %%
for value in X_values:
    try:
        Scatter_w_Trend(zip_controlled_df, value, 'Price_Transformed', )
        plt.title(f"Price vs {value}")
        plt.xlabel(value)
        plt.ylabel('Price_Transformed')
        plt.show()
    except:
        print(f"Error with {value}")
# %%
