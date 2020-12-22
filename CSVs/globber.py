#%%
import pandas as pd

#%%
x = 1
all_data = pd.read_csv(f"{x}.csv")
all_data['Total_Baths'] = 0
all_data['Full_Baths'] = 0 
all_data['Half_Baths'] = 0 

#%%
for index, row in all_data.iterrows():
    x= row['Bth'].replace('(','')
    x= x.replace(')','')
    x= x.split(' ')
    try:
        all_data.loc[index, 'Total_Baths'] = x[0]
        all_data.loc[index, 'Full_Baths'] = x[1]
        all_data.loc[index, 'Half_Baths'] = x[2]
    except:
        pass
for x in range(2,92):
    print(x)
    df = pd.read_csv(f"{x}.csv")
    df['Total_Baths'] = 0
    df['Full_Baths'] = 0 
    df['Half_Baths'] = 0 
    for index, row in df.iterrows():
        try:
            y= row['Bth'].replace('(','')
            y= y.replace(')','')
            y= y.split(' ')
            df.loc[index, 'Total_Baths'] = y[0]
            df.loc[index, 'Full_Baths'] = y[1]
            df.loc[index, 'Half_Baths'] = y[2]
        except:
            df.drop(index, inplace=True)
            print(f"error in df#{x}, row {index} dropped")
    all_data = pd.concat([all_data, df])

all_data
# %%
print(f"Size with duplicates: {all_data['MLS'].size}")
all_data = all_data.drop_duplicates()
print(f"Size without duplicates but with missing values: {all_data['MLS'].size}")

all_data = all_data.dropna()
print(f"Size without missing values or duplicates: {all_data['MLS'].size}")


# %%
def price_cleaner(x):
    try:
        x=x.replace('$','')
        x=x.replace(',','')
        x = int(x)
        return x
    except:
        print('Strange error here')

all_data['Price']=all_data['Price'].apply(price_cleaner)
all_data
#%%
all_data['Square Footage'] = all_data['Square Footage'].apply(lambda x: x.replace(',',''))
all_data.rename(columns = {
    'Status Date':'Status_Date',
    'P Type':'P_Type',
    'Municipality Township':'Municipality',
    'Square Footage': 'Square_Footage',
    'Listing Office ID':'Listing_Office_ID'
}, inplace = True)
all_data
#%%
all_data.set_index('MLS').to_csv('real_estate_data.csv')


# %%
