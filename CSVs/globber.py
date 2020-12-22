#%%
import pandas as pd

#%%
x = 1
all_data = pd.read_csv(f"{x}.csv")
for x in range(2,92):
    print(x)
    df = pd.read_csv(f"{x}.csv")
    all_data = pd.concat([all_data, df])

all_data
# %%

all_data = all_data.drop_duplicates()

# %%
print(all_data['MLS'].size)
all_data = all_data.dropna()
print(all_data['MLS'].size)

all_data.set_index('MLS').to_csv('real_estate_data.csv')


# %%
def price_cleaner(x):
    try:
        x=x.replace('$','')
        x=x.replace(',','')
        x = int(x)
        return x
    except:
        print('Strange error here')

# def bath_getter(x):
#     try:
#         x=x.replace('(','')
#         x=x.replace(')','')
#         x=x.split(' ')
#         print(x)
#         x = x[:2]
#         total = int(x[0])
#         full = int(x[1])
#         half = int(x[2])
#         return total, full, half
#     except:
#         print('Strange error here')
#%%
all_data['Total_Baths'] = 0
all_data['Full_Baths'] = 0 
all_data['Half_Baths'] = 0 

for index, row in all_data.head(5).iterrows():
    x= row['Bth'].replace('(','')
    x= x.replace(')','')
    x= x.split(' ')
    print(x)
    try:
        all_data.iloc(index, 'Total_Baths') = x[0]
        all_data.iloc(index, 'Full_Baths') = x[1]
        all_data.iloc(index, 'Half_Baths') = x[2]
    except:
        pass
all_data
# %%
