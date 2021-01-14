#%%
from splinter import Browser
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
executable_path = {'executable_path': '../chromedriver.exe'}
def get_zips():
    complete_df = pd.read_csv("real_estate_data_complete.csv")
    zip_codes = complete_df['RetZipCode'].dropna().to_list()
    global unique_zips
    unique_zips = []
    for zip_code in zip_codes:
        if (zip_code not in unique_zips):
            unique_zips.append(zip_code)
    print(len(unique_zips))
#%%
def scrape_zips():
    browser = Browser('chrome', **executable_path, headless=False)
    global zip_list
    zip_list = []
    for zip in unique_zips:
        try:
            new_dict = {
                'Zip_Code': zip
            }
            url = f"http://unitedstateszipcodes.org/{str(int(round(zip,0)))}"
            browser.visit(url)
            time.sleep(2)
            soup = BeautifulSoup(browser.html, 'html.parser')
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')[:4]
                for row in rows:
                    try:
                        new_dict[row.find('th').text] = row.find('td').text
                    except:
                        pass    
            zip_list.append(new_dict)
        except:
            browser.quit()
            browser = Browser('chrome', **executable_path, headless=False)
            time.sleep(5)
            new_dict = {
                'Zip_Code': zip
            }
            url = f"http://unitedstateszipcodes.org/{str(int(round(zip,0)))}"
            browser.visit(url)
            time.sleep(2)
            soup = BeautifulSoup(browser.html, 'html.parser')
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')[:5]
                for row in rows:
                    try:
                        new_dict[row.find('th').text] = row.find('td').text
                    except:
                        pass    
            zip_list.append(new_dict)
# %%
zip_df = pd.DataFrame(zip_list)
zip_df.to_csv('zipcode_data.csv')