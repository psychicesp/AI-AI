# %%
from splinter import Browser
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
executable_path = {'executable_path': '../chromedriver.exe'}
complete_data = pd.read_csv('real_estate_data_complete_ESPcopy.csv')

# for index, row in complete_data.iterrows():
#     if pd.isnull(row['RetZipCode']):
#%%
def scrape_zips():
    browser = Browser('chrome', **executable_path, headless=False)
    url = "https://www.realtor.com/realestateandhomes-detail/M8898102848"
    browser.visit(url)
    browser.find_by_id('autocomplete-input').fill("1947 Damato Ct")
    browser.find_elements_by_css_selector("aria-label=Search").click()
    time.sleep(2)
    soup = BeautifulSoup(browser.html, 'html.parser')