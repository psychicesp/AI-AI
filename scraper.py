#%%
from splinter import Browser
from bs4 import BeautifulSoup as bs
import requests as req
import time
import pandas as pd
import math
#%%
executable_path = {'executable_path': 'chromedriver.exe'}
browser = Browser('chrome', **executable_path, headless=False)
news_url="https://mars.nasa.gov/news/?page=7&per_page=40&order=publish_date+desc%2Ccreated_at+desc&search=&category=19%2C165%2C184%2C204&blank_scope=Latest"
browser.visit("https://matrix.marismatrix.com/Matrix/Results.aspx?c=AAEAAAD*****AQAAAAAAAAARAQAAAFUAAAAGAgAAAAQ1Njg4BgMAAAABMgYEAAAAATEKBgUAAAABNQYGAAAAATUNAgYHAAAAAjExDQ0GCAAAAAI1MAYJAAAAATANLwYKAAAAATENCgYLAAAACMKzwrrDmMKZDQIL")  
#%%
def scrape(n = 2000):
    browser.click_link_by_id("m_lnkCheckPageLink")
    for i in range(math.ceil(n/50)-1):
        time.sleep(0.4)
        browser.click_link_by_id("m_DisplayCore_dpy2")
        time.sleep(0.8)
        browser.click_link_by_id("m_lnkCheckPageLink")
    browser.click_link_by_partial_text("Export")
    browser.click_link_by_id("m_btnExport")
# %%
