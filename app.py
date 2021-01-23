#%%
import numpy as np
import pickle
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

with open('ML_Models/pickle_jar/rf_model.pickle', 'rb') as dill:
    rf_model = pickle.load(dill)

zip_df = pd.read_csv('zipcode_data.csv')
zip_df = zip_df.set_index('Zip_Code')



#%%
#################################################
# Database Setup
#################################################
#Need to add database engine
#engine = create_engine("sqlite:///")

# reflect an existing database into a new model
# Base = automap_base()
# # reflect the tables
# Base.prepare(engine, reflect=True)



#################################################
# Flask Setup
#################################################
app = Flask(__name__)


#################################################
# Flask Routes
#################################################
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/index.html")
def home2():
    return render_template('index.html')

@app.route('/Images/<path:path>')
def send_images(path):
    return send_from_directory('Images', path)

@app.route('/Logos/<path:path>')
def send_logo(path):
    return send_from_directory('Logos', path)

@app.route("/Market_Analytics.html")
def Market_Analytics():
    return render_template('Market_Analytics.html')

@app.route("/StLouis_Overview.html")
def StLouis_Overview():
    return render_template('StLouis_Overview.html')

@app.route("/About_Us.html")
def About_Us():
    return render_template('About_Us.html')

@app.route("/Predictive_Analysis.html")
def Predictive_Analysis():
    return render_template('Predictive_Analysis.html')

# NEW!!!!!! Just following the video
@app.route("/form", methods = ["POST"])
def form():
    zipcode = int(request.form.get("zipcode"))
    pop_dense = zip_df.loc[zipcode, "Population Density"]
    med_value = zip_df.loc[zipcode, "Median Home Value"]
    med_income = zip_df.loc[zipcode, "Median Household Income"]
    w_l_ratio = zip_df.loc[zipcode, "Water_Land_Percent"]
    bathrooms = int(request.form.get("bathrooms"))
    halfbaths = int(request.form.get("halfbaths"))
    baths = bathrooms +(halfbaths/2)
    bedrooms = int(request.form.get("bedrooms"))
    ageofhome = int(request.form.get("ageofhome"))
    acres = float(request.form.get("acres"))
    housesize =  int(request.form.get("housesize"))
    house_info = (bedrooms, ageofhome, housesize, acres, baths, pop_dense, med_value, w_l_ratio, med_income, 0, 25)
    house_info = np.reshape(house_info,(1,-1))
    estimate = '${:,.2f}'.format(round(rf_model.predict(house_info)[0],2))
# Bedrooms,	Age, Square_Footage, Acres, Combined_Baths, Population Density, Median Home Value, Water_Land_Percent, Median Household Income, Years_since, CDOM
    return render_template('Predictive_Analysis.html', estimate = estimate)

# this is how we send data to javascript
@app.route("/api/TBD")
def race_stats():
    

    session = Session(engine)

    

    return jsonify(all_race_stats)

if __name__ == '__main__':
    app.run(debug=True)
