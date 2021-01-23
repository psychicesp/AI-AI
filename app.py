import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory


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
@app.route("/form", methods = ["GET", "POST"])
def form():
    zipcode = request.form.get("zipcode")
    bathrooms = request.form.get("bathrooms")
    halfbaths = request.form.get("halfbaths")
    bedrooms = request.form.get("bedrooms")
    purchaseyear = request.form.get("purchaseyear")
    ageofhome = request.form.get("ageofhome")
    acres = request.form.get("acres")
    housesize =  request.form.get("housesize")
    return render_template('form.html')

# this is how we send data to javascript
@app.route("/api/TBD")
def race_stats():
    

    session = Session(engine)

    

    return jsonify(all_race_stats)

if __name__ == '__main__':
    app.run(debug=True)
