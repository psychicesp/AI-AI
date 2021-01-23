import numpy as np

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func
import pandas as pd
from flask import Flask, jsonify, render_template


#################################################
# Database Setup
#################################################
#Need to add database engine
#engine = create_engine("sqlite:///")

# reflect an existing database into a new model
Base = automap_base()
# reflect the tables
Base.prepare(engine, reflect=True)



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

@app.route("/Predictive_Analysis")
def Predictive_Analysis():
    return render_template('Predictive_Analysis.html')

#NEW!!!!!! Just following the video
# @app.route("/form", methods = ["POST"])
# def form():
      zipcode = request.form.get("zipcode")
      bathrooms = request.form.get("bathrooms")
      halfbaths = request.form.get("halfbaths")
      bedrooms = request.form.get("bedrooms")
      purchaseyear = request.form.get("purchaseyear")
      ageofhome = request.form.get("ageofhome")
      acres = request.form.get("acres")
      housesize =  request.form.get("housesize")
#     return render_template('form.html')

# this is how we send data to javascript
@app.route("/api/TBD")
def race_stats():
    

    session = Session(engine)

    

    return jsonify(all_race_stats)

if __name__ == '__main__':
    app.run(debug=True)
