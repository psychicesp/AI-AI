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

# this is how we send data to javascript
@app.route("/api/TBD")
def race_stats():
    

    session = Session(engine)

    

    return jsonify(all_race_stats)

if __name__ == '__main__':
    app.run(debug=True)
