from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)
app.config["DEBUG"] = True



# Enruta la landing page (endpoint /)
@app.route('/', methods=['GET'])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return "Bienvenido a mi API del modelo advertising"

# Enruta la funcion al endpoint /api/v1/predict
@app.route('/api/v1/predict', methods=['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
    with open('ad_model.pkl', 'rb') as f:
        model = pickle.load(f)

    ram_gb = request.args.get('Ram_GB', None)
    weight = request.args.get('Weight_kg', None)
    inches = request.args.get('Inches', None)

    print(ram_gb,weight,inches)
    print(type(ram_gb))

    if ram_gb is None or weight is None or inches is None:
        return "Args empty, not enough data to predict"
    else:
        prediction = model.predict([[float(ram_gb),float(weight),float(inches)]])
    
    return jsonify({'predictions': prediction[0]})

# Enruta la funcion al endpoint /api/v1/retrain


if __name__ == '__main__':
    app.run(debug=True)
