from flask import Flask, render_template, url_for, redirect, jsonify, request
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

app = Flask(__name__)

# primera ventana que veremos, con información básica
@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')


@app.route('/api/v1/predict', methods = ['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
    with open('ad_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Convertimos todos los argumentos a un diccionario con las claves en minusculas para controlar los errores de escritura manual
    args = {k.lower(): v for k, v in request.args.items()}
    ram_gb = args.get('ram_gb', None)
    weight = args.get('weight_kg', None)
    inches = args.get('inches', None)

    print(ram_gb,weight,inches)
    print(type(ram_gb))

    if ram_gb is None or weight is None or inches is None:
        return render_template('predict.html', title = 'predict', data = False, faltan_datos = True)
    else:
        ram_gb = float(ram_gb.replace(',', '.'))
        weight = float(weight.replace(',', '.'))
        inches = float(inches.replace(',', '.'))
        prediction = model.predict([[float(ram_gb),float(weight),float(inches)]])
        prediction = round(prediction[0], 2)
        try:
            csv_path = 'data/trainable.csv'   
            data = pd.read_csv(csv_path)

            X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Price_in_euros']), data['Price_in_euros'], test_size = 0.20, random_state=42)

            model = RandomForestRegressor(max_depth = 5, random_state = 42)
            model.fit(X_train, y_train)

            rmse = round(np.sqrt(mean_squared_error(y_test, model.predict(X_test))),2)
        except Exception as e:
            print(f'Error {e}')
        data = {
            'resultado':prediction,
            'rmse':rmse
        }
    return render_template('predict.html', title = 'predict', data = data)


@app.route('/status', methods = ['GET'])
def status():
    data = {
        'titulo': 'status'
    }
    return render_template('status.html', data = data), 200



@app.route('/api/v1/retrain/', methods = ['GET'])
def retrain(): # Ligado al endpoint '/api/v1/retrain/', metodo GET
    csv_path = "tmp/train_new.csv"
    model_path = "tmp/ad_model.pkl"
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Price_in_euros']), data['Price_in_euros'], test_size = 0.20, random_state=42)

        model = RandomForestRegressor(max_depth = 5, random_state = 42)
        model.fit(X_train, y_train)

        rmse = round(np.sqrt(mean_squared_error(y_test, model.predict(X_test))),4)
        mape = round(mean_absolute_percentage_error(y_test, model.predict(X_test)),4)

        model.fit(data.drop(columns=['Price_in_euros']), data['Price_in_euros'])
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        return render_template('m_retrain.html') # FALTA PONER EL HTML            #f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(mape)}"
    else:
        return render_template('m_no_retrain.html') # FALTA PONER EL HTML           #f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"
    


@app.route('/features', methods = ['GET'])
def features():
    return render_template('features.html', title = 'features')


@app.route('/inicio')
def redireccion():
    return redirect(url_for('index'))


def error_404(error):
    return render_template('404.html'), 404



if __name__ == '__main__':
    app.register_error_handler(404, error_404)
    app.run(debug=True)  