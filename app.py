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
    # con estos prints controlamos lo que se hace internamente
    print(ram_gb,weight,inches)
    print(type(ram_gb))
    # en caso de que no este introducidos los datos que necesita saltara un error
    if ram_gb is None or weight is None or inches is None:
        return render_template('predict.html', title = 'predict', data = False, faltan_datos = True)
    else: # de otra forma, se ejecuta para que salga el valor predicho, hay que hacer unos cambios en las variable spara controlar ese tipo de error al introducir los datos
        ram_gb = float(ram_gb.replace(',', '.'))
        weight = float(weight.replace(',', '.'))
        inches = float(inches.replace(',', '.'))
        prediction = model.predict([[float(ram_gb),float(weight),float(inches)]])
        prediction = round(prediction[0], 2)
        try:# prueba a usar los datos para sacar el RMSE, asi el cliente sabe con que error se le predice el precio
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
    return render_template('predict.html', title = 'predict', data = data) # finalemnte si todo ha ido bien, enseña por pantalla el precio

# Preguntamos el status, para saber si todo esta bien, necesitamos codigo 200
@app.route('/status', methods = ['GET'])
def status():
    data = {
        'titulo': 'status'
    }
    return render_template('status.html', data = data), 200


# si queremos reentrenar el modelo
@app.route('/api/v1/retrain/', methods = ['GET'])
def retrain(): # Ligado al endpoint '/api/v1/retrain/', metodo GET
    csv_path = "tmp/train_new.csv"
    model_path = "tmp/ad_model.pkl"
    if os.path.exists(csv_path):
        data = pd.read_csv(csv_path)

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Price_in_euros']), data['Price_in_euros'], test_size = 0.20, random_state=42)

        model = RandomForestRegressor(max_depth = 5, random_state = 42)
        model.fit(X_train, y_train)

        rmse = round(np.sqrt(mean_squared_error(y_test, model.predict(X_test))),2)
        mape = round(mean_absolute_percentage_error(y_test, model.predict(X_test)),4)

        model.fit(data.drop(columns=['Price_in_euros']), data['Price_in_euros'])
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        data = {
            'respuesta': True,
            'rmse': rmse,
            'mape': mape
        }
        print('todo bien')
        return render_template('retrain.html', data = data) 
    else:
        print('no reentrena')
        data = {
            'respuesta' : False
        }
        return render_template('retrain.html', data= data) 
    

# enseñamos por pantalla las features que tiene nuestro dataset
@app.route('/features', methods = ['GET'])
def features():
    return render_template('features.html', title = 'features')

# nos devuelve al inicio
@app.route('/inicio')
def redireccion():
    return redirect(url_for('index'))

# en caso de error navegando por nuestra web, salta error 404
def error_404(error):
    return render_template('404.html'), 404



if __name__ == '__main__':
    app.register_error_handler(404, error_404)
    app.run(debug=False)