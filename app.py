from flask import Flask, render_template, url_for, redirect, jsonify, request
import pickle
import warnings
warnings.filterwarnings('ignore')
app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    data = {
        'titulo': 'inicio'
    }
    return render_template('index.html', data=data)


@app.route('/api/v1/predict', methods = ['GET'])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
    with open('ad_model.pkl', 'rb') as f:
        model = pickle.load(f)
    # Convertir todos los argumentos a un diccionario con claves en minúsculas
    args = {k.lower(): v for k, v in request.args.items()}
    ram_gb = args.get('ram_gb', None)
    weight = args.get('weight_kg', None)
    inches = args.get('inches', None)

    print(ram_gb,weight,inches)
    print(type(ram_gb))


    if ram_gb is None or weight is None or inches is None:
        return "Args empty, not enough data to predict"
    else:
        ram_gb = float(ram_gb.replace(',', '.'))
        weight = float(weight.replace(',', '.'))
        inches = float(inches.replace(',', '.'))
        prediction = model.predict([[float(ram_gb),float(weight),float(inches)]])
        data = {
            'resultado':prediction[0]
        }
    return render_template('predict.html', title = 'predict', data = data) #jsonify({'predictions': prediction[0]})


@app.route('/status', methods = ['GET'])
def status():
    data = {
        'titulo': 'status'
    }
    return render_template('status.html', data = data), 200


# RETRAIN?????????????


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