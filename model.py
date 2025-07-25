import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os

os.chdir(os.path.dirname(__file__))

# cargamos todo y seleccionamos las features que necesitaremos
df = pd.read_csv('./data/train.csv')
df.index.name = None

target = 'Price_in_euros'

df['ram_gb'] = df['Ram'].str.replace('GB', '').astype(float)
df['weight_kg'] = df['Weight'].str.replace('kg', '').astype(float)
df['inches'] = df['Inches']
features_num = ['ram_gb', 'weight_kg', 'inches']


X = df[features_num].copy()
y = df['Price_in_euros'].copy()

# Hacemos test y test para ver como funciona nuestro modelo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

base_model = RandomForestRegressor(max_depth = 5, random_state = 42)
base_model.fit(X_train, y_train)

y_pred = base_model.predict(X_test)

print(f"RMSE:{root_mean_squared_error(y_test, y_pred)}")


predictions_submit = base_model.predict(df[features_num])
predictions_submit

# una vez comprobado, entrenamos el modelo ahora con todos los datos para ponerlo ya a producción y lo guardamos
modelo = base_model.fit(X,y)

with open('ad_model.pkl', 'wb') as f:
    pickle.dump(modelo, f)