import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import os

os.chdir(os.path.dirname(__file__))


df = pd.read_csv('./data/train.csv')

df.index.name = None

target = 'Price_in_euros'

df['ram_gb'] = df['Ram'].str.replace('GB', '').astype(float)
df['weight_kg'] = df['Weight'].str.replace('kg', '').astype(float)

features_num = ['Ram_GB', 'Weight_kg', 'Inches']


X = df[features_num].copy()
y = df['Price_in_euros'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

base_model = RandomForestRegressor(max_depth = 5, random_state = 42)
base_model.fit(X_train, y_train)

y_pred = base_model.predict (X_test)

print(f"RMSE:{root_mean_squared_error(y_test, y_pred)}")


predictions_submit = base_model.predict(df[features_num], df[target])
predictions_submit

with open('ad_model.pkl', 'wb') as f:
    pickle.dump(predictions_submit, f)