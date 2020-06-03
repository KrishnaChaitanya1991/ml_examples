import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_excel('http://cdn.sundog-soft.com/Udemy/DataScience/cars.xls')

print(df.head())
print(df[['Price', 'Make']].groupby(['Make']).mean())
print('columns: ', df.columns)

print('Make details: ', df['Make'].describe())

X = df[['Mileage', 'Make', 'Model', 'Trim', 'Type', 'Cylinder',
       'Liter', 'Doors', 'Cruise', 'Sound', 'Leather']]
y = df['Price']


X = pd.get_dummies(X, columns=["Make", "Model", "Trim", "Type"])
print('X: ', X.head())
print('Y: ', y.head())

model = RidgeCV(alphas=np.logspace(-10, 10, 100), normalize=True)
model.fit(X, y)
predict_values = model.predict(X)
r2_score_cal = r2_score(y, predict_values)
print('r2 score: ', r2_score_cal)

print(predict_values[0:5], y[0:5])
