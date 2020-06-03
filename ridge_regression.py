import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston, fetch_california_housing

# load hosuing price data set
x, y = fetch_california_housing(return_X_y=True)
print('x shape: ', x.shape, ' y shape: ', y.shape)
print('sample x: ', x[0:2])
print('sample y: ', y[0:2])

# build model using ridge_cv
# ridge_cv - linear regression with regularization

model = RidgeCV(alphas=(np.logspace(-10, 10, 30)), normalize=True)

model.fit(x, y)
print('scor: ', model.score(x, y))
print('r2 score: ', r2_score(y, model.predict(x)))
print('alpha: ', model.alpha_)
print('coeff: ', model.coef_)