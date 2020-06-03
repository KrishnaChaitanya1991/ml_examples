import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score, mean_squared_error

np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

# transform the input into polynomial features
pageSpeedsPoly = PolynomialFeatures(degree=5).fit_transform(pageSpeeds.reshape(1000, 1)) 
pageSpeedsPoly

model = Ridge(alpha=0.0)
model.fit(pageSpeedsPoly, purchaseAmount)
predict_amounts = model.predict(pageSpeedsPoly)
# r2_score the higher the better, max 1, min 0
r2_score_1 = r2_score(purchaseAmount, predict_amounts)
# mean squared error is not normalised
mse = mean_squared_error(purchaseAmount, predict_amounts)
print('r2_accuracy: ', r2_score_1)
print('mse: ', mse)

print("\n Using RidgeCV \n")
# we can use ridgecv to test with multiple alphas
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds

# transform the input into polynomial features
pageSpeedsPoly = PolynomialFeatures(degree=5).fit_transform(pageSpeeds.reshape(1000, 1)) 
pageSpeedsPoly
model2 = RidgeCV(alphas=np.logspace(-50, 10, 21), normalize=True)
model2.fit(pageSpeedsPoly, purchaseAmount)
predict_amounts = model2.predict(pageSpeedsPoly)
# r2_score the higher the better, max 1, min 0
r2_score_cal = r2_score(purchaseAmount, predict_amounts)
# mean squared error is not normalised
mse = mean_squared_error(purchaseAmount, predict_amounts)
print('r2_accuracy: ', r2_score_cal)
print('mse: ', mse)
print('best alpha; ', model2.alpha_)