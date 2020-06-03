import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

sample_size = 500
x = np.random.normal(3, 5, sample_size)
y = x * 10 + np.random.normal(10, 4, sample_size)
model = LinearRegression()
model.fit(x.reshape(sample_size, 1), y.reshape(sample_size, 1))
# print(model.coef_)
predicted_y = model.predict(x.reshape(sample_size, 1))
print('r2 score: ', r2_score(y, predicted_y))
plt.scatter(y, x)
plt.show()
plt.savefig('nann.jpg')