import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()
model.fit(x, y)

print(model.coef_)
print(model.intercept_)

y_pred = model.predict(x)
print(y_pred)

r_sq = model.score(x, y)
print(r_sq)

ssr = sum(np.square(y_pred - y))
print(ssr)