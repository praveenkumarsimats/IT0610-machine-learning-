"""
Lab 22 — K-Nearest Neighbour Regression
Aim: To implement KNN regression to predict a continuous target value based on the k
nearest training instances.
Algorithm: For a query instance, find the k nearest training points and predict the
(optionally distance-weighted) average of their target values.

NOTE: fetch_california_housing() downloads the dataset from the internet the first
time it is run. If you have no internet access, download the dataset manually or use
an offline copy before running this script.
"""

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target,
    test_size=0.2, random_state=1)

model = KNeighborsRegressor(n_neighbors=7).fit(X_train, y_train)
pred = model.predict(X_test)

print("RMSE:", mean_squared_error(y_test, pred) ** 0.5)
print("R^2 score:", r2_score(y_test, pred))

# Result: KNN regression predicted housing prices with a reasonable R^2 score,
# demonstrating instance-based regression.
