import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt


def plot_linear(x, y, title):
    plt.scatter(x, y)
    plt.title(title)
    plt.show()


data = pd.read_csv("student-mat.csv", sep=";")
label_column = "G3"
features = data.select_dtypes(include=["int64"]).drop([label_column], axis=1)
constrained_features = data[["G1", "G2"]]
feature_names = np.array(features.columns)
label = data[label_column]

linear = linear_model.LinearRegression()
number_of_iterations = 10000

accuracy_for_all = np.zeros(number_of_iterations)
accuracy_for_limited = np.zeros(number_of_iterations)

for X in zip([np.array(features), np.array(constrained_features)], [accuracy_for_all, accuracy_for_limited]):
    for iteration in range(number_of_iterations):
        y = np.array(label)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X[0], y, test_size=0.2)
        linear.fit(X_train, y_train)
        X[1][iteration] += linear.score(X_test, y_test)

print(np.mean(accuracy_for_all))
print(np.mean(accuracy_for_limited))
