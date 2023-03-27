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
feature_names = np.array(features.columns)
label = data[label_column]

X = np.array(features)
y = np.array(label)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)
linear = linear_model.LinearRegression()
linear.fit(X_train, y_train)
print(linear.score(X_test, y_test))

for index, feature in enumerate(feature_names):
    plot_linear(X[:, index], y, feature)
