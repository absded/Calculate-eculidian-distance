from tkinter import Y
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()  # load iris data
X, y = iris.data, iris.target  # X: data, y: target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)  # split data into training and test sets


# print(X_train.shape)
# print(X_train[0])

# print(y_train.shape)
# print(y_train)

# plt.figure(figsize=(10, 10))
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cmap)
# plt.show()
from knn import KNN  # import knn.py

clf = KNN(k=7)  # create a knn classifier
clf.fit(X_train, y_train)  # train the classifier
predictions = clf.predict(X_test)  # predict the test set

acc = np.sum(predictions == y_test) / len(y_test)  # calculate accuracy

print(acc)
