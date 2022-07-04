from msilib.schema import Class
import numpy as np
from collections import Counter


def eculidian_distance(x1, x2):  # calculate eculidian distance
    return np.sqrt(
        np.sum(np.square(x1 - x2) ** 2)  # calculate the sum of the squared differences
    )  # def plot_decision_boundary(self, X_test, y_pred):
    # plt.figure(figsize=(10, 10))


class KNN:  # create a knn classifier
    def __init__(self, k=3):  # initialize the classifier
        self.k = k

    def fit(self, X_train, y_train):  # train the classifier
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):  # predict the test set
        y_pred = np.zeros(len(X))  # initialize the prediction array
        for i in range(len(X)):  # for each test example
            distances = [
                eculidian_distance(x, X[i]) for x in self.X_train
            ]  # calculate the eculidian distance
            nearest_neighbors = np.argsort(distances)[
                : self.k
            ]  # find the k nearest neighbors
            top_k_y = [
                self.y_train[i] for i in nearest_neighbors
            ]  # get the class labels of the k nearest neighbors
            y_pred[i] = Counter(top_k_y).most_common(1)[0][0]  # predict the class label
        return y_pred  # return the prediction array
