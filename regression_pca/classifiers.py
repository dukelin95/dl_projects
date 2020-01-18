import numpy as np

class Classifier():
    def predict(self):
        raise NotImplementedError

    def get_update(self):
        raise NotImplementedError


class LogisticRegression(Classifier):

    def __init__(self, method='batch'):
        self.method = method

    def predict(self, weights, data):
        """
        Using logistic function on the linear combination of x and w

        :param weights: Weightsize x 1,
        :param data: N x Weightsize
        :return: prediction vector, N x 1
        """
        lc = np.matmul(data, weights)
        return np.exp(lc)

    def get_update(self, diff, data):
        """

        :param diff: N x 1,
        :param data: N x Weightsize
        :return: weight update vector, Weightsize x 1
        """
        # using batch gradient descent
        update = np.matmul(data.T, diff)
        return update


class SoftmaxRegression(Classifier):

    def __init__(self, method):
        self.method = method
