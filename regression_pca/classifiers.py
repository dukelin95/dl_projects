import numpy as np

class Classifier():
    def predict(self):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def get_update(self):
        raise NotImplementedError


class LogisticRegression(Classifier):

    def __init__(self):
        pass

    def get_loss(self, y, t):
        """
        Cross entropy loss
        :param y: predictions
        :param t: targets (0 or 1)
        :return: loss
        """
        temp1 = (t * np.log(y))
        temp2 = ((1-t) * np.log(1-y))
        loss = -np.sum(temp1 + temp2)
        return loss

    def predict(self, weights, data):
        """
        Using logistic function on the linear combination of x and w

        :param weights: Weightsize x 1,
        :param data: N x Weightsize
        :return: prediction vector, N x 1
        """
        lc = np.matmul(data, weights)
        actual_val = 1.0 / (1.0 + np.exp(lc))
        pred = actual_val > 0.5
        return pred.astype(float).reshape(len(pred), -1), actual_val

    def get_update(self, y, data, t):
        """
        Get gradient step
        :param y: predictions
        :param t: targets
        :param data: images
        :return: weight update
        """
        # using batch gradient descent
        diff = t - y
        update = np.matmul(data.T, diff)
        return update


class SoftmaxRegression(Classifier):

    def __init__(self):
        pass
