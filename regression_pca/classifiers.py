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

    def weight_init(self, weight_len):
        """
        Initialize weight to 0
        :param weight_len: size required
        :return: array of zeros
        """
        return np.zeros((weight_len + 1, 1))

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

    def __init__(self, num_classes):
        self.num_classes = num_classes
        pass
    
    def weight_init(self, num_inputs):
        """
        Initialize weight to 0
        :param num_inputs: input size required
        :return: array of zeros
        """
        return np.zeros((num_inputs + 1, self.num_classes))

    def get_one_hot_encoding(self, vec):
        '''
        Converts an array of integers of classes to one hot encoding
        '''
        assert isinstance(vec, np.ndarray)

        # return np.squeeze(np.eye(self.num_classes)[vec.reshape(-1)])
        one_hot = np.zeros((self.num_classes, len(vec)))
        for i in range(len(vec)):
            one_hot[int(vec[i]), i] = 1
        
        assert np.sum(one_hot) == len(vec)
        return one_hot

        
    def get_loss(self, y, t):
        """
        Cross entropy loss
        :param y: predictions
        :param t: targets (0 or 1)
        :return: loss
        """
        assert isinstance(y, np.ndarray)
        assert isinstance(t, np.ndarray)
        target_one_hot = self.get_one_hot_encoding(t)
        assert target_one_hot.shape == y.shape

        loss = -1.0 * np.sum(target_one_hot * np.log(y))
        
        return loss

    def predict(self, weights, data):
        """
        Using logistic function on the linear combination of x and w

        :param weights: d x c,
        :param data: d x N
        :return: prediction vector, N x 1
        """
        assert isinstance(weights, np.ndarray)
        assert isinstance(data, np.ndarray)
        data = data.T
        # print('weights.shape = ', weights.shape)
        # print('data.shape = ', data.shape)

        activation = weights.T @ data # size c x N
        actual_probabilites = self.softmax(activation) # size c x N
        prediction = np.argmax(actual_probabilites, axis=0).reshape(-1,1) # size N
        
        return prediction, actual_probabilites

    def softmax(self, activation):
        '''
        Computes softmax of activation using the softmax trick (https://jamesmccaffrey.wordpress.com/2016/03/04/the-max-trick-when-computing-softmax/)

        :param activation: c x N, where c is num classes and N is num of samples
        "return: prediction
        '''
        max_entry_each_column = np.amax(activation, axis=0) # size N
        numerator = np.exp(activation - max_entry_each_column) #size c x N
        denominator = np.sum(numerator, axis=0) # size c x N
        
        return numerator/denominator

    def get_update(self, y, data, t):
        """
        Get gradient step
        :param y: predictions
        :param t: targets
        :param data: images
        :return: weight update
        """
        # using batch gradient descent
        data = data.T
        target_one_hot = self.get_one_hot_encoding(t)
        diff = target_one_hot - y
        # print('diff.shape = ', diff.shape)
        # print('data.shape = ', data.shape)
        update = data @ diff.T
        return -update