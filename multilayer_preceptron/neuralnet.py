################################################################################
# CSE 253: Programming Assignment 2
# Code snippet by Manjot Bilkhu
# Winter 2020
################################################################################
# We've provided you with the dataset in PA2.zip
################################################################################
# To install PyYaml, refer to the instructions for your system:
# https://pyyaml.org/wiki/PyYAMLDocumentation
################################################################################
# If you don't have NumPy installed, please use the instructions here:
# https://scipy.org/install.html
################################################################################

import os, gzip
import yaml
import numpy as np

import matplotlib.pyplot as plt


def load_config(path):
    """
    Load the configuration from path
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)


def normalize_data(img):
    """
    Normalizes input img by subtracting mean and dividing by standard deviation
    :param img: (Nxd) size numpy array where N is number of images and d is its dimension
    """
    
    assert isinstance(img, np.ndarray)
    assert img.ndim == 2, 'Input array needs to be two dimensional'
    if img.shape[0] < img.shape[1]:
        print('WARNING: dim1 < dim2, input might be formatted wrong. dim1 = number of examples, dim2 = dimension')
    
    mean = np.mean(img, axis=0)
    std_dev = np.std(img, axis=0)
    img_normalized = np.divide((img - mean), std_dev)
    
    return img_normalized


def one_hot_encoding(labels, num_classes=10):
    """
    Encode labels using one hot encoding and return them.
    :param labels: (N,) size numpy array
    """
    assert isinstance(labels, np.ndarray)
    assert labels.ndim == 1, 'Input array needs to be one dimensional'
    
    N = labels.shape[0] #number of examples
    one_hot = np.zeros((N, num_classes)) #initializing one_hot_encoding matrix with all zeros
    for i in range(N):
        one_hot[i, int(labels[i])] = 1
    
    assert np.sum(one_hot) == N, 'One hot encoding failed. CHECK!'
    
    return one_hot


def load_data(path, mode='train'):
    """
    Load Fashion MNIST data.
    Use mode='train' for train and mode='t10k' for test.
    """

    labels_path = os.path.join(path, f'{mode}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{mode}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    normalized_images = normalize_data(images)
    one_hot_labels    = one_hot_encoding(labels, num_classes=len(set(labels)))

    return normalized_images, one_hot_labels


def softmax(x):
    """
    Implement the softmax function here.
    Remember to take care of the overflow condition.

    :param x: (N x d) matrix where N is number of examples, d is dimension
    """
    x = x.T # x = (dxN)
    max_entry_each_column = np.amax(activation, axis=0) # size N
    numerator = np.exp(activation - max_entry_each_column) #size c x N (uses broadcasting)
    denominator = np.sum(numerator, axis=0) # size c x N
    probabilities = numerator/denominator # (uses broadcasting)
    
    N = len(max_entry_each_column)
    assert np.abs(np.sum(probabilities) - N) < 1e-4, 'probabilities sum to {0}, not 1. CHECK'.format(np.sum(probabilities))
    
    return probabilities


class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    Example (for sigmoid):
        >>> sigmoid_layer = Activation("sigmoid")
        >>> z = sigmoid_layer(a)
        >>> gradient = sigmoid_layer.backward(delta=1.0)
    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        assert isinstance(activation_type, str) and len(activation_type) > 0

        if activation_type not in ["sigmoid", "tanh", "ReLU"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type
        # Placeholder for input. This will be used for computing gradients.
        self.x = None

    def __call__(self, a):
        """
        This method allows your instances to be callable.
        """
        return self.forward(a)

    def forward(self, a):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)

        elif self.activation_type == "tanh":
            return self.tanh(a)

        elif self.activation_type == "ReLU":
            return self.ReLU(a)

    def backward(self, delta):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()

        elif self.activation_type == "tanh":
            grad = self.grad_tanh()

        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()

        return grad * delta

    def sigmoid(self, x):
        """
        Implement sigmoid activation.
        """
        assert isinstance(x, np.ndarray)

        self.x = x
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        """
        Implements funny tanh.
        f(x) = 1.7159 * tanh(2x/3)
        """
        assert isinstance(x, np.ndarray)

        self.x = x
        return 1.7159 * np.tanh( (2/3) * x)

    def ReLU(self, x):
        """
        Implements ReLU.
        """
        assert isinstance(x, np.ndarray)

        self.x = x
        return np.maximum(0,x)

    def grad_sigmoid(self):
        """
        Computes the gradient of sigmoid.
        """
        numerator = self.x
        denominator = (1 + np.exp(-self.x))**2
        return np.divide(numerator, denominator)

    def grad_tanh(self):
        """
        Compute the gradient for tanh here.
        """
        tanh = self.tanh(self.x) #can optimize here by remembering output of tanh function
        return (2/3) * (1 - tanh**2)

    def grad_ReLU(self):
        """
        Compute the gradient for ReLU here.
        """
        return (self.x >= 0)*1.0 #makes the gradient 1 where x>=0


class Layer():
    """
    This class implements Fully Connected layers for your neural network.

    Example:
        >>> fully_connected_layer = Layer(784, 100)
        >>> output = fully_connected_layer(input)
        >>> gradient = fully_connected_layer.backward(delta=1.0)
    """

    def __init__(self, in_units, out_units):
        """
        Define the architecture and create placeholder.
        """
        np.random.seed(42)
        self.in_units = in_units
        self.out_units = out_units
        self.w = np.random.normal(0, 1, (out_units, in_units)) # Sampling weight from normal distribution
        self.b = np.random.normal(0, 1, (out_units, )) # Sampling bias from normal distribution
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        Compute the forward pass through the layer here.
        Do not apply activation here.
        Return self.a
        """
        assert isinstance(x, np.ndarray)
        self.x = x
        self.a = (self.x @ self.w.T) + self.b # (Nxd)x(dxk) + (k,) 

        assert self.x.shape[0] == self.a.shape[0] and self.w.shape[0] == self.a.shape[1], 'matrix multiplication fucked up here. CHECK!'
        return self.a

    def backward(self, delta):
        """
        This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        Return self.dx
        """
        assert isinstance(delta, np.ndarray)
        assert delta.ndim == 1, 'delta is expected to be one dimensional numpy array'
        assert delta.shape[0] == self.out_units, 'number of deltas != number of nodes'

        

        raise NotImplementedError("Backprop for Layer not implemented.")


class Neuralnetwork():
    """
    Create a Neural Network specified by the input configuration.

    Example:
        >>> net = NeuralNetwork(config)
        >>> output = net(input)
        >>> net.backward()
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []     # Store all layers in this list.
        self.x = None        # Save the input to forward in this
        self.y = None        # Save the output vector of model in this
        self.targets = None  # Save the targets in forward in this variable

        # Add layers specified by layer_specs.
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1]))
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        raise NotImplementedError("Forward not implemented for NeuralNetwork")

    def loss(self, logits, targets):
        '''
        compute the categorical cross-entropy loss and return it.
        '''
        raise NotImplementedError("Loss not implemented for NeuralNetwork")

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        raise NotImplementedError("Backprop not implemented for NeuralNetwork")


def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    Train your model here.
    Implement batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """

    raise NotImplementedError("Train method not implemented")


def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    """

    raise NotImplementedError("Test method not implemented")


if __name__ == "__main__":
    # Load the configuration.
    config = load_config("./")

    # Create the model
    model  = Neuralnetwork(config)

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    # Create splits for validation data here.
    # x_valid, y_valid = ...

    # train the model
    train(model, x_train, y_train, x_valid, y_valid, config)

    test_acc = test(model, x_test, y_test)
