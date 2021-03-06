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
import pickle

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

    img_normalized = (2.0 * (img) - 255.0) / 255.0
    
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
    max_entry_each_column = np.amax(x, axis=0) # size N
    numerator = np.exp(x - max_entry_each_column) #size c x N (uses broadcasting)
    denominator = np.sum(numerator, axis=0) # size c x N
    probabilities = numerator/denominator # (uses broadcasting)
    
    N = len(max_entry_each_column)
    assert np.abs(np.sum(probabilities) - N) < 1e-4, 'probabilities sum to {0}, not 1. CHECK'.format(np.sum(probabilities))
    
    return probabilities.T


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
        return 1 / (1 + np.exp(-x + 1e-9))

    def tanh(self, x):
        """
        Implements tanh.
        """
        assert isinstance(x, np.ndarray)

        self.x = x
        return np.tanh(x)

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
        return self.sigmoid(self.x) * (1 - self.sigmoid(self.x))

    def grad_tanh(self):
        """
        Computes the gradient for tanh.
        """
        tanh = self.tanh(self.x) 
        return (1 - tanh**2)

    def grad_ReLU(self):
        """
        Computes the gradient for ReLU.
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
        :param in_units: number of input units to layer
        :param out_units: number of output units
        """
        np.random.seed(42)
        self.in_units = in_units
        self.out_units = out_units
        self.w = np.random.normal(0, 1, (in_units, out_units)) # Sampling weight from normal distribution
        self.b = np.zeros((out_units, ))
        self.x = None    # Save the input to forward in this
        self.a = None    # Save the output of forward pass in this (without activation)

        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

        self.prev_d_w = None
        self.prev_d_b = None

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
        assert x.shape[1] == self.w.shape[0], 'Size of input incompatiable for matrix multiplication'
        
        self.x = x
        self.a = (self.x @ self.w) + self.b # (N x in) x (in x out) + (out,) 

        return self.a

    def backward(self, delta):
        """
        This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.

        The matrix sizes are:
        # delta = (N, out)
        # self.w = (in, out)
        # self.x = (N, in)

        :param delta: gradient from next layer
        :return self.dx: delta to pass to the previous layer

        """
        assert isinstance(delta, np.ndarray)
        assert delta.shape[1] == self.out_units, 'Delta shape is wrong. Check!'
        assert delta.shape[1] == self.w.shape[1], 'Matrix multiplication will fail. Check matrix sizes'
        assert delta.shape[0] == self.x.shape[0], 'Matrix multiplication will fail. Check matrix sizes'

        if self.d_w is None:
            self.prev_d_w = 0
            self.prev_d_b = 0
        else:
            self.prev_d_w = self.d_w
            self.prev_d_b = self.d_b

        # Calculating gradients
        self.d_x = delta @ self.w.T # (N, out) x (out, in) = (N, in)
        self.d_w = -1.0 * self.x.T @ delta # (in, N) x (N, out) = (in, out)
        self.d_b = -1.0 * np.sum(delta, axis=0) # (out, )


        return self.d_x

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
        Make NeuralNetwork instance callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):
        """
        Compute forward pass through all the layers in the network and return it.
        If targets are provided, return loss as well.
        """
        self.targets = targets

        # feed forward through all layers and activations
        for layer in self.layers:
            x = layer(x)

        # applying softmax to output
        self.y = softmax(x)

        # Calculating loss if possible
        if self.targets is not None:
            loss = self.loss(self.y, self.targets)
            return self.y, loss

        return self.y, None

    def loss(self, logits, targets):
        '''
        Compute the categorical cross-entropy loss and return it.
        '''
        loss = -1.0 * np.sum(targets * np.log(logits + 1e-7))/targets.shape[0]

        return loss

    def backward(self):
        '''
        Implement backpropagation here.
        Call backward methods of individual layer's.
        '''
        if self.targets is None:
            raise RuntimeError('targets not given! Cannot do backpropagation')
        
        # Computing delta at output layer
        delta = (self.targets - self.y)
        
        # Backpropagating delta and calculating gradients
        for layer in self.layers[::-1]:
            delta = layer.backward(-delta)


def get_batch_indices(data_size, batch_size):
    mini_size = int(data_size/batch_size)
    remainder = int(data_size%batch_size)
    if remainder == 0:
        b_indices = [np.array(range(batch_size)) + (batch_size * i) for i in range(mini_size)]
    else:
        b_indices = [np.array(range(batch_size)) + (batch_size * i) for i in range(mini_size)]
        b_indices.extend([np.array(range(data_size-remainder, data_size))])

    return b_indices


def get_k_fold_ind(k, x_data):

    ind = np.array_split(np.array(range(x_data.shape[0])), k)
    return ind


def update_plots(x_vec, y1_data, line1, y2_data, line2, title='', pause_time=0.0001):
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
        line2, = ax.plot(x_vec, y2_data, '-o', alpha=0.8)
        line1.set_label("Train")
        line2.set_label("Validation")
        # update plot label/title
        plt.ylabel('Loss')
        plt.title('{}'.format(title))
        ax.legend()
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    line2.set_ydata(y2_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
    if np.min(y2_data) <= line1.axes.get_ylim()[0] or np.max(y2_data) >= line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y2_data) - np.std(y2_data), np.max(y2_data) + np.std(y2_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1, line2

def save_model(model, epoch):
    """
    Pickles the model

    :param weights: weights to be saved
    :return: nothing
    """
    filehandler = open('weights_epoch{}'.format(epoch), 'wb')
    pickle.dump(model, filehandler)

def load_model(epoch):
    """
    Get model from pickle file
    :return: model
    """
    filehandler = open('weights_epoch{}'.format(epoch), 'rb')
    model = pickle.load(filehandler)
    return model

def train(model, x_train, y_train, x_valid, y_valid, config, live_plot=False):
    """
    Train your model here.
    Implement mini-batch SGD to train the model.
    Implement Early Stopping.
    Use config to set parameters for training like learning rate, momentum, etc.
    """
    # load parameters
    lr = config["learning_rate"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    early_stop = config["early_stop"]
    early_stop_epoch = config["early_stop_epoch"]
    l2_penalty = config["L2_penalty"]
    use_momentum = config["momentum"]
    momentum_gamma = config["momentum_gamma"]

    dataset_size = x_train.shape[0]
    batch_indices = get_batch_indices(dataset_size, batch_size)
    val_loss_threshold = np.inf
    count = 0

    # graphing utility
    x_vec = np.linspace(1, epochs, epochs)
    val_vec = np.zeros(len(x_vec))
    train_vec = np.zeros(len(x_vec))
    train_line = []
    val_line = []

    plot_data = []

    for epoch in range(epochs):
        print(epoch, end=' ')
        # shuffle data
        rand_ind = np.random.permutation(dataset_size)
        x_train = x_train[rand_ind]
        y_train = y_train[rand_ind]

        # batch sgd
        for ind in batch_indices:
            x_batch = x_train[ind]
            y_batch = y_train[ind]

            prediction, train_loss = model(x_batch, targets=y_batch)
            model.backward()

            # weight update
            for layer in model.layers:
                if isinstance(layer, Layer):
                    # Updating weights and bias
                    if use_momentum:
                        momentum_w = momentum_gamma * layer.prev_d_w
                        momentum_b = momentum_gamma * layer.prev_d_b
                        layer.w += (layer.d_w + momentum_w + 2 * l2_penalty * layer.w) * lr
                        layer.b += (layer.d_b + momentum_b) * lr
                    else:
                        layer.w += (layer.d_w + 2 * l2_penalty * layer.w) * lr
                        layer.b += (layer.d_b) * lr

        _, train_loss = model(x_train, targets=y_train)
        _, val_loss = model(x_valid, targets=y_valid)
        train_acc = test(model, x_train, y_train)
        val_acc = test(model, x_valid, y_valid)

        # dynamic plot
        if live_plot:
            train_vec[epoch] = train_loss
            val_vec[epoch] = val_loss
            train_line, val_line = update_plots(x_vec, train_vec, train_line, val_vec, val_line, "Loss")

        # save plot stuff
        plot_data.append((train_loss, val_loss, train_acc, val_acc))

        # save best model based on loss
        if val_loss < val_loss_threshold:
            val_loss_threshold = val_loss
            save_model(model, 0)
            count = 0
            best_epoch = epoch
        # else val_loss goes up, early stop?
        else:
            count += 1
            if count > early_stop_epoch and early_stop:
                print("Early stop on epoch {}".format(epoch))
                break

    if count <= early_stop_epoch:
        print("Trained all {} epochs".format(epochs+1))
    print("Best epoch: {}".format(best_epoch))
    return plot_data

def test(model, X_test, y_test):
    """
    Calculate and return the accuracy on the test set.
    :param model: model to be tested
    :param X_test: input data
    :param y_test: target
    :return: accuracy in percent
    """
    pred_props, loss = model(X_test, targets=y_test)
    pred = np.argmax(pred_props, axis=1)
    targets = np.argmax(y_test, axis=1)
    return sum(pred==targets)/y_test.shape[0]

def gradient_checker(x_train, y_train):
    '''
    Impements and prints first order gradient checking for weights 
    and biases of the output and hidden layers

    :param x_train: train data
    :param y_train: train labels
    '''
    config = load_config("config.yaml")
    model = Neuralnetwork(config)
    x, y = x_train[11:12], y_train[11:12]

    eps = 1e-2
    bias_index = 8
    # Bias gradient check
    gradient_check_bias(model, x, y, bias_index, eps)
    # Weights gradient check
    for weight_index in [(2,3), (2,2), (1,4), (3,1)]:
        gradient_check_weight(model, x, y, weight_index, eps)

    
def gradient_check_bias(model, x, y, bias_index, eps):
    '''
    Gradient check for biases of output and hidden layer 
    '''
    print('\n')
    for layer_index in [-1, -3]:
        model.layers[layer_index].b[bias_index] += eps
        _, E1 = model(x, targets=y)
        model.layers[layer_index].b[bias_index] -= 2*eps
        _, E2 = model(x, targets=y)
        
        model.layers[layer_index].b[bias_index] += eps #back to original bias value

        model(x,targets=y)
        model.backward()
        actual_gradient = model.layers[layer_index].d_b[bias_index]
        computed_gradient = (E2 - E1)/(2*eps)

        print(f'--------- For layer index={layer_index}, bias_index = {bias_index} ---------')
        print('actual_gradient, computed_gradient = ', actual_gradient, computed_gradient)
        print('difference in gradient = ', np.abs(actual_gradient - computed_gradient))

def gradient_check_weight(model, x, y, weight_index, eps):
    '''
    Gradient check for weights of output and hidden layer 
    '''
    print('\n')
    for layer_index in [-1, -3]:
        model.layers[layer_index].w[weight_index] += eps
        _, E1 = model(x, targets=y)
        model.layers[layer_index].w[weight_index] -= 2*eps
        _, E2 = model(x, targets=y)
        
        model.layers[layer_index].w[weight_index] += eps #back to original weight value

        model(x,targets=y)
        model.backward()
        actual_gradient = model.layers[layer_index].d_w[weight_index]
        computed_gradient = (E2 - E1)/(2*eps)

        print(f'--------- For layer index={layer_index}, weight_index = {weight_index} ---------')
        print('actual_gradient, computed_gradient = ', actual_gradient, computed_gradient)
        print('difference in gradient = ', np.abs(actual_gradient - computed_gradient))

if __name__ == "__main__":
    # Load the configuration.
    config = load_config("config.yaml")

    # Load the data
    x_train, y_train = load_data(path="./", mode="train")
    x_test,  y_test  = load_data(path="./", mode="t10k")

    gradient_checker(x_train, y_train)
