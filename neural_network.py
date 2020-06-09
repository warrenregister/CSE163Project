'''
This file contains the NeuralNetwork class.
'''
import numpy as np


class NeuralNetwork(object):
    '''
    Class for basic Feed Forward Neural Network for classification.
    '''
    def __init__(self, layers, learning_rate=0.1):
        '''
        Initiate fields, create bias and weight matrices to fit size of the
        layers contained in layers.

        Arguments:
        layers: list of ints representing size of each layer, first int is
        assumed to be the size of the input layer
        learning_rate: rate at which weights and biases are changed, higher
        means bigger jumps, and lower means smaller jumps.
        '''
        self._num_layers = len(layers)
        self._sizes = layers
        self._biases = [np.random.randn(1, y) for y in layers[1:]]
        self._weights = [np.random.randn(x, y) for x, y in
                         zip(layers[:-1], layers[1:])]
        self._alpha = learning_rate
        self._g = lambda x: 1 / (1 + np.exp(-np.clip(x, -600, 600)))
        self._g_prime = lambda x: self._g(x) * (1 - self._g(x))

    def predict(self, x):
        '''
        Takes in matrix of values similar to training examples and returns
        probabilities of each row being of each class.

        Arguments:
        x: np.array of values similar to training examples
        '''
        output = x.copy()
        for b, w in zip(self._biases, self._weights):
            output = self._g(np.dot(output, w) + b)
        return output

    def fit(self, x, y, epochs=100, batch_size=10, show_progress=False):
        '''
        Fit Neural Network to data using Stochastic Gradient Descent.

        Arguments:
        x: np.array of training examples of shape m, n where n is the size of
        the first layer of the network.
        y: np.array of labels for the training examples.
        epochs: number of times to optimize with Stochastic Gradient Descent.
        batch_size: number of examples to user per batch of SGD.
        '''
        y2 = y.copy()
        if len(y.shape) == 1:
            y2 = y2.reshape(y.shape[0], 1)
        k = y2.shape[1]
        data = np.hstack([x, y2])
        m, n = data.shape
        for _ in range(epochs):
            xcopy = data.copy()
            np.random.shuffle(data.copy())
            for i in range(0, m // batch_size, 2):
                index = i * batch_size
                if(index + batch_size < m):
                    batch = xcopy[index: index + batch_size]
                    self._descend(batch[:, :-k], batch[:, -k:])
            if show_progress:
                self.show_progress(x, y, _)

    def _descend(self, x, y):
        '''
        Performs one iteration of SGD, updating weights and biases based
        on their current accuracy on the training examples.

        Arguments:
        x: numpy.array of training examples to perform SGD with.
        y: numpy.array of training labels to perform SGD with.
        '''
        deriv_w = [np.zeros_like(matrix) for matrix in self._weights]
        deriv_b = [np.zeros_like(vector) for vector in self._biases]
        m, n = x.shape
        for num in range(m):
            deriv_2_w, deriv_2_b = self._back_propogate(x, y)
            for i in range(len(deriv_w)):
                deriv_w[i] += deriv_2_w[i]
                deriv_b[i] += deriv_2_b[i]
        eta = (self._alpha / m)
        for i in range(len(deriv_w)):
            self._weights[i] -= eta * deriv_w[i]
            self._biases[i] -= eta * deriv_b[i]

    def _back_propogate(self, x, y):
        '''
        Calculates error for each weight and bias in each layer using the back
        propogation algorithm.

        Arguments:
        Arguments:
        x: numpy.array of training examples to feed through model.
        y: numpy.array of training labels to feed through model.
        '''
        a = x
        a_s = [x]
        zs = list()
        for b, w in zip(self._biases, self._weights):
            z = a.dot(w) + b
            a = self._g(z)
            zs.append(z)
            a_s.append(a)

        deriv_w = [np.zeros_like(matrix) for matrix in self._weights]
        deriv_b = [np.zeros_like(vector) for vector in self._biases]
        change = (a - y) * self._g(z)
        deriv_w[-1] = np.mean(change * a_s[-2], axis=0)
        deriv_b[-1] = np.mean(change)
        for i in range(2, self._num_layers):
            change = np.dot(change, self._weights[-i + 1].T)
            change *= self._g_prime(zs[-i])
            deriv_w[-i] = np.dot(change.T, a_s[-i-1])
            deriv_b[-i] = np.mean(change, axis=0)
        deriv_w = [w.T for w in deriv_w]
        for i in range(len(deriv_w)):
            matrix = deriv_w[i]
            if len(matrix.shape) == 1:
                deriv_w[i] = matrix.reshape(matrix.shape[0], 1)
        return (deriv_w, deriv_b)

    def show_progress(self, x, y, epoch):
        '''
        Method which outputs the accuracy of the network.
        '''
        preds = self.predict(x)
        acc = 0

        if preds.shape[1] > 1:
            print('top')
            for index, row in enumerate(preds):
                max_index = np.argmax(preds[index])
                preds[index] = np.zeros_like(preds[index])
                preds[index, max_index] = 1
            acc = 1 - np.sum(abs(preds - y) / 2) / len(y)
        else:
            print('bottom')
            preds = [0 if x < 0.5 else 1 for x in preds]
            acc = 1 - np.sum(abs(y - preds)) / len(y)
        print('epoch ' + str(epoch) + ': ' + str(acc))
