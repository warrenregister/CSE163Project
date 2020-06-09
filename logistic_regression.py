'''
This file holds classes for implementing various forms of Logistic
Regression.
'''
from linear_model import LinearModel
import numpy as np


class LogisticRegression(LinearModel):
    '''
    Base class for Logistic Regression models
    '''
    def predict(self, x):
        '''
        Return probablities of each new example in x having the label
        y = 1.

        Arguments:
        x: numpy.array of examples similar to those used in training,  m x n
        '''
        x = self._add_intercept(x)
        return self._g(x.dot(self._theta))

    def _add_intercept(self, x):
        new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
        new_x[:, 0] = 1
        new_x[:, 1:] = x
        return new_x

    def _g(self, x):
        '''
        Return sigmoid(x).
        '''
        return 1 / (1 + np.exp(-x))


class LogisticRegressionBGD(LogisticRegression):
    '''
    Logistic Regression model using Batch Gradient Descent to fit the
    model to the data.
    '''
    def fit(self, x, y):
        '''
        Fit logistic regression classifier to training examples and labels
        (x, y) using Batch Gradient Descent for optimization.

        Arguments:
        x: numpy.array of training examples,  m x n
        y: numpy.array of training labels, m x 1
        '''
        x = self._add_intercept(x)
        m, n = x.shape
        self._theta = np.zeros(n)

        def gradient(x, y):
            '''
            Calculate derivative of cost function with respect to
            theta, return result.
            '''
            return (-1/m) * np.dot(x.T, self._g(np.dot(x, theta)) - y)
        old = 0

        while(True):
            theta = self._theta
            self._theta = self._theta + self._alpha * gradient(x, y)
            change = np.linalg.norm(self._theta - theta, ord=1)
            if np.abs(change - old) < self._change:
                break
            old = change


class LogisticRegressionNewtons(LogisticRegression):
    '''
    Logistic Regression model using Newtons Method to fit the
    model to the data.
    '''
    def fit(self, x, y):
        '''
        Fit logistic regression classifier to training examples and labels
        (x, y) using Netwons Method for optimization.

        Arguments:
        x: numpy.array of training examples,  m x n
        y: numpy.array of training labels, m x 1
        '''
        x = self._add_intercept(x)
        m, n = x.shape
        self._theta = np.zeros(n)

        while True:
            theta = self._theta
            C = - (1 / m) * (y - self._g(x.dot(theta))).dot(x)
            x_theta = x.dot(theta)
            H = self._g(x_theta).dot(self._g(1 - x_theta)) * (x.T).dot(x)
            H += (1/m)
            H_inv = np.linalg.inv(H)
            self._theta = theta - H_inv.dot(C)

            if np.linalg.norm(self._theta - theta, ord=1) < self._change:
                break
