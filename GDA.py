'''
This file contains the Gaussian Discriminany Analysis object class.
'''
import numpy as np
from numpy.linalg import inv
from linear_model import LinearModel


class Gaussian_Discriminant_Analysis(LinearModel):
    '''
    Class for using Gaussian Discriminant Analysis to create a linear model
    for binary classification.
    '''
    def fit(self, x, y):
        '''
        Fir GDA parameters do data and use them to solve for theta.

        Arguments:
        x: numpy.array of training examples,  m x n
        y: numpy.array of training labels, m x 1
        '''
        m, n = x.shape

        # solve for parameters
        phi = np.mean(y)
        mu0 = np.mean(x[y == 0], axis=0)
        mu1 = np.mean(x[y == 1], axis=0)
        copy = x.copy()
        copy[y == 0] -= mu0
        copy[y == 1] -= mu1
        sigma = (1 / m) * copy.T.dot(copy)

        sig_inv = inv(sigma)
        theta0 = (mu0.T.dot(sig_inv).dot(mu0) - mu1.T.dot(sig_inv).dot(mu1))
        theta0 = (1/2) * theta0 - np.log((1-phi)/phi)
        theta = sig_inv.T.dot(mu1 - mu0)
        self._theta = np.hstack([theta0, theta])

    def predict(self, x):
        '''
        Use theta to predict each training example in x.
        Arguments:
        x: numpy.array of training examples,  m x n
        '''
        def g(x):
            '''
            Returns x run through sigmoid function.
            '''
            return 1 / (1 + np.exp(- x.dot(self._theta)))

        new = np.ones((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
        new[:, 1:] = x
        preds = g(new)
        return np.array(preds)
