'''
This file holds the LinearModel class which all linear models will inheret
from.
'''


class LinearModel(object):
    '''
    Baseline class for all Linear Models.
    '''
    def __init__(self, learning_rate=0.01, change=1e-5):
        '''
        Initialize model parameters.
        '''
        self._alpha = learning_rate
        self._change = change
