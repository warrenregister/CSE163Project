'''
This file contains methods used in analysis.py
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from sklearn.model_selection import KFold
from GDA import Gaussian_Discriminant_Analysis
from neural_network import NeuralNetwork
from logistic_regression import LogisticRegressionBGD
from scipy.stats import ttest_rel
import pickle


def get_batches(directory):
    '''
    Method which gets pickled numpy arrays from a specified directory
    and returns them as a list.
    '''

    def load_data(file):
        '''
        Return a pickled numpy array stored in a given file path as a
        numpy array.
        '''
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data

    batches = list()
    for file in os.listdir(directory):
        if 'data' in file:
            batches.append(load_data(directory + file))
    return batches


def stack_data(batches):
    '''
    Method which takes a list of numpy arrays and combines them into
    a single array by vertically stacking them.
    '''
    n = batches[0][b'data'].shape[1] + 1
    data = np.ones((1, n))
    for batch in batches:
        a = np.array(batch[b'data'])
        b = np.array(batch[b'labels'])
        b = b.reshape(b.shape[0], 1)
        c = np.hstack([a, b])
        data = np.vstack([data, c])
    data = np.delete(data, 0, axis=0)
    return(data[:, :-1]/255, data[:, -1])


def get_ys(y):
    '''
    Takes in vector of multi-class labels and returns a list of lists
    of binary class labels for each class, one for each class in y.
    '''
    ys = []
    for num in range(max(y) + 1):
        zeroorone = [(0 if num != i else 1) for i in y]
        ys.append(np.array(zeroorone))
    return ys


def get_data(directory):
    '''
    Method which gets all CIFAR 10 data, combines it into a single array,
    splits that array into x and y, converts y into a list of binary
    labels, then returns x and y.
    '''
    batches = get_batches(directory)
    x, y = stack_data(batches)
    y2 = np.zeros((x.shape[0], 10))
    for i, label in enumerate(y):
        y2[i, int(label)] += 1
    y = y2
    return x, y


def train_model(x, y, model_type, params=None):
    '''
    Method which takes training and test data as well as a machine
    learning model and any parameters it might have and trains the
    model, and times the training, returning the time and the trained
    model.
    '''
    t0 = time.time()
    model = model_type()
    model.fit(x, y)
    t1 = time.time()
    return (t1 - t0, model)


def unflatten(x):
    '''
    Method which takes in a matrix of flattened RGB 32x32 pixel images as
    m x 3072 numpy.arrays and converts each image into a 32x32x3 color image.
    Returns an m x 32 x 32 x 3 matrix of color images.
    '''
    images = np.zeros((x.shape[0], 32, 32, 3))
    for i in range(50000):
        r = x[i][0:1024].reshape((32, 32))
        g = x[i][1024:2048].reshape((32, 32))
        b = x[i][2048:3072].reshape((32, 32))
        image = np.zeros((32, 32, 3))
        image[:, :, 0] = r
        image[:, :,  1] = g
        image[:, :, 2] = b
        images[i] = image
    return images


def grayscale_image(img):
    '''
    Method which takes in a RGB image and returnsa grayscaled version of
    that image.
    '''
    weights = np.array([0.2989, 0.5870, 0.1140])
    return np.dot(img, weights)


def edge_detection(image):
    '''
    Method which takes in a grayscaled image and performs a convolution of
    an edge detection kernel onto it. The result of this convolution is
    returned.
    '''
    kernel = (np.ones((3, 3)) * -1)
    kernel[1, 1] = 8
    h, k = kernel.shape
    m, n = image.shape
    padded_image = np.zeros((m + h-1, n + k-1))
    padded_image[1:-1, 1:-1] = image

    edge_img = np.zeros_like(image)
    for col in range(n):
        for row in range(m):
            edge_img[row, col] = np.sum(np.multiply(kernel,
                                        padded_image[row:row+3, col:col+3]))
    return edge_img


def test_simple_models(x, y):
    '''
    Method which takes in training features x and labels y and creates 10
    splits in the data, then trains each 'simple' model (logistic regression
    and GDA) in a 1 vs All classification attempt using 9 groups as training
    and the last group as test data for all 10 possible combinations of this.
    Returns 2 dictionaries one each for the times and accuracies of each model
    for each combination of data.
    '''
    kfold = KFold(n_splits=10)
    kfold.get_n_splits(x)
    folds = kfold.split(x)
    model_names = ['Gaussian Discriminant Analysis', 'Logistic Regression']
    dic = generate_dicts(model_names)

    fold = 1
    for train, test in folds:
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        for name in model_names:
            model_type = Gaussian_Discriminant_Analysis
            if(name == 'Logistic Regression'):
                model_type = LogisticRegressionBGD

            preds = []
            time_list = []
            for i in range(10):
                time_float, model = train_model(x_train, y_train[:, i],
                                                model_type)
                pred = model.predict(x_test)
                preds.append(pred)
                time_list.append(time_float)

            preds = np.array(preds).T
            for index, row in enumerate(preds):
                max_index = np.argmax(preds[index])
                preds[index] = np.zeros_like(preds[index])
                preds[index, max_index] = 1

            acc = 1 - np.sum(abs(y_test - preds) / 2) / len(y_test)
            tot_time = np.sum(time_list)
            dic['Accuracy'][name].append(acc)
            dic['Time'][name].append(tot_time)
            print('Fold ' + str(fold), name, acc, tot_time)
            print()

        fold += 1

    return dic


def generate_dicts(models):
    '''
    Method which generates dictionaries of times and accuracies with sub
    dictionaries for each model and sub-sub dictionaries for each combination
    of data used for training and testing.
    '''
    aandt = {'Accuracy': {}, 'Time': {}}
    for key in aandt.keys():
        for model in models:
            aandt[key][model] = []
    return aandt


def test_nn(x, y, archs):
    '''
    Method which takes in training features x and labels y and creates 10
    splits in the data, then trains a Neural Network for each of the
    architectures passed inusing 9 groups as training and the last group as
    test data for all 10 possible combinations of this.
    Returns 2 lists one each for the times and accuracies of each model
    for each combination of data.

    archs: list of lists of integers where each inner list represents the size
    of each layer in a Neural Network.
    '''
    kfold = KFold(n_splits=10)
    kfold.get_n_splits(x)
    folds = kfold.split(x)

    fold = 0
    names = ['Network ' + str(arch) for arch in archs]
    dic = generate_dicts(names)

    fold = 0
    for train, test in folds:
        x_train, x_test = x[train], x[test]
        y_train, y_test = y[train], y[test]

        for i, arch in enumerate(archs):
            model = NeuralNetwork(arch)
            t0 = time.time()
            model.fit(x_train, y_train, batch_size=100, epochs=50)
            t1 = time.time()

            dic['Time'][names[i]].append(t1 - t0)
            print('Time: ' + str(dic['Time'][names[i]][fold]))

            preds = model.predict(x_test)
            for index, row in enumerate(preds):
                max_index = np.argmax(preds[index])
                preds[index] = np.zeros_like(preds[index])
                preds[index, max_index] = 1
            acc = 1 - np.sum(abs(preds - y_test) / 2) / len(y)
            dic['Accuracy'][names[i]].append(acc)

    return dic


def get_stats(dics):
    '''
    Method which takes in dictionaries of time and accuracies for different
    models and returns a list of tuples of the means and variances for both
    statistics as well as a list of the names of each model and the t_test
    results of comparing the times and accuracies of each model.
    '''
    stats = []
    names = []
    values = []

    for dic in dics:
        for name in dic['Accuracy'].keys():
            tup = []
            values_tup = []
            tup.append(np.mean(dic['Accuracy'][name]))
            tup.append(np.var(dic['Accuracy'][name]))
            tup.append(np.mean(dic['Time'][name]))
            tup.append(np.var(dic['Time'][name]))
            values_tup.append(dic['Accuracy'][name])
            values_tup.append(dic['Time'][name])
            values.append(tuple(values_tup))
            names.append(name)
            stats.append(tuple(tup))

    t_test = []
    for i, tup in enumerate(values[:-1]):
        for j, tup2 in enumerate(values[i + 1:]):
            test = []
            test.append('' + names[i] + ' x ' + names[i + j + 1])
            test.append(ttest_rel(tup[0], tup2[0]))
            test.append(ttest_rel(tup[1], tup2[1]))
            t_test.append(tuple(test))
    return stats, names, t_test


def graph_stats(stats, models=None, savedir=None):
    '''
    Given a list of stats of various machine learning
    models, generates a bar plot of the mean accuracies
    and mean times of each model, saves to given directory
    if directory specified.
    '''
    mean_acc = [tup[0] for tup in stats]
    varis_acc = [tup[1] for tup in stats]
    mean_time = [tup[2] for tup in stats]
    varis_time = [tup[3] for tup in stats]
    cols = ['Mean Accuracy', 'Std. Dv Accuracy', 'Mean Time', 'Std Dv Time']
    df = pd.DataFrame(zip(mean_acc, varis_acc, mean_time, varis_time),
                      columns=cols)
    if models is None:
        models = ['Logistic Regression', 'GDA', 'Neural Network']
    df['Model'] = models
    fig, ax = plt.subplots(2)
    sns.barplot(data=df, x='Model', y='Mean Time', ax=ax[0])
    sns.barplot(data=df, x='Model', y='Mean Accuracy', ax=ax[1])
    if savedir:
        plt.savefig(savedir)
