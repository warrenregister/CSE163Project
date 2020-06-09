'''
This file continas methods for training a series of Machine Learning
algorithms on the CIFAR 10 image classification DataSet.
'''

import numpy as np
import analysis_helpers as ah


def iteration(x, y, fpath, inputsize):
    '''
    Method which runs one iteration of model training and testing. Trains 10
    of each model, tests each one, compares the statistics of each using a
    t test, graphs the times and accuracies.
    '''
    dic1 = ah.test_simple_models(x, y)
    dic2 = ah.test_nn(x, y, [[inputsize, 100, 10, 10]])  # Really, really, slow
    stats, names, t_test = ah.get_stats([dic1, dic2])
    for comparison in t_test:
        print('t test of ' + comparison[0] + ' True indicates passing:')
        print('Accuracy:')
        print(comparison[1][1] < 0.05)
        print('Time')
        print(comparison[2][1] < 0.05)
    ah.graph_stats(stats, fpath)


def main():
    x, y = ah.get_data('./cifar-10-batches-py/')
    iteration(x, y, './first_run.png', 3072)
    x = ah.unflatten(x)
    edge_imgs = np.zeros((x.shape[0], 32, 32))
    for i, image in enumerate(x):
        gray = ah.grayscale_image(image)
        edge_imgs[i] = ah.edge_detection(gray)

    edge_imgs = ah.edge_imgs.reshape(50000, 1024)

    iteration(ah.edge_imgs, y, './edge_detection.png', 1024)


if __name__ == '__main__':
    main()
