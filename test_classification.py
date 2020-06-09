'''
This file contains methods used to test Machine Learning
algorithms on randomly generated classification problems.
'''

import numpy as np
import pandas as pd
from GDA import Gaussian_Discriminant_Analysis
from neural_network import NeuralNetwork
from logistic_regression import LogisticRegressionNewtons
from logistic_regression import LogisticRegressionBGD
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns


def plot_contour(x, y, model, corr=1, cmap='Paired'):
    '''
    Take in x and y training data and a model, plot the model's
    decision boundary as a conotour plot.
    '''
    cmap = plt.get_cmap(cmap)
    fig, ax = plt.subplots()
    ax.scatter(x[:, -2], x[:, -1], c=y, cmap=cmap)
    xmin, xmax = x[:, 0].min() - 1, x[:, 0].max() + 1
    ymin, ymax = x[:, 1].min() - 1, x[:, 1].max() + 1
    x_span = np.linspace(xmin, xmax, 100)
    y_span = np.linspace(ymin, ymax, 100)
    xs, ys = np.meshgrid(x_span, y_span)
    labels = model.predict(np.c_[xs.ravel(), ys.ravel()])

    z = np.array([0 if x < 0.5 else 1 for x in labels])
    z = z.reshape(xs.shape)

    ax.contourf(xs, ys, z, cmap=cmap, alpha=0.5)


def main():
    data = data = make_classification(n_features=2, n_redundant=0)
    x = data[0]
    y = data[1]
    df = pd.DataFrame(x)
    df['y'] = y

    sns.relplot(data=df, x=0, y=1, hue='y')

    gda = Gaussian_Discriminant_Analysis()
    net = NeuralNetwork([2, 4, 1])
    lrn = LogisticRegressionNewtons()
    lrb = LogisticRegressionBGD(change=1e-7)
    models = [gda, net, lrn, lrb]
    names = ['Gaussian Discriminant Analysis', 'Neural Network',
             'Logsitic Regression Newtons',
             'Logistic Regression Gradient Descent']
    preds = list()
    accs = list()

    import time
    times = list()
    for model in models:
        t0 = time.time()
        model.fit(x, y)
        t1 = time.time()
        pred = model.predict(x)
        pred = [0 if x < 0.5 else 1 for x in pred]
        preds.append(pred)
        accs.append(1 - np.sum(abs(pred - y)) / 100)
        times.append(t1-t0)

    print('Accuracies and Times:')
    for i in range(len(times)):
        print(names[i] + ': ' + str(accs[i]) + '%, ' + str(times[i]) +
              ' seconds')

    model_data = pd.DataFrame(zip(names, accs, times),
                              columns=['Model', 'Accuracy', 'Time'])
    sns.set()
    fig, ax = plt.subplots()
    model_data.Accuracy.plot(kind='bar', color='green', ax=ax,
                             width=.3, position=0)
    model_data.Time.plot(kind='bar', color='blue', ax=ax, width=.3,
                         position=1)
    plt.xticks(range(4), labels=names, rotation=55)
    plt.legend(loc=1, labels=['Accuracy', 'Time (seconds)'])
    plt.title('Models on Test Binary Classification')
    plt.savefig('./test_graphs.png', bbox_inches='tight')

    plot_contour(x, y, models[0])
    plot_contour(x, y, models[1])
    plot_contour(x, y, models[2])
    plot_contour(x, y, models[3])


if __name__ == '__main__':
    main()
