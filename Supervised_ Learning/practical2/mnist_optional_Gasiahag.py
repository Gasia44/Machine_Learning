#!/usr/bin/env python3
"""
This is a boilerplate file for you to get started on MNIST dataset.

This file has code to read labels and data from .gz files you can download from
http://yann.lecun.com/exdb/mnist/

Files will work if train-images-idx3-ubyte.gz file and
train-labels-idx1-ubyte.gz files are in the same directory as this
python file.
"""
from __future__ import print_function
import argparse
import gzip
import struct
import numpy as np
import matplotlib.pyplot as plt
import math


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mnist-train-data',
                        default='train-images-idx3-ubyte.gz',  # noqa
                        help='Path to train-images-idx3-ubyte.gz file '
                        'downloaded from http://yann.lecun.com/exdb/mnist/')
    parser.add_argument('--mnist-train-labels',
                        default='train-labels-idx1-ubyte.gz',  # noqa
                        help='Path to train-labels-idx1-ubyte.gz file '
                        'downloaded from http://yann.lecun.com/exdb/mnist/')
    args = parser.parse_args(*argument_array)
    return args






def sigmoid(s):
    h = 1.0 / (1.0 + np.power(math.e, -s))
    ### also h = 1.0 / (1.0 + np.exp(-s))
    return h

def FeatureScaling(X, l):
    mean = np.zeros(X.shape[1] -1)
    sd =np.zeros(X.shape[1] - 1);
    temp = X.copy()
    lambda_mod = np.zeros(X.shape[1])
    lambda_mod[0] = l

    for i in range(1, X.shape[1]):
        mean[i-1] = np.mean(X.T[i])

    for j in range(1, X.shape[1]):
        for i in range(X.shape[0]):
            temp[i][j] = X[i][j] - mean[j-1]
            sd[j-1] = sd[j-1] + (X[i][j] - mean[j-1])**2
        sd[j-1] = np.sqrt(sd[j-1] / X.shape[0])
        lambda_mod[j] = l / sd[j-1]

    for j in range(1, X.shape[1]):
        for i in range(X.shape[0]):
            temp[i][j] = temp[i][j] / sd[j-1]
    return temp, lambda_mod**2, mean, sd


def normalized_gradient(X, Y, beta, l):
    """
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    bb=0
    for i in range(len(Y)):
        bb = bb + (X[i] * Y[i] * (1 - (sigmoid(Y[i] * np.dot(X[i], beta)))))
        ### if y is a number(scalar), then np.dot(x,y) and x * y are the same. Here Y[i] is a scalar.
    return (-2*bb + 2*l * beta)/ X.shape[0]

def gradient_descent(X, Y, epsilon=1e-6, l=1, step_size=1e-4, max_steps=1000):
    """
    Implement gradient descent using full value of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    beta1 = np.zeros(X.shape[1])
    beta2 = np.zeros(X.shape[1])

    #index_array = np.arange(X.shape[0])
    #np.random.shuffle(index_array)
    i = 0
    #XX, ll, mean, sd = FeatureScaling(X, l)
    XX =X
    ll=l

    for s in range(max_steps):
        #if s % 1000 == 0:
            #print(s, beta2)
        ridge = normalized_gradient(XX, Y, beta2, ll)
        if np.linalg.norm(beta2- beta1) < epsilon * np.linalg.norm(beta2):
            break
        beta1 = beta2
        beta2 = beta2 - step_size * ridge;

    #b2 = np.divide(beta2[1: beta2.shape[0]], sd)
    #b1 = beta2[0] - sum(b2 * mean)

    #return np.append(b1,b2)
    return beta2


def stochastic_gradient_descent(X, Y, epsilon=0.0001, l=1, step_size=0.01,
                                max_steps=1000):
    """
    Implement gradient descent using stochastic approximation of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    beta1 = np.zeros(X.shape[1])
    beta2 = np.zeros(X.shape[1])

    index_array = np.arange(X.shape[0])
    np.random.shuffle(index_array)
    i = 0
    #XX, ll, mean, sd = FeatureScaling(X, l)
    XX =np.array(X)
    ll=np.array(l)
    #ll[0] = 0
    for s in range(max_steps):
        i = s % XX.shape[0]
        bb =( np.dot(XX[i], Y[i]) ) * (sigmoid(Y[i]* np.dot(XX[i], beta2)))

        ridge=  (-2*bb + np.dot(2*l, beta2) )/ X.shape[0]

        if ( (np.linalg.norm( beta2- beta1) ) < epsilon* np.linalg.norm(beta2) ):
            break

        beta1 = beta2
        beta2 = beta2 - step_size * ridge;
    #b2 = np.divide(beta2[1: beta2.shape[0]], sd)
    #b1 = beta2[0] - sum(b2 * mean)

    #return np.append(b1,b2)
    return beta2
'''
def loss(X, Y, beta):
    """
    Calculate sum of error squares divided by number of points.

    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :return: 1/N * SUM (y - x beta)^2
    """
    loss_ =  np.sum (( Y - np.dot(X, beta))**2) / len(Y)
    return loss_
'''
### this loss function is for linear regression, but our problem is logistic regression, which does binary classification.
### I think it is better to count the percentage of true guesses on our training set.

def validation(X, Y, beta):
    valid = 0.0
    for x, y in zip(X, Y):
        valid += (y * np.dot(beta.T, x)) > 0
    return valid * 100.0 / len(Y)

def main(args):
    # Read labels file into labels
    with gzip.open(args.mnist_train_labels, 'rb') as in_gzip:
        magic, num = struct.unpack('>II', in_gzip.read(8))
        all_labels = struct.unpack('>60000B', in_gzip.read(60000))

    # Read data file into numpy matrices
    with gzip.open(args.mnist_train_data, 'rb') as in_gzip:
        magic, num, rows, columns = struct.unpack('>IIII', in_gzip.read(16))
        all_data = [np.reshape(struct.unpack('>{}B'.format(rows * columns),
                                             in_gzip.read(rows * columns)),
                               (rows, columns))
                    for _ in range(60000)]

    # Select only labels and matrices of 4 and 9 digits.
    labels, data = zip(*[pair for pair in zip(all_labels, all_data)
                         if pair[0] in (4, 9)])

    labels = np.array(labels)
    for i in range(len(labels)):
        labels[i] = 1 if labels[i] == 4 else -1 
    ### here I changed every label 4 with 1 and 9 with -1

    data = np.array(data)
    dd = data.reshape((data.shape[0], data.shape[1]**2))
    dd = dd/255
    ddd = np.column_stack((np.ones(dd.shape[0]), dd))

    print(ddd.shape)
    #beta_hat = stochastic_gradient_descent(np.array(ddd), np.array(labels), epsilon=0.0001, l=1, step_size=0.01, max_steps=100000)
    beta_hat = gradient_descent(ddd, labels, epsilon=1e-3, l=1, step_size=0.2, max_steps=200)
    _valid = validation(ddd, labels, beta_hat)
    print("validation: {}%".format(_valid))

    #plt.imshow(data[42], cmap='Greys')
    #plt.title(str(labels[42]))
    #plt.show()

if __name__ == '__main__':
  args = parse_args()
  main(args)
