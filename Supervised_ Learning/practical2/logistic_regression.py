import numpy as np
import math


def sigmoid(s):
    h = 1 / (1+np.power(math.e, s))
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
        bb = bb + ( np.dot(X[i], Y[i]) ) * (1 - sigmoid(Y[i]* np.dot(X[i], beta)))
    return (-2*bb + np.dot(2*l, beta) )/ X.shape[0]


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

    index_array = np.arange(X.shape[0])
    np.random.shuffle(index_array)
    i = 0
    XX, ll, mean, sd = FeatureScaling(X, l)

    for s in range(max_steps):
        ridge = normalized_gradient(XX, Y, beta2, ll)

        if ( (np.linalg.norm( beta2- beta1) ) < epsilon* np.linalg.norm(beta2) ):
            break
        beta1 = beta2
        beta2 = beta2 - step_size * ridge;

    b2 = np.divide(beta2[1: beta2.shape[0]], sd)
    b1 = beta2[0] - sum(b2 * mean)

    return np.append(b1,b2)



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
    XX, ll, mean, sd = FeatureScaling(X, l)
    ll[0] = 0
    for s in range(max_steps):
        i = s % XX.shape[0]
        bb =( np.dot(XX[i], Y[i]) ) * ( 1 - sigmoid(Y[i]* np.dot(XX[i], beta2)))

        ridge=  (-2*bb + np.dot(2*l, beta2) )/ X.shape[0]

        if ( (np.linalg.norm( beta2- beta1) ) < epsilon* np.linalg.norm(beta2) ):
            break

        beta1 = beta2
        beta2 = beta2 - step_size * ridge;
    b2 = np.divide(beta2[1: beta2.shape[0]], sd)
    b1 = beta2[0] - sum(b2 * mean)

    return np.append(b1,b2)
