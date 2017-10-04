#!/usr/bin/env python3
"""
Homework3, template code
If you have any questions ask in #homeworks channel on slack.
"""
from __future__ import print_function
import numpy as np
import sys


def fit_ridge_regression(X, Y, l):
    """
    Calculates and returns analityc solution for ridge regression.

    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: value of beta (1 dimensional np.array)
    """
    # TODO: Implement fit_ridge_regression (same as previous homeworks)
    D = X.shape[1]  # dimension + 1
    beta = np.zeros(D)  # FIXME: ridge regression formula.
    beta = np.dot(np.dot( np.linalg.inv(np.dot(X.T, X) + l * np.identity(D) ), X.T) ,Y)
    return beta


def gradient_descent(X, Y, l, epsilon, step_size, max_steps):
    """
    Implement gradient descent using the value of the gradient
    divided by number of samples.

    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will
        terminate.
    :return: value of beta (1 dimensional np.array)
    """
    # TODO: Implement iterations.
    # Use normalized_gradient to calculate the gradient
    beta = np.zeros(X.shape[1])
    h = np.dot(X, beta) - Y ;
    for s in range(max_steps):
        loss1 = 0;
        loss2= loss(X, Y, beta);
        rloss= ridge_loss_gradient(X,Y,beta,l)
        if (np.absolute(loss2 - loss1) < epsilon):
            break
        loss1 = loss2
        beta = beta - step_size * rloss;

    return beta


def ridge_loss_gradient(X, Y, beta, l):
    """
    This function calculates the gradient for ridge regression for
    parameter values beta.

    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param beta: value of beta (1 dimensional np.array)
    :param l: regularization parameter lambda
    :return: normalized gradient, i.e. gradient normalized according to data
    """
    bb=0
    # TODO: Implement
    for i in range(Y.shape[0]):
        bb = bb + np.dot((Y[i] - np.dot(beta.T, X[i])), X[i])
    return (-2*bb/ X.shape[0]) + np.dot(2*l, beta)


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


def d_dimensional_comparison(d, beta_star, num_points, sigma, l=1):
    # Generate data, no need to touch this code.
    beta_star = np.array(beta_star)
    X_list = [np.random.uniform(-1, 1, num_points) for _ in range(d)]
    X = np.column_stack(X_list)
    X = np.column_stack((np.ones(num_points), X))
    Y = np.random.normal(X.dot(beta_star), sigma)

    # Calculate analytic and gradient descent beta hats.
    beta_hat_analytic = fit_ridge_regression(X, Y, l=l)
    beta_hat_grad = gradient_descent(X, Y, l=l, epsilon=1e-8, step_size=1e-2,
                                     max_steps=10000)

    # Testing code for your esimates.
    if np.linalg.norm(beta_star - beta_hat_analytic) > 1.:
        print('Your analytical betas is too far apart from beta star')
        print('Analytical: ', beta_hat_analytic)
        print('Beta star: ', beta_star)
        sys.exit(1)

    if np.linalg.norm(beta_hat_grad - beta_hat_analytic) > 1e-4:
        print('Your gradient descent beta is too far apart from analytical '
              'solution')
        print('Beta gradient: ', beta_hat_grad)
        print('Analytical: ', beta_hat_analytic)
        sys.exit(1)

    l_a = loss(X, Y, beta_hat_analytic)
    l_gd = loss(X, Y, beta_hat_grad)
    if abs((l_a - l_gd) / l_a) > 1e-8:
        print('Your gradient and analytical losses are too far apart')
        print('analytical loss:', l_a)
        print('gradient loss:', l_gd)
        sys.exit(1)

    print('Passed')


if __name__ == '__main__':
    # Fist test the signature of your gradient descent function.
    beta_est = gradient_descent(np.array([[1, 2], [1, 3], [1, 4], [1, 5]]),
                                np.array([2, 3, 4, 5.01]),
                                l=0,
                                epsilon=1e-4,
                                step_size=1e-3,
                                max_steps=2)
    assert beta_est.shape == (2,)
    # Call comparison function with the given 5-dimensional beta (b0, ..., b5)
    beta5d = [1.5, 2.2, 3.5, 4.4, 1.1, 3.9]
    d_dimensional_comparison(5, beta5d, 200, 2, l=0.)
