#!/usr/bin/env python3
"""
Test your regression.py functions
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from regression import fit_ridge_regression
from regression import gradient_descent
from regression import stochastic_gradient_descent


def loss(X, Y, beta):
    return sum((X.dot(beta) - Y) ** 2) / X.shape[0]


def d_dimensional_comparison(d, beta_star, num_points, sigma, l=1):
    beta_star = np.array(beta_star)

    X_list = [np.random.uniform(-20, 80, num_points) for _ in range(d)]
    X = np.column_stack(X_list)
    X = np.column_stack((np.ones(num_points), X))
    Y = np.random.normal(X.dot(beta_star), sigma)

    beta_hat_ridge = fit_ridge_regression(X, Y, l=l)
    beta_hat_grad = gradient_descent(X, Y, l=l, epsilon=1e-8, step_size=1e-2,
                                     max_steps=10000)
    beta_hat_stoc = stochastic_gradient_descent(X, Y, epsilon=1e-8, l=1, step_size=0.001,
                                    max_steps=500000)


    print('ridge beta', beta_hat_ridge)
    print('grad beta', beta_hat_grad)
    print('beta_hat_stoc', beta_hat_stoc)
    assert loss(X, Y, beta_hat_grad) < 1.25 * loss(X, Y, beta_hat_ridge)
    print('Passed')

if __name__ == '__main__':
    beta5d = [15, 2.2, 3.5, 4.4, 1.1, 3.9]
    beta_est = gradient_descent(np.array([[1, 2], [1, 3], [1, 4], [1, 5]]),
                                np.array([2, 3, 4, 5.01]),
                                max_steps=2)
    assert beta_est.shape == (2,)
    d_dimensional_comparison(5, beta5d, 200, 2, l=1)
