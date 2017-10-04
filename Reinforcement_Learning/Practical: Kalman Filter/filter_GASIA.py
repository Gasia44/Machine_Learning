#!/usr/bin/env python3
"""
Write a Kalman Filter
"""
from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from numpy.linalg import norm
from numpy import matmul


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', help='File with the signal to be filtered.',
                        default='noisy_1d.csv')
    args = parser.parse_args(*argument_array)
    return args


class KalmanFilter:

    def __init__(self, measurement_sigma, process_sigma, covariance_prior,
                 location_prior):
        self.R = measurement_sigma
        self.process_sigma = process_sigma

        F = np.identity(2)
        self.x = (np.dot(F, location_prior))
        self.p =np.random.normal(location_prior, covariance_prior)

    def step(self, observation, delta_t=1.):
        H = np.array([[1, 0]])
        F = np.array([[1, delta_t], [0, 1]])

        G = np.array([[0.5 * delta_t**2], [delta_t]])
        Q = matmul(G , G.T )* (self.process_sigma ** 2)

        next_prediction = np.zeros(observation.shape)  # FIXME
        x_kk = self.x
        p_kk = self.p

        x_kk_ = np.dot(F, x_kk)
        p_kk_ =matmul( matmul(F,p_kk), F.T) + Q

        y_k = observation - matmul(H, x_kk)
        S = self.R + matmul(matmul(H , p_kk_ ), H.T)
        k = matmul(matmul(p_kk_, H.T ), np.linalg.inv(S))

        x_kk = x_kk_ + (matmul(k , y_k))
        p_kk = (np.identity(2) - matmul(matmul(k,H), p_kk_))
        y_kk = observation - matmul(H, x_kk)

        self.x = x_kk
        self.p = p_kk

        next_prediction = x_kk_[0]

        return next_prediction


def main(args):
    df = pd.read_csv(args.csv)
    unfiltered = [np.array([row['XX']]) for i, row in df.iterrows()]
    kf = KalmanFilter(3000, 0.001, np.identity(2), np.array([[0],[0]]))
    filtered = [kf.step(x) for x in unfiltered]
    plt.plot(unfiltered, label = 'unfiltered' )
    plt.plot(filtered, label = 'filtered')
    plt.legend(fontsize=14)
    plt.show()


if __name__ == '__main__':
  args = parse_args()
  main(args)
