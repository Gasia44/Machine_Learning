import numpy as np
from kmeans import KMeans
from scipy.stats import multivariate_normal
import math

def multi_normal_pdf(x, mean, covariance):
    """
    Evaluates Multivariate Gaussian Distribution density function
    :param x: location where to evaluate the density function
    :param mean: Center of the Gaussian Distribution
    :param covariance: Covariance of the Gaussian Distribution
    :return: density function evaluated at point x
    """
    var = multivariate_normal(mean=mean, cov=covariance)
    return var.pdf(x)


class GaussianMixtureModel(object):
    def __init__(self, num_mixtures):
        self.K = num_mixtures
        self.centers = []  # List of centers
        self.weights = []  # List of weights
        self.covariances = []  # List of covariances
        self.r = None  # Matrix of responsibilities, i.e. gamma

    def initialize(self, data):
        """
        :param data: data, numpy 2-D array
        """
        # TODO: Initialize cluster centers, weights, and covariances
        # Hint: Use K-means
        km = KMeans(self.K)
        rr = km.fit(data)

        self.weights = np.sum(rr,axis=0) / np.sum(rr)
        self.centers= km.get_centers()

        dat_cov = []
        self.covariances = np.zeros([self.K, 2, 2])
        for j in range(rr.shape[1]):

            dat_cov = data[rr[:, j] == 1]
            self.covariances[j] = np.cov(dat_cov,rowvar = False)

        self.r = np.zeros([len(data), self.K])


    def fit(self, data, max_iter=150, precision=1e-16):
        """
        :param data: data to fit, numpy 2-D array
        """
        # TODO: Initialize Mixtures, then run EM algorithm until it converges.
        self.initialize(data)

        p_sum_old = np.sum(np.log(np.sum(self.r, axis=1)))

        for iteration in range(1, max_iter + 1):

            #print('----------',iteration)

            for k in range(self.K):
                self.r[:, k] = multi_normal_pdf(data, self.centers[k], self.covariances[k]) * self.weights[k]

            pp = self.r
            print(np.sum(np.log(np.sum(self.r, axis = 1))))
            self.r = self.r / self.r.sum(axis=1, keepdims=True)

            N_k = np.sum(self.r, axis = 0)

            self.weights =  N_k/ np.sum(N_k)

            for i in range(self.K):
                self.centers[i] = np.dot(self.r[:, i] , data) / N_k[i]
                self.covariances[i]=np.dot((self.r[:, i] * (data - self.centers[i]).T), (data - self.centers[i])) / N_k[i]


            p_sum = np.sum(np.log(np.sum(self.r, axis = 1)))

            if np.abs(p_sum - p_sum_old) < precision:
                break

            p_sum_old = p_sum

    def get_centers(self):
        return self.centers

    def get_covariances(self):
        return self.covariances

    def get_weights(self):
        return self.weights

    def predict_cluster(self, data):

        """
        Return index of the clusters that each point is most likely to belong.
        :param data: data, numpy 2-D array
        :return: labels, numpy 1-D array
        """
        return np.argmax(self.r, axis=1)