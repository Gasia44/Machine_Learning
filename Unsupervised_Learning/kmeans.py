import numpy as np
import math


def random_initialize(data_array, num_clusters):
    # TODO: Initialize cluster centers by sampling `num_clusters` points
    # uniformly from data_array.
    index_uni = np.random.choice(range(len(data_array)), size= num_clusters, replace=False)
    return data_array[np.array(index_uni)].tolist()


def plus_plus_initialize(data_array, num_clusters):
    # TODO: Initialize cluster centers using k-means++ algorithm.

    first_index= np.random.randint(0, len(data_array), 1)

    data_centers=[]
    data_centers.append(data_array[np.array(first_index)].tolist())

    for i in range(num_clusters-1):
        data_distance = np.ones(len(data_array))*math.inf

        for centers in data_centers:
            ind_data = 0
            for data in data_array:
                dist_temp = ((data[0] - centers[0][0])**2 + (data[1] - centers[0][1])**2)
                if data_distance[ind_data] > dist_temp:
                    data_distance[ind_data] = dist_temp
                ind_data+= 1

        data_distance = data_distance / np.sum(data_distance)
        index_uni = np.random.choice(range(len(data_distance)), size=1, replace=False, p = data_distance)

        data_centers.append(data_array[index_uni].tolist())

    flat_list = [item for sublist in data_centers for item in sublist]

    return flat_list


class KMeans(object):
    def __init__(self, num_mixtures):
        self.K = num_mixtures
        self.mus = []

    def initialize(self, data):
        """
        :param data: data, numpy 2-D array
        """
        # TODO: Initialize cluster centers
        # Hint: Use one of the function at the top of the file.
        self.mus = plus_plus_initialize(data, self.K)
        #self.mus = random_initialize(data, self.K)

    def fit(self, data):
        """
        :param data: data to fit, numpy 2-D array
        """
        # TODO: Initialize Mixtures, then run EM algorithm until it converges.
        self.initialize(data)

        old_predict = np.ones(len(data)) *math.inf
        r = np.zeros([len(data), self.K])

        while(np.sum((self.predict(data) - old_predict)) !=0):

            #initialize clusters
            r = np.zeros([len(data), self.K])

            for data_index in range(len(data)):
                data_distance = math.inf
                center_cluster = 0

                for center_index in range(len(self.mus)):
                    dist_temp  = np.sqrt((data[data_index][0] - self.mus[center_index][0]) **2 + (data[data_index][1] - self.mus[center_index][1])**2)

                    if(data_distance > dist_temp):
                        data_distance = dist_temp
                        center_cluster = center_index

                r[data_index][center_cluster] = 1

            old_predict = self.predict(data)

            for i in range(len(self.mus)):
                self.mus[i] = np.dot(r[:, i] , data) / np.sum(r[:, i])

        return(r)


    def predict(self, data):
        """
        Return index of the cluster the point is most likely to belong.
        :param data: data, numpy 2-D array
        :return: labels, numpy 1-D array
        """
        #r = np.zeros([len(data), self.K])
        data_cluster = np.zeros([len(data)])

        for data_index in range(len(data)):
            data_distance = math.inf
            center_cluster = 0

            for center_index in range(len(self.mus)):
                dist_temp = np.sqrt((data[data_index][0] - self.mus[center_index][0]) ** 2 + (data[data_index][1] - self.mus[center_index][1]) ** 2)

                if (data_distance > dist_temp):
                    data_distance = dist_temp
                    center_cluster = center_index

            #r[data_index][center_cluster] = 1
            data_cluster[data_index] = center_cluster
        return data_cluster

    def get_centers(self):
        return self.mus

