#!/usr/bin/env python3
"""
Run regression on apartment data.
"""
from __future__ import print_function
import argparse
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import getpass


def parse_args(*argument_array):
    parser = argparse.ArgumentParser()
    parser.set_defaults(function=main)
    parser.add_argument('--user', default=getpass.getuser(),
                        help='Override system username with something else to '
                             'be include in the output file.')
    subs = parser.add_subparsers()
    test_parser = subs.add_parser('test')
    test_parser.set_defaults(function=test_function_signatures)
    parser.add_argument('--csv', default='yerevan_april_9.csv.gz',
                        help='CSV file with the apartment data.')
    args = parser.parse_args(*argument_array)
    return args


def featurize(apartment):
    """
    :param apartment: Apartment DataFrame row (a dictionary like object)
    :return: (x, y) tuple, where x is a numpy vector, and y is a number
    """
    col =['condition', 'num_rooms', 'area', 'num_bathrooms', 'floor', 'ceiling_height', 'max_floor', ]
    a= pd.DataFrame(apartment[col])
    di = {'zero condition' : 0, 'good': 1, 'newly repaired': 2}
    a = a.replace({"condition": di})
    x = np.array(a.values)
    x = np.column_stack((1, x))
    return x, apartment['price']

def poly_featurize(apartment, degree=2):
    """
    :param apartment: Apartment DataFrame row (a dictionary like object)
    :return: (x, y) tuple, where x is a numpy vector, and y is a number
    """
    x, y = featurize(apartment)
    x, y = featurize(apartment)
    #poly_x = # TODO: use itertools.product to get higher degree elements.
    return poly_x, y


def fit_ridge_regression(X, Y, l=0.1):
    """
    :param X: A numpy matrix, where each row is a data element (X)
    :param Y: A numpy vector of responses for each of the rows (y)
    :param l: ridge variable
    :return: A vector containing the hyperplane equation (beta)
    """
    D = X.shape[1]  # dimension + 1
    beta = np.zeros(D)  # FIXME: ridge regression formula.
    beta = np.dot(np.dot( np.linalg.inv(np.dot(X.T, X) + l * np.identity(D) ), X.T) ,Y)
    return beta


def cross_validate(X, Y, fitter, folds=5):
    """
    :param X: A numpy matrix, where each row is a data element (X)
    :param Y: A numpy vector of responses for each of the rows (y)
    :param fitter: A function that takes X, Y as parameters and returns beta
    :param folds: number of cross validation folds (parts)
    :return: list of corss-validation scores
    """
    scores = []
    # TODO: Divide X, Y into `folds` parts (e.g. 5)
    index_array = np.arange(X.shape[0])
    np.random.shuffle(index_array)

    #ex: if fold=5 , we need to have 5 parts, each part has fold_part elements
    fold_part =int(np.floor(X.shape[0] / folds))

    scores = np.zeros(folds)
    for i in range(folds):

        row_idx_train = index_array[ np.r_[0 : i*fold_part - 0, (i+1) * fold_part: X.shape[0]] ]
        train_X = X[row_idx_train[:,],:]
        train_Y = Y[row_idx_train]

        beta = fitter(train_X, train_Y, l=500000)
        print(beta)
        row_idx_test = index_array[ i*fold_part : (i+1) * fold_part ]
        test_X = X[row_idx_test[:,],:]
        test_Y = Y[row_idx_test]

        scores[i] = np.sqrt( (np.sum(test_Y - np.dot( test_X, beta))**2) / test_Y.shape[0] )
        print("scores", scores[i])
        # TODO: train on the rest
        # TODO: Add corresponding score to scores
        pass
    return scores


def my_featurize(apartment):
    """
    This is the function we will use for scoring your implmentation.
    :param apartment: apartment row
    :return: (x, y) pair where x is feature vector, y is the response variable.
    """
    col =np.array([1, 2, 0, 0, 0, 0, 0, 0 ])
    a= pd.DataFrame(apartment[col])
    if(apartment.get('condition')== 'good'):
        col[1] =1
    else:
        if(apartment.get('condition')== 'zero condition'):
            col[1] =  0
    col[2] =apartment.get('num_rooms')
    col[3] =apartment.get('area')
    col[4] =apartment.get('num_bathrooms')
    col[5] =apartment.get('floor')
    col[6] =apartment.get('ceiling_height')
    col[7] =apartment.get('max_floor')

    return col, apartment['price']


def my_beta():
    """
    :return: beta_hat that you estimate.
    """
    #return np.array([-5.43309139, 47.00607968, -31.36861761, 1057.83381763, 41.49706981, 9.75755089, 4.73004373, 12.81618895])

    return np.array([-7.37979572, 44.36496289, -29.21068086, 1064.29280332, 39.99747004, 6.97372412, -1.29671082, 1.5024365 ])


def main(args):

    df = pd.read_csv(args.csv)
    #print(df)
    #di = {1: "A", 2: "B"}
    flag = 0
    n= df.shape[0]
    for i in range(n):
        xx, yy = featurize(df.iloc[[i]])
        if flag == 0:
            X=xx
            Y=yy
            flag = 1
        else:
            X = np.vstack((X,xx))
            Y = np.vstack((Y,yy))

    beta = fit_ridge_regression(X, Y)
    #print(beta)

    scores = cross_validate(X, Y, fit_ridge_regression)
    print(np.mean(scores))


def test_function_signatures(args):
    apt = pd.Series({'price': 65000.0, 'condition': 'good', 'district': 'Center', 'max_floor': 9, 'street': 'Vardanants St', 'num_rooms': 3, 'region': 'Yerevan', 'area': 80.0, 'url': 'http://www.myrealty.am/en/item/24032/3-senyakanoc-bnakaran-vacharq-Yerevan-Center', 'num_bathrooms': 1, 'building_type': 'panel', 'floor': 4, 'ceiling_height': 2.7999999999999998})  # noqa
    #print(apt.get('condition'))
    x, y = my_featurize(apt)
    beta = my_beta()

    assert type(y) == float
    assert len(x.shape) == 1
    assert x.shape == beta.shape

if __name__ == '__main__':
    args = parse_args()
    args.function(args)
