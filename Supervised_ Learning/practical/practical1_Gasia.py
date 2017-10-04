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

    return col, apartment


def my_beta():
    """
    :return: beta_hat that you estimate.
    """
    #return np.array([-5.43309139, 47.00607968, -31.36861761, 1057.83381763, 41.49706981, 9.75755089, 4.73004373, 12.81618895])
    return np.array([-7.37979572, 44.36496289, -29.21068086, 1064.29280332, 39.99747004, 6.97372412, -1.29671082, 1.5024365 ])
