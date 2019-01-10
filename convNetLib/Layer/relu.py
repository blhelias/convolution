# -*- coding: utf-8 -*-

import numpy as np

from convNetLib.Layer import Activation


class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)

def relu(x):
    """relu

    :param feature_map: np.array()

    --- DEscription
    Rectified Linear Unit is an activation fonction that
    replace all negative values of a matrix by 0.
    --> z = max(0, z)
    """
    x_shape = x.shape
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            for k in range(x_shape[2]):
                x[i, j, k] = max(0, x[i, j, k])
    print("[ReLU] output shape: ", x_shape)
    return x

def relu_prime(x):
    """relu_prime

    :param x: np.array()

    -- Descripion
    dy/dx = 1 if    x > 0
            0  else  x <= 0
    """
    x_shape = x.shape
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            for k in range(x_shape[2]):
                if x[i, j, k] > 0:
                    x[i, j, k] = 1
                else:
                    x[i, j, k] = 0
    return x



