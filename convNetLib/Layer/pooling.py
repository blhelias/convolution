# -*- coding: utf-8 -*-
from convNetLib.Layer import Layer

import numpy as np


class Pooling(Layer):
    """Pooling"""
    def __init__(self, pool_size=2, strides=2):
        super().__init__()
        self.pool_size = pool_size
        self.strides = strides

    def forward(self, inputs):
        """forward

        :param inputs: np.array()
        """
        input_shape = inputs.shape
        # Get output size
        H = (input_shape[0] - self.pool_size) // self.strides + 1
        W = (input_shape[1] - self.pool_size) // self.strides + 1
        D = input_shape[2]
        output_shape = (H, W, D)
        # Empty array initialization
        output = np.zeros(output_shape)
        for i in range(D):
            output[:, :, i] = self.max_pooling(inputs[:, :, i], H, W)
        print("[MAX POOL] output size: ", output.shape)
        return output

    def max_pooling(self, feature_map, H, W):
        """max_pooling
        :param feature_map: np.array, matrix of Integer
        :param pool_size: Integer, size of the max pooling window
        :param strides: Integer, Factor by which to downscale

        --- Description
        Down sampling operation that prevent overfitting

        TODO: find other solution rather than crop the image if image_size % pool_size != 0
        """
        fm_shape = feature_map.shape
        output_map = np.zeros((H, W))
        for i in range(H):
            for j in range(W):
                max_pool_value = np.max(feature_map[i*self.strides: i*self.strides + self.pool_size,
                                        j*self.strides: j*self.strides + self.pool_size])
                output_map[i][j] = max_pool_value

        # print("[MAX POOL] output size --> {0} ".format(output_map.shape))
        return output_map
