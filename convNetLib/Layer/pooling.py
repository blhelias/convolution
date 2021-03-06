# -*- coding: utf-8 -*-
from convNetLib.Layer import Layer

import time

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
        start_time = time.time()
        input_shape = inputs.shape
        # Get output size
        H = (input_shape[0] - self.pool_size) // self.strides + 1
        W = (input_shape[1] - self.pool_size) // self.strides + 1
        D = input_shape[2]
        output_shape = (H, W, D)
        # Empty array initialization
        self.output = np.zeros(output_shape)
        self.index_memory = np.zeros(inputs.shape)
        for i in range(D):
            self.output[:, :, i] = self.max_pooling(inputs[:, :, i], H, W, i) # i = depth at each iteration

        self.compute_time = time.time() - start_time
        print(self)
        return self.output

    def backward(self, grad):
        """backward

        :param grad: np.array()
        """
        depth = grad.shape[2]
        output = np.zeros((grad.shape[0]*self.pool_size, grad.shape[1]*self.pool_size, depth))
        for i in range(depth): # On parcourt la profondeur de la grille
            reversed_pool = grad[:, :, i].repeat(self.pool_size, axis=0).repeat(self.pool_size, axis=1)
            output[:, :, i] = reversed_pool * self.index_memory[:, :, i]
        return output

    def max_pooling(self, feature_map, H, W, depth_i):
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

        for i in range(0, H):
            for j in range(0, W):

                region = feature_map[i*self.strides: i*self.strides + self.pool_size,
                                        j*self.strides: j*self.strides + self.pool_size]

                # get coordinates of the max value of the region
                max_pool_index = np.unravel_index(np.argmax(region, axis=None), region.shape)
                # Insert max value of the region in the corresponding output index
                output_map[i][j] = region[max_pool_index[0]][max_pool_index[1]]
                # Keep track of the max value index
                self.index_memory[i*self.strides+max_pool_index[0], j*self.strides+max_pool_index[1], depth_i] = 1

        return output_map

    def __repr__(self):
        return "[MAX POOL] output size --> {0} | {1} seconds ".format(self.output.shape, self.compute_time)

