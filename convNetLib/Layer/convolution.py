#-*- coding: utf-8 -*-
from convNetLib.Layer import Layer
import numpy as np

import time


class Convolution(Layer):
    def __init__(self, n_filters, kernel_shape, strides=(1, 1)):
        super().__init__()
        self.n_filters = n_filters
        self.kernel_shape = kernel_shape # (Height, Width, Channels)
        self.strides = strides
        # initialize filters (n_filters, kernel_shape)
        self.filters = np.random.randint(-1, 2, (self.n_filters,
                                                 self.kernel_shape[0],
                                                 self.kernel_shape[1],
                                                 self.kernel_shape[2]))

    def forward(self, inputs):
        """forward

        :param inputs: np.array

        -- Description
        The forward operation
        """
        start_time = time.time()

        inputs = np.pad(inputs, ((1, 1), (1, 1), (0, 0)), mode='constant')
        input_shape = inputs.shape
        H = (input_shape[0] - self.kernel_shape[0]) // self.strides[0] + 1
        W = (input_shape[1] - self.kernel_shape[1]) // self.strides[1] + 1
        D = self.n_filters

        self.output_shape = (H, W, D)
        output = np.zeros(output_shape)
        for fltr_i in range(D):
            output[:, :, fltr_i] = self._convolution(inputs, self.filters[fltr_i], H, W)
        self.compute_time = time.time() - start_time
        print(self)
        return output


    def _convolution(self, inputs, fltr, H, W):
        """convolution

        :param inputs: np.array()
        :param fltr: np.array()
        :param H: Height of the feature map
        :parm W: Width of the feature map

        --- Description
        Apply filter with weights to generate feature map
        """
        input_shape = inputs.shape

        # check that the filter size is smaller than the image
        assert self.kernel_shape[0] <= input_shape[0] and self.kernel_shape[1] <= input_shape[1]
        # check dimensions of the input
        assert len(input_shape) == 3
        # Check that filter and imput have the same depth
        assert input_shape[2] == self.kernel_shape[2]
        # determine the size of the feature map

        feature_map_size = (H, W)
        depth = input_shape[2]

        for d in range(depth):

            feature_map = np.zeros(shape=(feature_map_size))

            # Slide the filter over the input image
            for i in range(feature_map_size[0]):
                for j in range(feature_map_size[1]):
                    # Element wise multiplication
                    inputs_chunck = inputs[i: i + self.kernel_shape[0],
                                           j: j + self.kernel_shape[1],
                                           d]
                    # m = np.multiply(fltr[:, :, d], inputs_chunck)
                    # Add the outputs
                    # feature_map[i][j] += m.sum()
                    feature_map[i][j] = np.tensordot(fltr[:, :, d], inputs_chunck, axes=((0,1),(0,1)))

        # print("[CONV] output size --> {0}".format(feature_map.shape))
        return feature_map

    def __repr__(self):
        return "[CONV 2D] -> {0} | {1} seconds ".format(self.output_shape, self.compute_time)
