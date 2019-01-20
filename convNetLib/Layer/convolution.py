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
        # Initialize timer
        start_time = time.time()

        # Add zero padding to preserve output size
        inputs = np.pad(inputs, ((1, 1), (1, 1), (0, 0)), mode='constant')

        # Retrieve inputs shape
        input_shape = inputs.shape

        # Compute dimension of the output
        H = (input_shape[0] - self.kernel_shape[0]) // self.strides[0] + 1
        W = (input_shape[1] - self.kernel_shape[1]) // self.strides[1] + 1
        D = self.n_filters

        # Save output shape and initialize empty output filled with zeros
        self.output_shape = (H, W, D)
        output = np.zeros(self.output_shape)

        # Iterate over depth of the inputs
        for fltr_i in range(D):
            # Apply convolution
            output[:, :, fltr_i] = self.convolution(inputs,
                                                    self.filters[fltr_i],
                                                    H, W)

        # Save processed time
        self.compute_time = time.time() - start_time
        print(self)
        return output

    def backward(self, grad):
        raise NotImplementedError

    def convolution(self, inputs, fltr, H, W):
        """convolution

        :param inputs: np.array()
        :param fltr: np.array()
        :param H: Height of the feature map
        :parm W: Width of the feature map

        --- Description
        Apply filter with weights to generate feature map
        """
        # Retrieve input shape
        input_shape = inputs.shape

        # check that the filter size is smaller than the image
        assert self.kernel_shape[0] <= input_shape[0] and self.kernel_shape[1] <= input_shape[1]
        # check dimensions of the input
        assert len(input_shape) == 3
        # Check that filter and imput have the same depth
        assert input_shape[2] == self.kernel_shape[2]

        # compute the size of the feature map
        feature_map_size = (H, W)
        depth = input_shape[2]

        for d in range(depth):
            # Initialize empty output filled with zeros
            feature_map = np.zeros(shape=(feature_map_size))

            # Slide the filter over the input image
            for i in range(feature_map_size[0]):
                for j in range(feature_map_size[1]):
                    # Element wise multiplication
                    inputs_chunck = inputs[i: i + self.kernel_shape[0],
                                           j: j + self.kernel_shape[1],
                                           d]

                    # Add the outputs
                    feature_map[i][j] = np.tensordot(fltr[:, :, d],
                                                     inputs_chunck,
                                                     axes=((0,1),(0,1)))

        # Apply normalization
        feature_map = feature_map // (feature_map.max() / 255)
        return feature_map

    def __repr__(self):
        return "[CONV 2D] -> {0} | {1} seconds ".format(self.output_shape, self.compute_time)
