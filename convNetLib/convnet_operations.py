# -*- coding: utf-8 -*-
# There are 3 main operations in convNet:
#  * Convolution
#  * Non-linearity
#  * Pooling

from ..Layers import Layers

import numpy as np


def convolution(grid, filter_grid, stride=1):
    """convolution

    :param grid: np.array()
    :param filter_grid: np.array()

   --- Description
    Apply filter with weights to generate feature map

    TODO: allow user to change stride (K)
    """
    filter_shape = filter_grid.shape
    grid_shape = grid.shape
    # check if the filter size is smaller than the image
    assert filter_shape[0] <= grid_shape[0] and filter_shape[1] <= grid_shape[1]
    # determine the size of the feature map
    feature_map_size = (((grid_shape[0] + 1) - filter_shape[0]),
                        ((grid_shape[1] + 1) - filter_shape[1]))

    starting_row = 0
    starting_col = 0

    if len(grid_shape) == 2:
        channels = 1
    elif len(grid_shape) < 2:
        raise ValueError("Wrong Dimensions: D < 2")
    else:
        assert grid_shape[2] == filter_shape[2]
        channels = grid_shape[2]

    for c in range(channels):
        feature_map = np.zeros(shape=(feature_map_size))
        # Slide the filter over the input image
        for i in range(feature_map_size[0]):
            for j in range(feature_map_size[1]):
                start_coor = (i, j)
                # Check if it's a grey scale image
                if channels == 1:
                    # Element wise multiplication
                    grid_chunck = grid[start_coor[0]:start_coor[0]+filter_shape[0],
                                       start_coor[1]:start_coor[1]+filter_shape[1]]
                    temp_grid = np.multiply(filter_grid, grid_chunck)
                    # Add the outputs
                    feature_map[i][j] = temp_grid.sum()
                # It's an RGB image
                else:
                    # Element wise multiplication
                    grid_chunck = grid[start_coor[0]:start_coor[0]+filter_shape[0],
                                       start_coor[1]:start_coor[1]+filter_shape[1],
                                       c]
                    temp_grid = np.multiply(filter_grid[:, :, c], grid_chunck)
                    # Add the outputs
                    feature_map[i][j] += temp_grid.sum()

    print("[CONV] output size --> {0}".format(feature_map.shape))
    return feature_map

def max_pooling(feature_map, pool_size=2, strides=2):
    """max_pooling
    :param feature_map: np.array, matrix of Integer
    :param pool_size: Integer, size of the max pooling window
    :param strides: Integer, Factor by which to downscale

    --- Description
    Down sampling operation that prevent overfitting

    TODO: find other solution rather than crop the image if image_size % pool_size != 0
    """
    fm_shape = feature_map.shape
    W = (fm_shape[1] - pool_size) // strides + 1
    H = (fm_shape[0] - pool_size) // strides + 1
    output_map = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            max_pool_value = np.max(feature_map[i*strides: i*strides + pool_size,
                                    j*strides: j*strides + pool_size])
            output_map[i][j] = max_pool_value

    print("[MAX POOL] output size --> {0} ".format(output_map.shape))
    return output_map

def ReLU(feature_map):
    """ReLU

    :param feature_map: np.array()

    --- DEscription
    Rectified Linear Unit is an activation fonction that
    replace all negative values of a matrix by 0.
    --> z = max(0, z)
    """
    fm_shape = feature_map.shape
    for i in range(fm_shape[0]):
        for j in range(fm_shape[1]):
            feature_map[i][j] = max(0, feature_map[i][j])
    return feature_map


if __name__ == "__main__":
    import timeit
    # start timer
    start = timeit.default_timer()

    D_filter = np.array([[(-1, -1, 0), (-1, -1, 1), (0, -1, -1)],
                         [(-1, 1 ,0), (-1, 1, 0), (-1, -1, 0)],
                         [(-1, 0, 1), (-1, 1, 1), (1, 0, 0)]])
    import random as rd
    from PIL import Image
    # Read image
    img = Image.open("data/lena_modif.jpeg")
    img.show()
    img_array = np.asarray(img)
    print("[INPUT] image size --> {0}".format(img_array.shape))
    # Convolution layer
    convolution_matrix = convolution(img_array, D_filter)
    img = Image.fromarray(np.uint8(convolution_matrix)).show()
    # ReLU activation
    relu = ReLU(convolution_matrix)
    img = Image.fromarray(np.uint8(relu)).show()
    # Max pooling layer
    max_pooling = max_pooling(convolution_matrix)
    img = Image.fromarray(np.uint8(max_pooling)).show()
    # stop timer
    stop = timeit.default_timer()
    print('Time: ', stop - start)
