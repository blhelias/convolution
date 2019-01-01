# -*- coding: utf-8 -*-
# There are 3 main operations in convNet:
#  * Convolution
#  * Non-linearity
#  * Pooling

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
                if channels == 1:
                    # Element wise multiply
                    temp_grid = e_wise_product(start_coor, grid, filter_grid)
                    # Add the outputs
                    feature_map[i][j] = temp_grid.sum()
                else:
                    # Element wise multiply
                    temp_grid = e_wise_product(start_coor, grid[:, :, c],
                                               filter_grid[:, :, c])
                    # Add the outputs
                    feature_map[i][j] += temp_grid.sum()

    return feature_map

def max_pooling(feature_map, pool_size=2, strides=2):
    """max_pooling
    :param feature_map: np.array, matrix of Integer
    :param pool_size: Integer, size of the max pooling window
    :param strides: Integer, Factor by which to downscale

    --- Description
    Down sampling operation that prevent overfitting
    crop the image if image_size % pool_size != 0
    """
    fm_shape = feature_map.shape
    W = (fm_shape[1] - pool_size) // strides + 1
    H = (fm_shape[0] - pool_size) // strides + 1
    output_map = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            output_map[i][j] = np.max(feature_map[i*strides: i*strides + pool_size, j*strides: j*strides + pool_size])
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

def e_wise_product(coor, grid, filter_grid):
    """Une operation binaire qui pour deux matrices de memes dimensions,
    associe une autre matrice, de meme dimension, et ou chaque coefficient
    est le produit terme a terme des deux matrices.
    """
    filter_shape = filter_grid.shape
    new_grid = np.zeros(shape=filter_shape)

    for i in range(filter_shape[0]):
        for j in range(filter_shape[1]):
            new_grid[i][j] = grid[coor[0]+i][coor[1]+j] * filter_grid[i][j]

    return new_grid

if __name__ == "__main__":
    ########## TEST ARRAYS ##########
    filter_grid = np.array([[1, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1]])

    grid = np.array([[1, 1, 1, 0, 0],
                     [0, 1, 1, 1, 0],
                     [0, 0, 1, 1, 1],
                     [0, 0, 1, 1, 0],
                     [0, 1, 1, 0, 0]])

    test_relu =  np.array([[-1, -1, 1, 0, 0],
                           [0, -1, 1, -1, 0],
                           [0, 0, 1, -1, -1],
                           [0, 0, 1, -1, 0],
                           [0, -1, 1, 0, 0]])

    D_grid = np.array([[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)],
                        [(0, 0, 0), (0, 0, 0), (1, 2, 1), (2, 1, 0), (1, 0, 2), (2, 0, 0), (0, 0, 0)],
                        [(0, 0, 0), (1, 1, 1), (1, 1, 2), (0, 1, 2), (1, 2, 0), (0, 0, 2), (0, 0, 0)],
                        [(0, 0, 0), (0, 2, 2), (0, 1, 2), (2, 0, 0), (2, 2, 0), (1, 0, 0), (0, 0, 0)],
                        [(0, 0, 0), (1, 1, 2), (1, 1, 0), (1, 2 ,2), (0, 1, 0), (0, 0, 0), (0, 0, 0)],
                        [(0, 0, 0), (2, 0, 2), (1, 1, 2), (0, 1, 0), (0, 1, 2), (0, 0, 1), (0, 0, 0)],
                        [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]])

    D_filter = np.array([[(-1, -1, 0), (-1, -1, 1), (0, -1, -1)],
                          [(-1, 1 ,0), (-1, 1, 0), (-1, -1, 0)],
                          [(-1, 0, 1), (-1, 1, 1), (1, 0, 0)]])
    #################################

    convolution_matrix = convolution(D_grid, D_filter)
    relu = ReLU(test_relu)
    max_pooling = max_pooling(grid)

