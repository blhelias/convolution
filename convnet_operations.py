# There are 3 main operations in convNet:
#  * Convolution
#  * Non-linearity
#  * Pooling

import numpy as np


def convolution(grid, filter_grid):
    """Apply filter with weights to generate feature map

    TODO: Allow 3 channels (RGB)
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

    feature_map = np.zeros(shape=(feature_map_size))
    # Slide the filter over the input image
    for i in range(feature_map_size[0]):
        for j in range(feature_map_size[1]):
            start_coor = (i, j)
            # Element wise multiply
            temp_grid = e_wise_product(start_coor, grid, filter_grid)
            # Add the outputs
            feature_map[i][j] = temp_grid.sum()

    return feature_map

def max_pooling(pool_size=2, strides=None):
    """max_pooling

    :param pool_size: Integer, size of the max pooling window
    :param strides: Integer, Factor by which to downscale

    --- Description
    Down sampling operation that
    """

    return

def ReLU(feature_map):
    """Rectified Linear Unit is an activation fonction that
    replace all negative values of a matrix by 0
    """
    fm_shape = feature_map.shape
    for i in range(fm_shape[0]):
        for j in range(fm_shape[1]):
            if feature_map[i][j] < 0:
                feature_map[i][j] = 0
    return feature_map

def e_wise_product(coor, grid, filter_grid):
    """Une opération binaire qui pour deux matrices de mêmes dimensions,
    associe une autre matrice, de même dimension, et où chaque coefficient
    est le produit terme à terme des deux matrices.
    """
    filter_shape = filter_grid.shape
    new_grid = np.zeros(shape=filter_shape)
    # We could use the function multiply from NumPy library
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
    ##################################

    convolution_matrix = convolution(grid, filter_grid)
    relu = ReLU(test_relu)
