from ..convnet_operations import max_pooling, convolution, ReLU
import numpy as np

########## TEST ARRAYS ##########
filter_grid = np.array([[1, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1]])

grid = np.array([[1, 1, 1, 0, 0],
                 [0, 1, 1, 1, 0],
                 [0, 0, 1, 1, 1],
                 [0, 0, 1, 1, 0],
                 [0, 1, 1, 0, 0]])

relu =  np.array([[-1, -1, 1, 0, 0],
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

def test_convolution():
    convolution_matrix = convolution(D_grid, D_filter)
    assert convolution_matrix.shape == (5, 5)

def test_ReLU():
    relu_grid = ReLU(relu)
    reponse = np.array([[0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 1, 0, 0]])
    np.array_equal(relu_grid, reponse)

def test_max_pooling():
    max_pool = max_pooling(grid)
    assert max_pool.shape == (2, 2)
