from ..convnet_operations import max_pooling, convolution, ReLU
import numpy as np

#img_shape = (100, 100, 3)
#img_array = np.zeros(img_shape)

#for i in range(img_shape[0]):
#    for j in range(img_shape[1]):
#        for k in range(img_shape[2]):
#            a = rd.randint(0, 255)
#            b = rd.randint(0, 255)
#            c = rd.randint(0, 255)
#            img_array[i][j] = (a, b, c)

# img = Image.fromarray(np.uint8(img_array))
# img.show()
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
