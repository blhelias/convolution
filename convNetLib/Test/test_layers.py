from convNetLib.Layer import Convolution, Pooling, ReLU
import numpy as np
from PIL import Image

### IMG TEST ###
img = Image.open("convNetLib/data/lena_modif.jpeg")
img_array = np.asarray(np.uint8(img))
################
def test_convolution():
    conv1 = Convolution(2, (3, 3, 3))
    feature_map = conv1.forward(img_array)
    img1 = Image.fromarray(feature_map[:, :, 0])
    img1.show()
    img2 = Image.fromarray(feature_map[:, :, 1])
    img2.show()

def test_pooling():
    conv2 = Convolution(2, (3, 3, 3))
    feature_map = conv2.forward(img_array)
    pool = Pooling()
    res = pool.forward(feature_map)

def test_relu():
    conv3 = Convolution(10, (3, 3, 3))
    feature_map = conv3.forward(img_array)
    relu = ReLU()
    output_relu = relu.forward(feature_map)
    pool = Pooling()
    output_pool = pool.forward(feature_map)
