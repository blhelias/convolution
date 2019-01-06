from convNetLib.Layers import Convolution, Pooling
import numpy as np
from PIL import Image


def test_convolution():
    conv1 = Convolution(2, (3, 3, 3))
    img = Image.open("convNetLib/data/lena_modif.jpeg")
    img_array = np.asarray(np.uint8(img))
    feature_map = conv1.forward(img_array)
    img1 = Image.fromarray(feature_map[:, :, 0])
    img1.show()
    img2 = Image.fromarray(feature_map[:, :, 1])
    img2.show()

def test_pooling():
    conv2 = Convolution(2, (3, 3, 3))
    img = Image.open("convNetLib/data/lena_modif.jpeg")
    img_array = np.asarray(np.uint8(img))
    feature_map = conv2.forward(img_array)
    pool = Pooling()
    res = pool.forward(feature_map)
