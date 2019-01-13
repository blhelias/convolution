from convNetLib.Layer import Convolution, ReLU, Pooling
from convNetLib.ConvNet import ConvNet

from PIL import Image
import numpy as np


if __name__ == "__main__":
    ### IMG TEST ###
    img = Image.open("convNetLib/data/sunset.png")
    img_array = np.asarray(np.uint8(img))
    img.show()
    print("[INPUT] input size: {}".format(img_array.shape))
    ################
    model = ConvNet([
            Convolution(64, (3, 3, 3)),
            ReLU(),
            Convolution(64, (3, 3, 64)),
            ReLU(),
            Pooling()
        ])
    model.forward(img_array)
