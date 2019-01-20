#-*- coding: utf_8 -*-
from convNetLib.Layer import Convolution, ReLU, Pooling
from convNetLib.ConvNet import ConvNet

from PIL import Image
import numpy as np


if __name__ == "__main__":

    ### IMG TEST ###
    img = Image.open("convNetLib/data/rabbit.jpeg")
    img_array = np.asarray(np.uint8(img))
    img.show()
    print("[INPUT] input size: {}".format(img_array.shape))
    ################

    # Initialize Layers
    pool = Pooling()
    conv = Convolution(5, (3, 3, 3))
    relu = ReLU()
    # END

    # forward pass CONV
    inputs = conv.forward(img_array)
    # forward pass ReLU
    inputs = relu.forward(inputs)
    for i in range(5):
        img = Image.fromarray(inputs[:, :, i])
        img.show()
        print(inputs[: ,:, i])

    # forward pass pooling
    inputs = pool.forward(inputs)

    # backward pass pooling
    grad = pool.backward(inputs)

    # backward pass reLU
    grad = relu.backward(grad)
    #model = ConvNet([
    #    Convolution(2, (3, 3, 3)),
    #    ReLU(),
    #    Pooling()
    #    ])

    #model.forward(img_array)

    #VGG16 = ConvNet([
    #        Convolution(64, (3, 3, 3)),
    #        ReLU(),
    #        Convolution(64, (3, 3, 64)),
    #        ReLU(),
    #        Pooling(),
    #        Convolution(128, (3, 3, 64)),
    #        ReLU(),
    #        Convolution(128, (3, 3, 128)),
    #        ReLU(),
    #        Pooling(),
    #        Convolution(256, (3, 3, 128)),
    #        ReLU(),
    #        Convolution(256, (3, 3, 256)),
    #        ReLU(),
    #        Convolution(256, (3, 3, 256)),
    #        ReLU(),
    #        Pooling(),
    #        Convolution(512, (3, 3, 256)),
    #        ReLU(),
    #        Convolution(512, (3, 3, 512)),
    #        ReLU(),
    #        Convolution(512, (3, 3, 512)),
    #        ReLU(),
    #        Pooling(),
    #        Convolution(512, (3, 3, 512)),
    #        ReLU(),
    #        Convolution(512, (3, 3, 512)),
    #        ReLU(),
    #        Convolution(512, (3, 3, 512)),
    #        ReLU(),
    #        Pooling(),
    #        # FC,
    #        # FC,
    #        # FC
    #    ])

    #VGG16.forward(img_array)
    ## Train example

    #for epoch in range n_epochs:
    #    pred = VGG16.forward(img_array) # replace img_array with batch of images
    #    epoch_loss = Loss() # MSE, RMSE, CROSS_entropy ....
    #    grad = loss.grad(predicted, batch.targets)
    #    VGG16.backward(grad)

