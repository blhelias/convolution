from convNetLib.Layer import Convolution, ReLU, Pooling
from PIL import Image
import numpy as np


if __name__ == "__main__":
    ### IMG TEST ###
    img = Image.open("convNetLib/data/sunset.png")
    img_array = np.asarray(np.uint8(img))
    img.show()
    print("[INPUT] input size: {}".format(img_array.shape))
    ################
    conv1 = Convolution(64, (3, 3, 3))
    feature_map = conv1.forward(img_array)
    relu1 = ReLU()
    output_relu = relu1.forward(feature_map)
    #for fltr_img in range(output_relu.shape[2]):
    #    fltr_img_pil = Image.fromarray(output_relu[:, :, fltr_img])
    #    fltr_img_pil.show()
    pool = Pooling()
    output_pool = pool.forward(output_relu)
    #for pool_img in range(output_pool.shape[2]):
    #    pool_img_pil = Image.fromarray(output_pool[:, :, pool_img])
    #    pool_img_pil.show()
