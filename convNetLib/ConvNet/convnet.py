"""NeuralNet behaves a lot like a layer itself, although
we are not going to make it one
"""

from convNetLib.Layer import Layer


class ConvNet:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def __repr__(self):
        raise NotImplementedError
