# -*- coding: utf-8 -*-

import numpy as np

from convNetLib.Layer import Layer


class Activation(Layer):
    def __init__(self, f, f_prime):
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs):
        self.inputs = inputs
        return self.f(inputs)

    def backward(self, grad):
        return self.f_prime(self.inputs) * grad



