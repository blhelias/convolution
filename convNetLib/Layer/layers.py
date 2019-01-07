# -*- coding: utf-8 -*-

import numpy as np


class Layer:
    def __init__(self):
        pass

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


