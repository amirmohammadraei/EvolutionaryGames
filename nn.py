import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):
        self.w1 = np.random.randn(layer_sizes[0], layer_sizes[1])
        self.b1 = np.random.randn(layer_sizes[1], 1)
        self.w2 = np.random.randn(layer_sizes[1], layer_sizes[2])
        self.b2 = np.random.randn(layer_sizes[2], 1)

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        z1 = self.w1.T @ x + self.b1
        a1 = self.activation(z1)
        z2 = self.w2.T @ a1 + self.b2
        a2 = self.activation(z2)
        return a2
