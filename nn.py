import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):

        self.w1 = np.random.normal(size=(layer_sizes[1], layer_sizes[0]))
        self.b1 = np.zeros((layer_sizes[1], 1))
        self.w2 = np.random.normal(size=(layer_sizes[2], layer_sizes[1]))
        self.b2 = np.zeros((layer_sizes[2], 1))
        self.y = 0

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        z1 = np.dot(self.w1, x) + self.b1
        z1 = self.activation(z1)
        z2 = np.dot(self.w2, z1) + self.b2
        z2 = self.activation(z2)
        return z2
