import numpy as np
import copy

class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initialize = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initialize, self.bias_initializer)
        self.layers.append(layer)
    
    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return self.loss_layer.forward(input_tensor, self.label_tensor)

    def backward(self):
        # _, label_tensor = self.data_layer.next()
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        # return error_tensor

    def train(self, iterations):
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor


    



    