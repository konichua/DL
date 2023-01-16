from Layers import Base
import copy


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = []          # used in method train
        self.layers = []        # used in method append_layer
        self.data_layer = None  # init as net.data_layer = dataset
        self.loss_layer = None  # init as net.loss_layer = Loss.CrossEntropyLoss()
        # self.data_loss = None # reg loss in every layer
        self.forward_output = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self._phase = None

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, phase):
        self._phase = phase
        for layer in self.layers:
            layer.testing_phase = phase

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        reg_loss = 0
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if layer.trainable and self.optimizer.regularizer:
                reg_loss += layer.optimizer.regularizer.norm(layer.weights)
        return self.loss_layer.forward(input_tensor, self.label_tensor) + reg_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer: Base.BaseLayer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations): # + regularization
        self.phase = False
        for _ in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        self.phase = True
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor
