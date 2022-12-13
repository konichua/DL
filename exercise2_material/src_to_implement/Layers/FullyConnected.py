from .Base import *
# from Optimization.Optimizers import Sgd


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self._optimizer = None
        self._gradient_weights = None
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size + 1, output_size)
        # self.weights = None

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.vstack((weights, bias))

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    def forward(self, input_tensor):
        # add bias to X
        self.input_tensor = np.hstack((input_tensor, np.ones(input_tensor.shape[0]).reshape(input_tensor.shape[0], 1)))  # X
        return self.input_tensor @ self.weights

    def backward(self, error_tensor):
        self._gradient_weights = self.input_tensor.T @ error_tensor
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return error_tensor @ self.weights[:-1, :].T


