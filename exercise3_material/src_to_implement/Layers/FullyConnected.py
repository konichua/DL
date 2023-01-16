from .Base import *


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self._gradient_weights = None
        self.input_size = input_size
        self.output_size = output_size
        self._optimizer = None
        self.weights = np.random.rand(input_size + 1, output_size)  # adding bias
        
    @property
    def optimizer(self):
        return self._optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        self.input_tensor = np.c_[input_tensor, np.ones(input_tensor.shape[0])] 
        return self.input_tensor @ self.weights  # (a @ b.T).T = b @ a.T
    
    def backward(self, error_tensor):   
        self._gradient_weights = self.input_tensor.T @ error_tensor
        new_error_tensor = error_tensor @ self.weights.T[:, :-1]
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        return new_error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        # self.weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)
        # bias = bias_initializer.initialize((1, self.output_size), 1, self.output_size)
        self.weights = np.r_[weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size),
                            bias_initializer.initialize((1, self.output_size), 1, self.output_size)]
        

    