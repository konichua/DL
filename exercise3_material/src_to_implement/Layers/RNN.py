from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .Sigmoid import Sigmoid
from .TanH import TanH
import numpy as np


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.hidden_state = np.zeros((1, hidden_size))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memorize = False
        self.fc_input = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)
        self.tanh = TanH()
        self.sigm = Sigmoid()

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_input.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        output = np.zeros((input_tensor.shape[0], self.output_size))
        for b in range(input_tensor.shape[0]):
            state = self.fc_input.forward(np.hstack((input_tensor[b, None], self.hidden_state)))
            self.hidden_state = self.tanh.forward(state)
            state = self.fc_output.forward(self.hidden_state)
            output[b] = self.sigm.forward(state)
        return output

