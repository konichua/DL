from .Base import BaseLayer
from .FullyConnected import FullyConnected
from .Sigmoid import Sigmoid
from .TanH import TanH
import numpy as np
from copy import deepcopy


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.hidden_state = [np.zeros((1, hidden_size))]
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.memorize = False
        self.fc_input = FullyConnected(input_size + hidden_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)
        self.tanh = TanH()
        self.sigm = Sigmoid()
        self._weights = None
        self._gradient_weights = None
        self._optimizer = None
        self.input_tensor = None
        self.input_save = None
        self.output_save = None

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_output = deepcopy(optimizer)


    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def weights(self):
        return self.fc_input.weights

    @weights.setter
    def weights(self, weights):
        self.fc_input.weights = weights

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_input.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        if not self.memorize:
            self.hidden_state[-1] = np.zeros((1, self.hidden_size))

        output = np.zeros((input_tensor.shape[0], self.output_size))
        for b in range(input_tensor.shape[0]):
            state = self.fc_input.forward(np.hstack((input_tensor[b, None], self.hidden_state[-1])))
            self.hidden_state.append(self.tanh.forward(state))
            state = self.fc_output.forward(self.hidden_state[-1])
            output[b] = self.sigm.forward(state)
        return output

    def backward(self, error_tensor):
        new_error = np.zeros_like(self.input_tensor)

        hidden_err = np.zeros((1, self.hidden_size))
        gradient_weights_output = np.zeros_like(self.fc_output.weights)
        self._gradient_weights = np.zeros_like(self.fc_input.weights)
        for idx, err in reversed(list(enumerate(error_tensor))):
            fc1_output = self.fc_input.forward(np.hstack((self.input_tensor[idx, None], self.hidden_state[idx])))
            hidden_output = self.tanh.forward(fc1_output)
            fc2_output = self.fc_output.forward(hidden_output)
            self.sigm.forward(fc2_output)
            ### BACKWARD ###
            down_grad = self.sigm.backward(err)
            down_grad = self.fc_output.backward(down_grad) + hidden_err
            down_grad = self.tanh.backward(down_grad)
            down_grad = self.fc_input.backward(down_grad)
            new_error[idx] = down_grad[:, :self.input_size]
            hidden_err = down_grad[:, self.input_size:]
            gradient_weights_output += self.fc_output.gradient_weights
            self._gradient_weights += self.fc_input.gradient_weights
        if self._optimizer:
            self.fc_input.weights = self._optimizer.calculate_update(self.fc_input.weights, self._gradient_weights)
            self.fc_output.weights = self._optimizer_output.calculate_update(self.fc_output.weights, gradient_weights_output)
        return new_error
