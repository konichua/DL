from .Base import *


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.p = probability

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        self.mask = np.random.uniform(size=input_tensor.shape) < self.p
        return input_tensor * self.mask / self.p

    def backward(self, error_tensor):
        return error_tensor * self.mask / self.p