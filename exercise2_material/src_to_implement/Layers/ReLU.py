from .Base import *


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        # return max(0, x)
        self.input_tensor = input_tensor
        return abs(input_tensor) * (input_tensor > 0)
        # return input_tensor[input_tensor < 0] = 0

    def backward(self, error_tensor):
        # return e if input > 0
        return error_tensor * np.ones_like(self.input_tensor) * (self.input_tensor > 0)


