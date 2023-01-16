from .Base import *

class Flatten(BaseLayer):
    def forward(self, input_tensor):
        self.shape = input_tensor.shape  
        return input_tensor.reshape((input_tensor.shape[0], -1))

    def backward(self, error_tensor): 
        return error_tensor.reshape(self.shape)
