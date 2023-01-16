from .Base import *
from scipy import signal
import copy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels: int) -> None:
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = np.random.rand(num_kernels, *convolution_shape)
        self.bias = np.random.rand(num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None

    
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._bias_optimizer = copy.deepcopy(optimizer)
    
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output = np.zeros((input_tensor.shape[0], 
                           self.num_kernels, 
                           *input_tensor.shape[2:]))
        for b in range(input_tensor.shape[0]):
            for k in range(self.num_kernels):
                for c in range(input_tensor.shape[1]):
                    output[b, k] += signal.correlate(input_tensor[b, c], self.weights[k, c], mode='same')
                output[b, k] += self.bias[k]
        if len(self.stride_shape) > 1:
            output = output[:, :, ::self.stride_shape[0], ::self.stride_shape[1]]
        else:
            output = output[:, :, ::self.stride_shape[0]]
        return output

    def backward(self, error_tensor):   
        output = np.zeros_like(self.input_tensor)
        upsampled_error = np.zeros((*error_tensor.shape[:2], *self.input_tensor.shape[2:]))
        self._gradient_weights = np.zeros_like(self.weights)
        self._gradient_bias = np.zeros_like(self.bias)

        ##Upsamling if we have striding 
        if len(self.stride_shape) > 1:
            upsampled_error[:,:, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
        else:
            upsampled_error[:,:, ::self.stride_shape[0]] = error_tensor

        # padded_upsampled_error = self.padding(upsampled_error, self.weights)
        padded_input = self.padding(self.input_tensor, self.weights)
        
        for b in range(self.input_tensor.shape[0]):
            for k in range(self.num_kernels):
                for c in range(self.input_tensor.shape[1]):
                    output[b, c] += signal.convolve(upsampled_error[b, k], self.weights[k, c], mode='same')
                    self._gradient_weights[k, c] += signal.correlate(padded_input[b, c], 
                                                                     upsampled_error[b, k], mode='valid')
                self._gradient_bias[k] += np.sum(error_tensor[b, k])
        

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        return output


    def initialize(self, weights_initializer, bias_initializer):
        self.fan_in = np.prod(self.convolution_shape)
        self.fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, self.fan_in, self.fan_out)

    @staticmethod        
    def padding(input_matrix, k):
        flag = (len(k.shape) == 3) #if 1d input then True
        if flag:
            k = k[:, :, np.newaxis, :]
            input_matrix = input_matrix[:, :, np.newaxis, :]

        padding = np.array(k.shape[2:]) - 1
        padding_bottom, padding_right =  np.ceil(padding / 2).astype(int)
        padding_top, padding_left = padding - [padding_bottom, padding_right]

        padded_matrix = np.pad(input_matrix, ((0, 0), (0, 0), (padding_top,padding_bottom), (padding_left,padding_right)))
        
        if flag:
            return np.squeeze(padded_matrix, axis=2)
        return padded_matrix