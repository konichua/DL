from .Base import *
from scipy import signal
# from .Helpers import get_conv_as_matmul, padding, kernel_reshape


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
        self._optimizer = None

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
        output = np.zeros_like(self.input_tensor) # b,c,mn
        upsampled_error = np.zeros((*error_tensor.shape[:2], *self.input_tensor[2:])) # b,c, m,n
        # if len(self.stride_shape) > 1:
        #     upsampled_error[:, :, ::self.stride_shape[0], ::self.stride_shape[1]] = error_tensor
        # else:
        #     upsampled_error[:, :, ::self.stride_shape[0]] = error_tensor

        # print(f'{self.weights.shape=}')
        if self.weights.ndim == 4:
            new_weights = np.transpose(self.weights, (1, 0, 2, 3))
        else:
            new_weights = np.transpose(self.weights, (1, 0, 2))

        for b in range(upsampled_error.shape[0]): # error batch
            for ec in range(upsampled_error.shape[1]): # error channel
                for k in range(new_weights.shape[0]):  # num new kernels (channels)
                    for wc in range(new_weights.shape[1]):  # num new channels
                        pass
                        # output[b,k] += signal.convolve(upsampled_error[b,ec], new_weights[k,wc], mode='same')


        # if self._optimizer:
        #     self.weights = self._optimizer.calculate_update(self.weights, )


        return output

    def initialize(self, weights_initializer, bias_initializer):
        self.fan_in = np.prod(self.convolution_shape)
        self.fan_out = np.prod(self.convolution_shape[1:]) * self.num_kernels
        self.weights = weights_initializer.initialize(self.weights.shape, self.fan_in, self.fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, self.fan_in, self.fan_out)


# if __name__ == '__main__':
# batch_size = 2
# input_shape = (3, 10, 14)
# input_size = 14 * 10 * 3
# kernel_shape = (3, 5, 8)
# num_kernels = 4
#
# conv = Conv((1, 1), kernel_shape, num_kernels)
# input_tensor = np.array(range(np.prod(input_shape) * batch_size), dtype=float)
# input_tensor = input_tensor.reshape(batch_size, *input_shape)
# output_tensor = conv.forward(input_tensor)
# error_tensor = conv.backward(output_tensor)