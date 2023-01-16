from .Base import *
from .Helpers import *
import copy


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self._optimizer = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.c = channels
        self.initialize()
        self.decay = 0.8
        self.mean_moving = None
        self.var_moving = None
        self.tensor_shape = None
        self.image_like = False

    def initialize(self):
        self.bias = np.zeros((1, self.c))
        self.weights = np.ones((1, self.c))

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._bias_optimizer = copy.deepcopy(optimizer)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.tensor_shape = input_tensor.shape
        if input_tensor.ndim == 4:
            self.input_tensor = self.reformat(input_tensor)
            self.image_like = True
        if not self.testing_phase:
            mean = self.input_tensor.mean(axis=0)
            var = self.input_tensor.var(axis=0)
        else:
            mean = self.mean_moving
            var = self.var_moving
        if self.mean_moving is None:
            self.mean_moving = mean
            self.var_moving = var
        if not self.testing_phase:
            self.mean_moving = self.decay * self.mean_moving + (1 - self.decay) * mean
            self.var_moving = self.decay * self.var_moving + (1 - self.decay) * var
        output = self.weights * (self.input_tensor - mean) / np.sqrt(var + np.finfo(float).eps) + self.bias
        if self.image_like:
            output = self.reformat(output)
        return output

    def backward(self, error_tensor):
        if self.image_like:
            error_tensor = self.reformat(error_tensor)
        self._gradient_weights = np.sum(error_tensor * (self.input_tensor - self.input_tensor.mean(axis=0)) /
                                    np.sqrt(self.input_tensor.var(axis=0) + np.finfo(float).eps), axis=0, keepdims=True)
        self._gradient_bias = np.sum(error_tensor, axis=0, keepdims=True)
        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)
        output = compute_bn_gradients(error_tensor, self.input_tensor, self.weights,
                                    self.input_tensor.mean(axis=0),
                                    self.input_tensor.var(axis=0))
        if self.image_like:
            output = self.reformat(output)
        return output

    def reformat(self, tensor):
        if tensor.ndim == 4:
            tensor = tensor.reshape(*tensor.shape[:2], tensor.shape[2] * tensor.shape[3])
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2])
        elif tensor.ndim == 2:
            tensor = tensor.reshape(self.tensor_shape[0], np.prod(self.tensor_shape[2:]), self.tensor_shape[1])
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = tensor.reshape(*tensor.shape[:2], self.tensor_shape[2], self.tensor_shape[3])
        return tensor


# batch_size = 5
# channels = 2
# input_shape = (channels, 3, 4)
# input_size = np.prod(input_shape)
#
# np.random.seed(0)
# input_tensor = np.abs(np.random.random((input_size, batch_size))).T
# input_tensor_conv = np.random.uniform(-1, 1, (batch_size, *input_shape))
#
# layer = BatchNormalization(channels)
# layer.forward(input_tensor_conv)

