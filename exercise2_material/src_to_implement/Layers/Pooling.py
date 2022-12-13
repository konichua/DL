from .Base import *

# only for 2D case
class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        self.input_pooled_indexes = None

    def find_index(self, arr, el):
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] == el:
                    return i, j

    def forward(self, input_tensor):
        result = np.zeros((*input_tensor.shape[:2],
                           np.ceil((input_tensor.shape[2] - self.pooling_shape[0] + 1) / self.stride_shape[0]).astype(int),
                           np.ceil((input_tensor.shape[3] - self.pooling_shape[1] + 1) / self.stride_shape[1]).astype(int)))
        self.input_shape = input_tensor.shape
        self.input_pooled_indexes = np.zeros((*result.shape, 2))
        for b in range(input_tensor.shape[0]):
            for c in range(input_tensor.shape[1]):
                input_pooled = []
                input_pooled_indexes = []
                for i in range(input_tensor.shape[2] - self.pooling_shape[0] + 1):
                    row = []
                    row_indexes = []
                    for j in range(input_tensor.shape[3] - self.pooling_shape[1] + 1):
                        el = input_tensor[b, c, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]].max()
                        row.append(el)
                        k, l = self.find_index(input_tensor[b, c, i:i+self.pooling_shape[0], j:j+self.pooling_shape[1]], el)
                        row_indexes.append([k + i, l + j])
                    input_pooled.append(row)
                    input_pooled_indexes.append(row_indexes)
                result[b, c] = np.asarray(input_pooled)[::self.stride_shape[0], ::self.stride_shape[1]]
                self.input_pooled_indexes[b, c] = np.asarray(input_pooled_indexes)[::self.stride_shape[0], ::self.stride_shape[1]]
        return result

    def backward(self, error_tensor):
        output = np.zeros(self.input_shape)
        for b in range(error_tensor.shape[0]):
            for c in range(error_tensor.shape[1]):
                for i in range(error_tensor.shape[2]):
                    for j in range(error_tensor.shape[3]):
                        ind_i, ind_j = self.input_pooled_indexes[b, c, i, j]
                        output[b, c, int(ind_i), int(ind_j)] += error_tensor[b, c, i, j]
        return output
